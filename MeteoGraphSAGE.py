import os
import math
import argparse
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv, GINConv
from torch_geometric.utils import subgraph
from typing import List, Iterator, Optional, Tuple
from copy import deepcopy
# Opcional: matplotlib per a gràfiques (es pot instal·lar al cluster si cal)
import matplotlib.pyplot as plt
# Opcional: per exportar a NetCDF
from netCDF4 import Dataset
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from collections import deque
import sys


# Configuració del logger per mostrar informació durant l'entrenament i predicció
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Silenciar només el FutureWarning de torch.load(weights_only)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*torch\\.load.*weights_only.*"
)

# Sempre escriure la barra encara que no sigui TTY
tqdm_kwargs = dict(file=sys.stderr)

# Noms de característiques (features) tal com es van utilitzar en el preprocessament
FEATURE_NAMES = ['Temp', 'Humitat', 'Pluja', 'VentFor', 'Patm', 'Alt_norm',
                 'VentDir_sin', 'VentDir_cos', 'hora_sin', 'hora_cos',
                 'dia_sin', 'dia_cos', 'cos_sza', 'DewPoint', 'PotentialTemp']

logger = logging.getLogger(__name__)

# —— Reproducibilitat ——————————————————————————————————————————————————————
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# Opcional: fer CuDNN determinista (pot alentir lleugerament)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def create_sequences(
    data_dir: str,
    period: str = "day",
    window_size: Optional[int] = None,
    stride: int = 1,
    min_seq_len: int = 2,
    lazy: bool = False
) -> Iterator[dict]:
    """
    Genera seqüències temporals normalitzades de Data objects, totes amb el mateix 
    ordre i nombre de nodes. Cada seqüència és un dict amb:
      - 'data': List[Data]  … grafs remapejats i padded
      - 'mask': Tensor       … [seq_len, N] on N = num_total_nodes (1=presència,0=pad)
      - 'ids': List[id]      … llista d'ids canònics de longitud N
    Paràmetres:
      data_dir: Directori amb fitxers pt i/o amb agrupacions 'sequence_*pt'.
      period: "day"|"month" (només si window_size és None).
      window_size: longitud fixa de finestra (si None, agrupa per period).
      stride: salt entre finestra i finestra.
      min_seq_len: seqüències amb menys passos es descarten.
      lazy: si True, retorna un generator que carrega els pt quan cal.
    """
    # 1) Llista tots els fitxers pt
    seq_files = sorted(f for f in os.listdir(data_dir) if f.startswith("sequence_") and f.endswith("pt"))
    if seq_files:
        file_paths = [os.path.join(data_dir, f) for f in seq_files]
    else:
        all_files = []
        for root, _, files in os.walk(data_dir):
            for fn in files:
                if fn.endswith("pt") and not fn.startswith(("sequence_","group_")):
                    all_files.append(os.path.join(root, fn))
        file_paths = sorted(all_files)

    # 2) Scan ids
    def scan_ids(fp_list):
        ids_map, times = [], []
        for fp in tqdm(fp_list, desc="[scan_ids] fitxers", unit="fitxer"):
            d = torch.load(fp) if not lazy else torch.load(fp, map_location="cpu")
            ids_map.append(list(d.ids))
            times.append(getattr(d, "timestamp", None))
        return ids_map, times

    ids_list, timestamps = scan_ids(file_paths)

    # 3) Canonicalitzar ids
    all_ids = sorted({i for ids in ids_list for i in ids})
    id2idx = {i: k for k, i in enumerate(all_ids)}
    N = len(all_ids)

    # 4) Funció remap_and_pad
    def remap_and_pad(data: Data) -> Data:
        feat_dim = data.x.size(1)
        x = torch.zeros((N, feat_dim), dtype=data.x.dtype)
        mask = torch.zeros(N, dtype=torch.bool)
        old_ids = list(data.ids)
        for old_idx, node_id in enumerate(old_ids):
            new_idx = id2idx[node_id]
            x[new_idx] = data.x[old_idx]
            mask[new_idx] = True
        # Remap pos
        pos = None
        if hasattr(data, 'pos'):
            pos_dim = data.pos.size(1)
            pos = torch.zeros((N, pos_dim), dtype=data.pos.dtype)
            for old_idx, node_id in enumerate(old_ids):
                pos[id2idx[node_id]] = data.pos[old_idx]
        # Remap edges
        remapped_edges, remapped_attrs = [], []
        has_attr = hasattr(data, 'edge_attr') and data.edge_attr is not None
        for ei, (u, v) in enumerate(data.edge_index.t().tolist()):
            ui, vi = id2idx[old_ids[u]], id2idx[old_ids[v]]
            remapped_edges.append([ui, vi])
            if has_attr:
                remapped_attrs.append(data.edge_attr[ei])
        edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(remapped_attrs) if has_attr else None
        new_data = Data(x=x, edge_index=edge_index)
        new_data.mask = mask
        new_data.ids = all_ids
        if pos is not None: new_data.pos = pos
        if edge_attr is not None: new_data.edge_attr = edge_attr
        return new_data

    L = len(file_paths)

    # 5) Sliding window o grup per period
    if window_size:
        step = stride if stride and stride > 0 else window_size
        buf = deque()
        # Pre-carrega
        for fp in tqdm(file_paths[:window_size], desc="[create_sequences] preload", unit="fitxer", disable=(rank!=0), **tqdm_kwargs):
            d = torch.load(fp, map_location='cpu') if lazy else torch.load(fp)
            buf.append(remap_and_pad(d))
        if len(buf) == window_size:
            yield {"data": list(buf), "ids": all_ids, "mask": torch.stack([d.mask for d in buf], dim=0)}
        # Finestra mòbil
        for idx in tqdm(range(window_size, L, step), desc="[create_sequences] seqüències", unit="pas", disable=(rank!=0), **tqdm_kwargs):
            for _ in range(min(step, len(buf))): buf.popleft()
            for fp in file_paths[idx: min(idx+step, L)]:
                d = torch.load(fp, map_location='cpu') if lazy else torch.load(fp)
                buf.append(remap_and_pad(d))
            if len(buf) == window_size:
                yield {"data": list(buf), "ids": all_ids, "mask": torch.stack([d.mask for d in buf], dim=0)}
    else:
        # Agrupació dia/mes
        from collections import defaultdict
        grp = defaultdict(list)
        for i, ts in enumerate(timestamps):
            key = ts.split()[0] if ts is not None and period == 'day' else ts[:7] if ts is not None else i
            grp[key].append(i)
        indices = []
        for seq in grp.values():
            if len(seq) >= min_seq_len:
                for start in range(0, len(seq) - min_seq_len + 1, stride):
                    indices.append(seq[start:start + min_seq_len])
        for idxs in tqdm(indices, desc="[create_sequences] seqüències", unit="seqüència", disable=(rank!=0), **tqdm_kwargs):
            seq_data = []
            for i in idxs:
                fp = file_paths[i]
                d = torch.load(fp) if not lazy else torch.load(fp, map_location='cpu')
                seq_data.append(remap_and_pad(d))
            yield {"data": seq_data, "ids": all_ids, "mask": torch.stack([d.mask for d in seq_data], dim=0)}


class MeteoGraphSAGEEnhanced(nn.Module):
    """
    Model de predicció meteorològica amb millores espacials i temporals.
    - Agregació espacial configurable: GraphSAGE, GAT o GIN.
    - Arquitectura temporal flexible: estàtica, RNN (LSTM/GRU) o EvolveGCN (capes GNN dinàmiques).
    - Embeddings d'estació dinàmics o basats en ID.
    """
    def __init__(self, in_features, hidden_dim, out_features, 
                 aggregator: str = 'gat',       # Tipus d'agregador: 'sage', 'gat' o 'gin'
                 temporal_model: str = 'EvolveGCN',  # 'none', 'LSTM', 'GRU', 'EvolveGCN', 'Transformer'
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.2,
                 station_embedding_dim: int = 0, num_stations: int = None):
        super().__init__()
        # Guardem paràmetres
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.aggregator = aggregator.lower()
        self.temporal_model = temporal_model.lower()
        self.num_layers = num_layers
        self.num_heads = num_heads if self.aggregator == 'gat' else 1  # caps d'atenció (només GAT)
        
        # Embedding d'estació (opcional, per capturar microclima local de cada estació)
        if station_embedding_dim > 0 and num_stations is not None:
            self.station_embedding = nn.Embedding(num_stations, station_embedding_dim)
        else:
            self.station_embedding = None
        
        # Calcul de la dimensió d'entrada de la primera capa (afegint embedding estació si escau)
        input_dim = in_features + (station_embedding_dim if self.station_embedding is not None else 0)
        # Projecció inicial a hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Definició de les capes de convolució de graf segons l'agregador seleccionat
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        if self.aggregator == 'sage':
            # GraphSAGE amb agregació de mitjana (baseline)
            for _ in range(num_layers):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        elif self.aggregator == 'gat':
            # Graph Attention Network: usem concat=False per mantenir dimensió hidden_dim
            for _ in range(num_layers):
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=0.1))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        elif self.aggregator == 'gin':
            # Graph Isomorphism Network: definim una MLP simple per a l'agregació
            for _ in range(num_layers):
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                # train_eps=True permet a GIN aprendre un pes residual epsilon
                conv = GINConv(nn=mlp, train_eps=True)
                self.convs.append(conv)
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        else:
            raise ValueError(f"Aggregador desconegut: {aggregator}. Triar entre 'sage', 'gat', 'gin'.")
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Mòdul temporal: RNN (LSTM/GRU) o Transformer o EvolveGCN
        self.rnn = None
        self.transformer_encoder = None
        self.evolve_gcns = None
        if self.temporal_model in ['lstm', 'gru']:
            rnn_hidden_dim = hidden_dim
            if self.temporal_model == 'lstm':
                self.rnn = nn.LSTM(hidden_dim, rnn_hidden_dim, batch_first=False)
            elif self.temporal_model == 'gru':
                self.rnn = nn.GRU(hidden_dim, rnn_hidden_dim, batch_first=False)
        elif self.temporal_model == 'transformer':
            # Codificació posicional sinusoidal per a seqüències temporals
            self.max_seq_len = 1000
            pe = torch.zeros(self.max_seq_len, hidden_dim)
            position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float) * (-math.log(10000.0) / hidden_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('positional_encoding', pe)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, 
                                                      dim_feedforward=hidden_dim*2, dropout=dropout, batch_first=False)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        elif self.temporal_model == 'evolvegcn':
            # EvolveGCN: utilitzem una GRUCell per evolupar els paràmetres de cada capa GNN
            # Inicialitzem l'estat ocult de cada capa GNN com els seus pesos vectoritzats
            self.evolve_gcns = nn.ModuleList()
            # Guardem els pesos inicials de cada capa (com a tensor pla)
            self.init_weights = []
            for conv in self.convs:
                flat = torch.cat([p.data.view(-1) for p in conv.parameters()], dim=0)
                self.init_weights.append(flat.clone())
            for conv in self.convs:
                # dimensionem hidden_size com nombre de paràmetres de la capa conv
                num_params = 0
                for p in conv.parameters():
                    num_params += p.numel()
                # GRUCell que pren com input un vector de longitud hidden_dim (p.ex. embedding mitjà) i genera nou pes (hidden_size = num_params)
                self.evolve_gcns.append(nn.GRUCell(hidden_dim, num_params))
            # Atribut per emmagatzemar l'estat dels pesos evolutius de cada capa (inicialment els pesos inicials aplanats)
            # Calculem la llargada màxima de vector de paràmetres
            max_len = max(w.numel() for w in self.init_weights)

            # Creem el tensor d'estat inicial a partir dels pesos inicials
            hidden = torch.zeros(len(self.convs), max_len, dtype=torch.float)
            for i, w in enumerate(self.init_weights):
                hidden[i, :w.numel()] = w  # copiem els primers numel(w) valors

            # Registram el buffer amb l'estat inicialitzat als pesos del model
            self.register_buffer('evolve_hidden_state', hidden)
        # Mecanisme d'atenció temporal multi-cap (només aplicable combinat amb RNN)
        self.attention = None
        if self.temporal_model in ['lstm', 'gru']:
            # Afegir atenció multi-head sobre la seqüència temporal de sortides del RNN
            self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=False)
        
        # Capa de sortida (decodificador)
        # Si hi ha atenció temporal, la sortida combina hidden state + context d'atenció -> dimensió 2*hidden_dim
        decoder_in_dim = hidden_dim * 2 if self.attention is not None else hidden_dim
        self.decoder = nn.Linear(decoder_in_dim, out_features)
        
    def forward_graph(self, data):
        """Aplica la/les capes GNN espacials (GraphSAGE/GAT/GIN) a un graf individual i retorna embeddings nodals."""
        x, edge_index = data.x, data.edge_index
        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)

        # Embedding d'estació opcional
        if self.station_embedding is not None and hasattr(data, 'station_idx'):
            station_idx = torch.tensor(data.station_idx, dtype=torch.long, device=device)
            station_emb = self.station_embedding(station_idx)
            x = torch.cat([x, station_emb], dim=1)

        # Projecció inicial
        h = self.input_proj(x)

        # Propagació per les capes de graf
        for i, conv in enumerate(self.convs):
            if self.temporal_model == 'evolvegcn':
                # 1) Estat ocult previ i característiques resum
                avg_feature = h.mean(dim=0, keepdim=True)  # (1, hidden_dim)
                prev_weights = self.evolve_hidden_state[i, :self.evolve_gcns[i].hidden_size].unsqueeze(0)

                # 2) Calculem l'offset amb GRUCell
                offset_flat = self.evolve_gcns[i](avg_feature, prev_weights)  # (1, num_params_i)

                # 3) Actualitzem l'estat ocult
                self.evolve_hidden_state[i, :offset_flat.size(1)] = offset_flat.squeeze(0)

                # 4) Combine with initial weights
                init_flat = self.init_weights[i].to(offset_flat.device).unsqueeze(0)
                new_weight_flat = init_flat + offset_flat  # (1, num_params_i)

                # 5) Assignem els nous valors sense canviar l'objecte Parameter
                idx = 0
                for param in conv.parameters():
                    numel = param.numel()
                    seg = new_weight_flat[0, idx:idx + numel].view_as(param)
                    param.data.copy_(seg)
                    idx += numel

                # 6) Propaguem amb la capa actualitzada
                h_new = conv(h, edge_index)
            else:
                # Propagació normal
                h_new = conv(h, edge_index)

            # Post-processament: BN -> ReLU -> Dropout
            h_new = self.batch_norms[i](h_new)
            h_new = self.activation(h_new)
            h_new = self.dropout(h_new)

            # Residual
            h = h + h_new

        return h  # tensor de mida [num_nodes, hidden_dim]

    
    def forward(self, data_sequence):
        """
        Execució endavant per a una seqüència temporal de grafs (data_sequence pot ser llista de Data o un sol Data).
        Retorna les prediccions per als nodes del darrer graf de la seqüència.
        """
        # Si es passa un únic graf, el convertim en llista
        if isinstance(data_sequence, torch_geometric.data.Data):
            data_sequence = [data_sequence]
        seq_len = len(data_sequence)
        device = next(self.parameters()).device
        
        # Garantim que tots els grafs tenen els mateixos nodes si es fa servir un model estàtic (no EvolveGCN)
        if self.temporal_model != 'evolvegcn':
            # Identifiquem IDs comuns
            common_ids = None
            for data in data_sequence:
                ids = list(data.ids) if hasattr(data, 'ids') else list(range(data.x.size(0)))
                id_set = set(ids)
                common_ids = id_set if common_ids is None else common_ids & id_set
            common_ids = sorted(list(common_ids)) if common_ids is not None else []
            if common_ids and any(data.x.size(0) != len(common_ids) for data in data_sequence):
                # Filtrar cada graf perquè només contingui els nodes comuns
                new_seq = []
                for data in data_sequence:
                    # indices a conservar
                    if hasattr(data, 'ids'):
                        idx_keep = [i for i, id_val in enumerate(data.ids) if id_val in common_ids]
                    else:
                        idx_keep = list(range(len(common_ids)))
                    idx_keep_tensor = torch.tensor(idx_keep, dtype=torch.long)
                    # Creem subgraf amb aquests nodes
                    new_edge_index, new_edge_attr = subgraph(idx_keep_tensor, 
                                              data.edge_index, getattr(data, 'edge_attr', None), relabel_nodes=True)
                    new_x = data.x[idx_keep_tensor]
                    new_data = torch_geometric.data.Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
                    if hasattr(data, 'ids'):
                        new_data.ids = [data.ids[i] for i in idx_keep]
                    if hasattr(data, 'station_idx'):
                        new_data.station_idx = [data.station_idx[i] for i in idx_keep]
                    if hasattr(data, 'timestamp'):
                        new_data.timestamp = data.timestamp
                    new_seq.append(new_data)
                data_sequence = new_seq
                seq_len = len(data_sequence)
        
        # Llista per emmagatzemar els embeddings de nodes a cada pas
        node_embeddings_seq = []
        for t, data in enumerate(data_sequence):
            # Obtenim embeddings espacials per al pas t
            h_t = self.forward_graph(data)
            node_embeddings_seq.append(h_t)
        # Tensor de mida [seq_len, num_nodes, hidden_dim]
        node_embeddings_seq = torch.stack(node_embeddings_seq, dim=0).to(device)
        
        # Si no hi ha model temporal, prenem directament l'últim embedding
        if self.temporal_model in ['none', '', None]:
            final_emb = node_embeddings_seq[-1]  # darrer pas temporal
            preds = self.decoder(final_emb)
            return preds  # [num_nodes, out_features]
        
        # Si hi ha RNN temporal (LSTM/GRU)
        if self.rnn is not None:
            rnn_out, _ = self.rnn(node_embeddings_seq)  # sortida de mida [seq_len, num_nodes, hidden_dim]
            last_out = rnn_out[-1]  # últim pas (per cada node)
            if self.attention is not None:
                # Atenció multi-cap sobre totes les sortides temporals del RNN
                query = last_out.unsqueeze(0)   # shape (1, num_nodes, hidden_dim)
                key = value = rnn_out           # shape (seq_len, num_nodes, hidden_dim)
                attn_output, attn_weights = self.attention(query, key, value)
                context = attn_output.squeeze(0)   # [num_nodes, hidden_dim]
                # concatenem l'últim hidden state amb el context atencional
                last_out = torch.cat([last_out, context], dim=1)  # [num_nodes, 2*hidden_dim]
            preds = self.decoder(last_out)  # [num_nodes, out_features]
            return preds
        
        # Si hi ha Transformer temporal
        if self.transformer_encoder is not None:
            seq_len, num_nodes, feat_dim = node_embeddings_seq.shape
            # Afegim encoding posicional a la seqüència
            if seq_len > self.max_seq_len:
                # si la seqüència és molt llarga, calculem posicional al vol
                position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, feat_dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / feat_dim))
                pos_enc = torch.zeros(seq_len, feat_dim, device=device)
                pos_enc[:, 0::2] = torch.sin(position * div_term)
                pos_enc[:, 1::2] = torch.cos(position * div_term)
            else:
                pos_enc = self.positional_encoding[:seq_len, :].to(device)
            node_embeddings_seq = node_embeddings_seq + pos_enc.unsqueeze(1)  # afegim pos enc a cada node
            transformer_out = self.transformer_encoder(node_embeddings_seq)  # [seq_len, num_nodes, hidden_dim]
            last_out = transformer_out[-1]  # [num_nodes, hidden_dim]
            preds = self.decoder(last_out)
            return preds
        
        # Si hi ha EvolveGCN: ja hem anat evolucionant la GNN durant forward_graph a cada pas,
        # per tant la seqüència node_embeddings_seq incorpora ja l'evolució. Només cal predir del darrer.
        final_emb = node_embeddings_seq[-1]  # [num_nodes, hidden_dim]
        preds = self.decoder(final_emb)
        return preds

# Funcions d'entrenament i avaluació actualitzades
def train_model(
    model: MeteoGraphSAGEEnhanced,
    train_data: list[dict],
    val_data: list[dict] = None,
    epochs: int = 50,
    learning_rate: float = 0.001,
    target_idx: int = 0
):
    """
    Entrena el model sobre seqüències de grafs. 
    train_data i val_data són llistes de dict amb claus:
      - 'data': List[Data]
      - 'mask': Tensor [seq_len, N]
      - 'ids' : List[id]  (opcional, si el model no fa servir station_idx)
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    logger = logging.getLogger(__name__)
    model.train()
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        # Barregem seqüències
        np.random.shuffle(train_data)
        total_loss = 0.0

        for seq_dict in train_data:
            seq = seq_dict['data']         # ara és List[Data]
            mask = seq_dict.get('mask')    # opcional
            if len(seq) < 2:
                continue

            # Preparem input i target
            input_seq   = seq[:-1]
            target_graph = seq[-1]

            # Forward
            preds = model(input_seq)

            # True values
            if model.out_features == 1:
                # No aplanem: mantenim la dimensió de feature com a 1
                pred_vals = preds                                     # [N, 1]
                true_vals = target_graph.x[:, target_idx].unsqueeze(1).to(device)  # [N, 1]
            else:
                pred_vals = preds                                     # [N, F]
                true_vals = target_graph.x.to(device)                 # [N, F]


            # Pèrdua amb màscara (si existeix)
            if mask is not None:
                # mask: [N], errors: [N, F]  → fem broadcast a [N, F]
                node_mask = mask[-1].to(device).float().unsqueeze(1)  # ara [N,1]
                errors = (pred_vals - true_vals)**2                  # [N, F]
                masked_errors = errors * node_mask                   # [N, F], zeros on padded nodes
                # sumar sobre tots dos eixos i dividir pel nombre real d'elements (nodes actius × features)
                n_active = node_mask.sum() * pred_vals.size(1) + 1e-6
                loss = masked_errors.sum() / n_active
            else:
                loss = criterion(pred_vals, true_vals)


            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Historial i logging entrenament
        avg_loss = total_loss / len(train_data) if train_data else 0.0
        history['train_loss'].append(avg_loss)

        # Validació
        val_loss = 0.0
        if val_data:
            model.eval()
            with torch.no_grad():
                for seq_dict in val_data:
                    seq = seq_dict['data']
                    mask = seq_dict.get('mask')
                    if len(seq) < 2:
                        continue
                    input_seq    = seq[:-1]
                    target_graph = seq[-1]
                    preds = model(input_seq)

                    # 1) Preparem pred_vals i true_vals mantenint sempre [N, F]
                    if model.out_features == 1:
                        pred_vals = preds                                    # [N,1]
                        true_vals = target_graph.x[:, target_idx].unsqueeze(1).to(device)  # [N,1]
                    else:
                        pred_vals = preds                                    # [N,F]
                        true_vals = target_graph.x.to(device)                # [N,F]

                    # 2) Pèrdua amb màscara
                    if mask is not None:
                        node_mask = mask[-1].to(device).float().unsqueeze(1)  # [N,1]
                        errors = (pred_vals - true_vals)**2                  # [N,F]
                        masked_errors = errors * node_mask                   # [N,F]
                        n_active = node_mask.sum() * pred_vals.size(1) + 1e-6
                        loss_val = masked_errors.sum() / n_active
                    else:
                        loss_val = criterion(pred_vals, true_vals)

                    val_loss += loss_val.item()
            val_loss = val_loss / len(val_data) if val_data else 0.0
            history['val_loss'].append(val_loss)
            model.train()

        logger.info(f"Època {epoch}/{epochs} - entr.: {avg_loss:.4f}, val.: {val_loss:.4f}")

    return history

def evaluate_model(model: MeteoGraphSAGEEnhanced, test_data: list, target_idx: int = 0):
    """
    Avalua el model amb seqüències de test.
    Retorna un dict amb:
      - 'MSE': scalar o llista de MSE per variable
      - 'RMSE': scalar o llista de RMSE per variable
    """
    device = next(model.parameters()).device
    model.eval()
    total_mse = None
    count = 0

    with torch.no_grad():
        for seq in test_data:
            if len(seq) < 2:
                continue
            input_seq = seq[:-1]
            target_graph = seq[-1]
            preds = model(input_seq)  # [N] o [N, F]

            if model.out_features == 1:
                true_vals = target_graph.x[:, target_idx].to(device)       # [N]
                pred_vals = preds.view(-1)                                 # [N]
                errors = (pred_vals - true_vals) ** 2                      # [N]
                mse_sample = errors.mean()                                # scalar
            else:
                true_vals = target_graph.x.to(device)                     # [N, F]
                pred_vals = preds                                         # [N, F]
                errors = (pred_vals - true_vals) ** 2                     # [N, F]
                # MSE per variable: mean sobre l’eix dels nodes
                mse_sample = errors.mean(dim=0)                          # [F]

            # Acumulem
            if total_mse is None:
                total_mse = mse_sample
            else:
                total_mse = total_mse + mse_sample
            count += 1

    if count == 0:
        return {"MSE": 0.0, "RMSE": 0.0}

    # Mitjana de totes les mostres
    avg_mse = total_mse / count

    # RMSE: sqrt(MSE)
    if isinstance(avg_mse, torch.Tensor):
        rmse = torch.sqrt(avg_mse)
        mse_out = avg_mse.tolist()
        rmse_out = rmse.tolist()
    else:
        rmse = math.sqrt(avg_mse.item())
        mse_out = avg_mse.item()
        rmse_out = rmse

    print(f"Resultats de Test - MSE: {mse_out}, RMSE: {rmse_out}")
    return {"MSE": mse_out, "RMSE": rmse_out}

def predict_for_station(model: MeteoGraphSAGEEnhanced, data_sequence: list, station_id: int,
                        target_idx: int = 0, horizon: int = 1):
    """
    Realitza prediccions iteratives (autoregressives) per a una estació individual donada (station_id) 
    a partir d'una seqüència històrica. Retorna una llista amb les prediccions successives fins a 'horizon' passos endavant.
    """
    device = next(model.parameters()).device
    model.eval()
    preds_list = []
    # Còpia de la seqüència per no modificar l'original
    seq_copy = [deepcopy(data) for data in data_sequence]
    # Ens assegurem que la seqüència està al device correcte
    for data in seq_copy:
        data.to(device)
    current_sequence = seq_copy
    # Trobar l'índex del node corresponent a l'estació dins de l'últim graf de la seqüència
    station_index = None
    if hasattr(current_sequence[-1], 'ids'):
        if station_id in current_sequence[-1].ids:
            station_index = current_sequence[-1].ids.index(station_id)
    else:
        station_index = station_id if station_id < current_sequence[-1].x.size(0) else None
    if station_index is None:
        logging.warning(f"L'estació {station_id} no es troba en l'últim graf de la seqüència. No es pot fer predicció per aquesta estació.")
        return preds_list
    with torch.no_grad():
        for h in range(horizon):
            # Predir el següent pas temporal amb el model
            preds = model(current_sequence)
            if model.out_features == 1:
                pred_value = preds.view(-1)[station_index].item()
            else:
                pred_value = preds[station_index, target_idx].item()
            preds_list.append(pred_value)
            # Actualitzar la seqüència afegint el nou pas predit
            last_graph = current_sequence[-1]
            new_graph = last_graph.clone()
            # Substituïm la variable objectiu de l'estació pel valor predit
            if model.out_features == 1:
                new_graph.x[station_index, target_idx] = pred_value
            else:
                new_graph.x[station_index, target_idx] = pred_value
            # Afegim el nou graf al final de la seqüència (podem opcionalment treure el primer per mantenir finestra fixa)
            current_sequence.append(new_graph)
            # (Opcional) Eliminar current_sequence[0] per mantenir la mida de finestra constant
    return preds_list

def predict_region_to_netcdf(
    model: MeteoGraphSAGEEnhanced,
    data_sequence: list,
    grid_res: float = 0.1,
    target_idx: int = 0,
    file_path: str = "prediction.nc"
):
    """
    Prediu la variable objectiu al proper pas temporal per a totes les estacions,
    i genera un mapa regional interpolat en format NetCDF.

    Args:
        model: instància de MeteoGraphSAGEEnhanced ja entrenat.
        data_sequence: llista de Data objects ordenats cronològicament.
        grid_res: resolució de la graella en graus (lat/lon).
        target_idx: índex de la variable objectiu dins de data.x (quan out_features>1).
        file_path: ruta de sortida del fitxer NetCDF.
    """
    if not data_sequence:
        logging.error("Seqüència buida: no es pot generar predicció.")
        return

    device = next(model.parameters()).device
    model.eval()

    # 1) Portar tots els Data al mateix device
    seq = []
    for data in data_sequence:
        d = deepcopy(data)  # per no mutar l'original
        # Always move x and edge_index
        d.x = d.x.to(device)
        d.edge_index = d.edge_index.to(device)
        # Only move edge_attr if it exists
        if hasattr(d, 'edge_attr') and d.edge_attr is not None:
            d.edge_attr = d.edge_attr.to(device)
        # Only move pos if it exists
        if hasattr(d, 'pos') and d.pos is not None:
            d.pos = d.pos.to(device)
        seq.append(d)


    # 2) Calcular la predicció per al darrer pas
    with torch.no_grad():
        preds = model(seq).cpu().numpy()

    last = seq[-1]
    if not hasattr(last, 'edge_attr'): logging.warning("No edge_attr: s'usarà graella sense atributs")
    if last.pos.size(1) < 2: raise ValueError("pos ha de tenir almenys lat i lon")
    coords = last.pos[:, :2].cpu().numpy()   # array Nx2: [lat, lon]
    N = coords.shape[0]

    # 3) Extreure vector de prediccions
    if model.out_features == 1:
        values = preds.reshape(-1)
    else:
        values = preds[:, target_idx]

    # 4) Desnormalització si convé
    var_name = FEATURE_NAMES[target_idx]
    if hasattr(last, "norm_params") and var_name in last.norm_params:
        mean = last.norm_params[var_name]["mean"]
        std = last.norm_params[var_name]["std"]
        values = values * std + mean
        # Invertir escales específiques
        if var_name in ("Temp", "DewPoint", "PotentialTemp"):
            values -= 273.15
        elif var_name == "Humitat":
            values *= 100.0
        elif var_name == "Patm":
            values += 1013.0
        elif var_name == "Pluja":
            values = np.expm1(values)

    # 5) Definir graella regular
    min_lat, min_lon = coords.min(axis=0)
    max_lat, max_lon = coords.max(axis=0)
    lat_grid = np.arange(min_lat, max_lat + 1e-9, grid_res)
    lon_grid = np.arange(min_lon, max_lon + 1e-9, grid_res)
    nl, nm = lat_grid.size, lon_grid.size

    # 6) Vectoritzar interpolació IDW
    # Meshgrid de punts: shape (nl*nm, 2)
    mesh_lon, mesh_lat = np.meshgrid(lon_grid, lat_grid)
    grid_pts = np.column_stack([mesh_lat.ravel(), mesh_lon.ravel()])  # (G,2), G=nl*nm

    # Calcular distàncies Euclidianes aproximades en km
    # dlat ~ 111 km/deg, dlon ~ 111*cos(lat)/deg
    cos_lat = np.cos(np.deg2rad(grid_pts[:, 0]))  # (G,)
    dlat = (coords[:, 0:1] - grid_pts[None, :, 0]) * 111.0   # (N, G)
    dlon = (coords[:, 1:2] - grid_pts[None, :, 1]) * 111.0 * cos_lat[None, :]
    dists = np.hypot(dlat, dlon)  # (N, G)

    eps = 1e-6
    # Pesos IDW
    weights = 1.0 / (dists + eps)         # (N, G)
    # Casos on la distància és quasi zero
    zero_mask = dists < 1e-6              # (N, G)
    if zero_mask.any():
        # Assignar pes 1 al node exacte i 0 la resta
        weights[:] = 0.0
        weights[zero_mask] = 1.0

    # Normalitzar pesos per columna (cadascun dels G punts)
    w_sum = weights.sum(axis=0, keepdims=True)  # (1, G)
    weights /= w_sum

    # Sumar per obtenir valors interpolats
    pred_flat = (weights * values[:, None]).sum(axis=0)  # (G,)

    # Reconstruir matriu (nl, nm)
    pred_grid = pred_flat.reshape(nl, nm).astype(np.float32)

    # 7) Escriure a NetCDF
    with Dataset(file_path, "w", format="NETCDF4") as nc:
        nc.createDimension("lat", nl)
        nc.createDimension("lon", nm)

        lat_var = nc.createVariable("lat", "f4", ("lat",))
        lon_var = nc.createVariable("lon", "f4", ("lon",))
        var = nc.createVariable(var_name, "f4", ("lat", "lon"), zlib=True, complevel=4)

        lat_var[:] = lat_grid
        lon_var[:] = lon_grid

        # Assignar unitats segons variable
        units_map = {
            "Temp": "degC", "DewPoint": "degC", "PotentialTemp": "degC",
            "Humitat": "%", "Pluja": "mm", "Patm": "hPa"
        }
        var.units = units_map.get(var_name, "units")
        var[:, :] = pred_grid

        nc.title = f"Predicció interpolada de '{var_name}'"
        nc.history = f"Generat amb MeteoGraphSAGEEnhanced v2"
    
    logging.info(f"NetCDF guardat a: {file_path}")

if __name__ == "__main__":
    # Configuració d'arguments (hiperparàmetres i opcions)
    parser = argparse.ArgumentParser(description="Entrenament i execució del model MeteoGraphSAGE")
    parser.add_argument("--data_dir", type=str, default="D:/DADES_METEO_PC_TO_DATA",
                        help="Directori amb els fitxers pt de dades (grafs horaris i seqüències)")
    parser.add_argument("--group_by", type=str, choices=["day", "month"], default="day",
                        help="Tipus d'agrupació de seqüències temporals (day = per dia, month = per mes)")
    parser.add_argument("--history_length", type=int, default=None,
                        help="Longitud de la seqüència (finestra mòbil) si no s'agrupa per període fix")
    parser.add_argument("--target_variable", type=str, default="Temp",
                        help="Nom de la variable objectiu a predir (ha de ser una de FEATURE_NAMES)")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Horitzó de predicció (nombre de passos temporals endavant a predir per a prediccions iteratives)")
    parser.add_argument("--out_all_vars", action="store_true",
                        help="Si s'especifica, el model predirà totes les variables de cop en comptes d'una sola variable objectiu")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'èpoques d'entrenament")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimensionalitat oculta (nombre de neurones ocultes)")
    parser.add_argument("--graph_layers", type=int, default=2, help="Nombre de capes GraphSAGE (profunditat del graf)")
    parser.add_argument("--station_embedding_dim", type=int, default=8,
                        help="Dimensió de l'embedding d'estació (0 per no utilitzar embedding d'estació)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Taxa d'aprenentatge")
    parser.add_argument("--output_netcdf", action="store_true", help="Generar fitxer NetCDF amb predicció regional interpolada")
    parser.add_argument("--predict_station", type=int, default=None, help="ID d'una estació per fer una predicció individual de demostració")
    parser.add_argument("--aggregator", type=str, default="gat",
                    choices=["sage","gat","gin"],
                    help="Tipus d'agregació espacial")
    parser.add_argument("--temporal_model", type=str, default="evolvegcn",
                    choices=["none","lstm","gru","transformer","evolvegcn"],
                    help="Model temporal")
    parser.add_argument("--num_attention_heads", type=int, default=4, help="Nombre de caps per a GAT i atenció temporal (multi-head)")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--stride", type=int, default=None, help="Salt (stride) entre finestres de seqüències. Per defecte, igual a history_length (no solapat).")

    args = parser.parse_args()

    # ——— Iniciem DDP ————————————————————————————————
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # ————————————————————————————————————————————————

    # Carrega seqüències en streaming, assignant-les a cada rank per round-robin
    seq_dicts = []
    for idx, seq in enumerate(create_sequences(
         args.data_dir,
         period=args.group_by,
         window_size=args.history_length,
         stride=(args.stride or args.history_length),
         min_seq_len=2,
         lazy=True
    )):
       # només conservar aquelles seqüències on idx % world_size == rank
       if idx % world_size == rank:
           seq_dicts.append(seq)
    
    if not seq_dicts:
        logging.error("No s'han trobat seqüències de dades per entrenar/predictir.")
        exit(1)

    # 2) Separem les llistes de Data dels altres camps
    sequences = [sd["data"] for sd in seq_dicts]      # List[List[Data]]
    all_ids    = seq_dicts[0]["ids"]                  # List[id], comú a tots
    masks      = [sd["mask"] for sd in seq_dicts]      # List[Tensor(seq_len, N)]

    # Assegurar que hi ha correspondència de nombre de features amb FEATURE_NAMES (p.ex. si s'han afegit Vent_u, Vent_v)
    example_data = sequences[0][0]
    in_features = example_data.x.shape[1]
    if in_features != len(FEATURE_NAMES):
        # Si hi ha dues columnes extra, assumim que són Vent_u i Vent_v
        if in_features == len(FEATURE_NAMES) + 2:
            FEATURE_NAMES.extend(['Vent_u', 'Vent_v'])
            logging.info("S'han detectat Vent_u i Vent_v en les dades. FEATURE_NAMES actualitzat amb aquests camps.")
        else:
            logging.warning(f"El nombre de features ({in_features}) no coincideix amb l'esperat ({len(FEATURE_NAMES)}). Procedint igualment.")
    # Identificar l'índex de la variable objectiu
    if args.target_variable in FEATURE_NAMES:
        target_idx = FEATURE_NAMES.index(args.target_variable)
    else:
        logging.warning(f"Variable objectiu {args.target_variable} no reconeguda, s'utilitzarà {FEATURE_NAMES[0]} per defecte.")
        target_idx = 0
    # Determinar out_features (1 o totes)
    if args.out_all_vars:
        out_features = in_features
    else:
        out_features = 1
    # Preparar embedding d'estacions: obtenir tots els IDs únics
    all_station_ids = set()
    for seq in sequences:
        for data in seq:
            if hasattr(data, 'ids'):
                for sid in data.ids:
                    all_station_ids.add(sid)
    all_station_ids = sorted(list(all_station_ids))
    id_to_idx = {sid: idx for idx, sid in enumerate(all_station_ids)}
    num_stations = len(all_station_ids)
    # Afegir índex d'estació a cada node de cada graf (per fer servir embedding)
    if args.station_embedding_dim > 0:
        for seq in sequences:
            for data in seq:
                if hasattr(data, 'ids'):
                    data.station_idx = [id_to_idx[sid] for sid in data.ids]

    # Inicialitzar model
    model = MeteoGraphSAGEEnhanced(
        in_features=in_features,
        hidden_dim=args.hidden_dim,
        out_features=out_features,
        aggregator=args.aggregator,
        temporal_model=args.temporal_model,
        num_layers=args.graph_layers,
        num_heads=args.num_attention_heads,
        dropout=0.2,
        station_embedding_dim=(args.station_embedding_dim if args.station_embedding_dim > 0 else 0),
        num_stations=(num_stations if args.station_embedding_dim > 0 else None)
    ).to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Com que ja hem distribuït seq_dicts per rank, ara train_seq = seq_dicts[:train_end], etc.
    total_seq = len(seq_dicts)
    train_end = int(total_seq * 0.8)
    val_end   = int(total_seq * 0.9)
    train_seq = seq_dicts[:train_end]
    val_seq   = seq_dicts[train_end:val_end] if val_end > train_end else []
    test_seq  = seq_dicts[val_end:] if val_end < total_seq else []
    logging.info(f"Seqüències: Train={len(train_seq)}, Val={len(val_seq)}, Test={len(test_seq)}")

    # Només el procés 0 guarda gràfiques/logs globals
    is_master = (rank == 0)
    # ————————————————————————————————————————————————

    logging.info(f"Model MeteoGraphSAGE inicialitzat amb {sum(p.numel() for p in model.parameters())} paràmetres.")
    # Entrenar el model
    history = train_model(
        model, train_seq, val_seq,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        target_idx=target_idx
    )

    train_loss = history['train_loss']
    val_loss   = history.get('val_loss', [])

    # Avaluació (només al master, o sincronitzar resultats)
    if is_master:
        # Extraiem només la llista de Data de cada seqüència
        test_data = [sd['data'] for sd in test_seq]
        metrics = evaluate_model(model, test_data, target_idx=target_idx)
        logging.info(f"Mètriques finals en test: {metrics}")

    # Guardar gràfica de l'evolució de la pèrdua
    epochs_range = range(1, len(train_loss) + 1)
    plt.figure()
    plt.plot(epochs_range, train_loss, label="Train Loss")
    if val_loss:
        plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.xlabel("Època")
    plt.ylabel("MSE")
    plt.title("Evolució de la pèrdua durant l'entrenament")
    plt.legend()
    plt.savefig("loss_evolution.png")
    # Visualització de prediccions vs reals per a una estació (de l'última seqüència de test, o de train si test no existeix)
    if test_seq or train_seq:
        if test_seq:
            sample_seq = test_seq[-1]['data']
        else:
            sample_seq = train_seq[-1]['data']
        if len(sample_seq) > 1:
            vis_station_idx = 0
            vis_station_id = sample_seq[-1].ids[vis_station_idx] if hasattr(sample_seq[-1], 'ids') else vis_station_idx
            true_series = [data.x[vis_station_idx, target_idx].item() for data in sample_seq]
            model.eval()
            pred_series = predict_for_station(model, sample_seq[:-1], vis_station_id, target_idx=target_idx, horizon=len(sample_seq)-1)
            plt.figure()
            plt.plot(range(len(true_series)), true_series, label="Real")
            plt.plot(range(len(true_series)), [None] + pred_series, label="Predicció", linestyle='--')
            plt.xlabel("Pas temporal")
            plt.ylabel(args.target_variable)
            plt.title(f"Predicció vs Real - Estació {vis_station_id}")
            plt.legend()
            plt.savefig("prediction_vs_real.png")
    # Si s'ha indicat fer una predicció específica per una estació (diferent de la visualitzada anteriorment)
    if args.predict_station is not None:
        seq_for_pred = test_seq[-1]['data'] if test_seq else train_seq[0]['data']
        preds_list = predict_for_station(model, seq_for_pred[:-1], args.predict_station, target_idx=target_idx, horizon=args.horizon)
        logging.info(f"Prediccions per a l'estació {args.predict_station} (variable {args.target_variable}): {preds_list}")
    # Si s'ha demanat generar fitxer NetCDF regional
    if args.output_netcdf:
        seq_for_map = test_seq[-1]['data'] if test_seq else train_seq[-1]['data']
        predict_region_to_netcdf(model, seq_for_map, grid_res=0.1, target_idx=target_idx, file_path="prediccio_region.nc")
