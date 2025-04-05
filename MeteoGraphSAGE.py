#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeteoGraphSAGE.py

Model de predicció meteorològica basat en GraphSAGE amb components seqüencials i atenció temporal.
Autor: Nil Farrés Soler (Treball Final de Grau)

Aquest codi assumeix que prèviament s'han executat:
 - prep_GPU_parallel.py (preprocessat de dades crues a CSV filtrats)
 - toData_GPU_parallel.py (conversió de CSV preprocessats a objectes Data de PyTorch Geometric)
 - compute_PC_norm_params.py (càlcul de paràmetres de normalització globals, si escau)

El model MeteoGraphSAGE definit aquí utilitza els objectes Data generats (grafs horaris d'estacions meteorològiques)
per entrenar un sistema de predicció temporal de variables meteorològiques. Incorpora:
 - Capa(s) GraphSAGE per a agregació espacial de veïns en el graf.
 - Capa seqüencial (LSTM o GRU) per a modelar la dinàmica temporal de cada node (estació) al llarg del temps.
 - Mecanisme d'atenció temporal multi-cap (opcional) per enfocar-se en determinats instants passats rellevants.
 - Possibilitat de predir una variable objectiu o múltiples variables.
 - Funcions de predicció per estació i per a tota la regió (amb interpolació a una graella i exportació a NetCDF).

Es considera l'altitud i altres variables derivades físiques en les entrades per millorar la capacitat predictiva en terreny complex.
El codi inclou una secció main per entrenar el model, avaluar-lo i produir sortides de predicció i gràfiques.
"""
import os
import math
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
# Opcional: matplotlib per a gràfiques (es pot instal·lar al cluster si cal)
import matplotlib.pyplot as plt
# Opcional: per exportar a NetCDF
from netCDF4 import Dataset

# Configurem el logger per mostrar informació durant l'entrenament i predicció
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definim els noms de features (columnes) tal com es van utilitzar en el preprocessament
FEATURE_NAMES = ['Temp', 'Humitat', 'Pluja', 'VentFor', 'Patm', 'Alt_norm',
                 'VentDir_sin', 'VentDir_cos', 'hora_sin', 'hora_cos',
                 'dia_sin', 'dia_cos', 'cos_sza', 'DewPoint', 'PotentialTemp']

def create_sequences(data_dir: str, period: str = "day", window_size: int = None):
    """
    Crea o carrega seqüències temporals de grafs (objectes Data) a partir dels fitxers .pt generats.
    - Si hi ha fitxers 'sequence_*.pt' al directori, es carreguen com a seqüències (ja ordenades per temps).
    - Si no, agrupa fitxers individuals segons 'period' (day, month) o per finestra mòbil de longitud window_size.
    
    Retorna una llista de seqüències, on cada seqüència és una llista d'objectes Data ordenats cronològicament.
    """
    sequences = []
    sequence_files = [f for f in os.listdir(data_dir) if f.startswith("sequence_") and f.endswith(".pt")]
    if sequence_files:
        # Carreguem seqüències preparades
        for fname in sorted(sequence_files):
            fpath = os.path.join(data_dir, fname)
            try:
                seq = torch.load(fpath)
                if seq:  # seqüència no buida
                    sequences.append(seq)
            except Exception as e:
                logging.error(f"No s'ha pogut carregar {fname}: {e}")
    else:
        # No hi ha seqüències predefinides, generem-les manualment
        # Llegeix tots els fitxers .pt individuals i ordena per timestamp
        data_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".pt") and not file.startswith("sequence_") and not file.startswith("group_"):
                    data_files.append(os.path.join(root, file))
        # Sort by timestamp if possible
        data_list = []
        for fpath in sorted(data_files):
            try:
                data = torch.load(fpath)
                data_list.append(data)
            except Exception as e:
                logging.error(f"Error carregant {fpath}: {e}")
        # Ara agrupem segons period o window_size
        if window_size:
            # Finestra mòbil: creem seqüències de longitud window_size
            for i in range(0, len(data_list) - window_size + 1):
                seq = data_list[i:i+window_size]
                sequences.append(seq)
        elif period in ["day", "month"]:
            # Agrupem per dia o mes basant-nos en l'atribut timestamp de Data (hauria d'existir)
            grouped = {}
            for data in data_list:
                if hasattr(data, "timestamp"):
                    # timestamp és string "YYYY-MM-DD HH:MM:SS"
                    ts = data.timestamp if isinstance(data.timestamp, str) else str(data.timestamp)
                    try:
                        date = ts.split(" ")[0]  # "YYYY-MM-DD"
                    except:
                        date = ts
                    if period == "day":
                        key = date  # ja és YYYY-MM-DD
                    elif period == "month":
                        key = date[:7]  # YYYY-MM
                    grouped.setdefault(key, []).append(data)
            for key, seq in grouped.items():
                # Ordenem la seqüència per hora
                seq.sort(key=lambda d: d.timestamp if hasattr(d, "timestamp") else 0)
                sequences.append(seq)
            sequences.sort(key=lambda seq: seq[0].timestamp if hasattr(seq[0], "timestamp") else "")
        else:
            # Cap agrupació: cada Data per separat com a seqüència de longitud 1
            for data in data_list:
                sequences.append([data])
    logging.info(f"S'han preparat {len(sequences)} seqüències de període '{period}'")
    return sequences

class MeteoGraphSAGE(nn.Module):
    """
    Model neuronal combinant GraphSAGE per a agregació espacial i LSTM/GRU per a modelat temporal.
    Opcionalment, incorpora atenció temporal multi-cap sobre les seqüències.
    """
    def __init__(self, in_features: int, hidden_dim: int, out_features: int,
                 num_layers: int = 2, use_lstm: bool = True, use_gru: bool = False,
                 use_transformer: bool = False, use_attention: bool = False,
                 num_attention_heads: int = 4, dropout: float = 0.2):
        super(MeteoGraphSAGE, self).__init__()
        # Guardem paràmetres
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.num_layers = num_layers
        self.use_lstm = use_lstm
        self.use_gru = use_gru
        self.use_transformer = use_transformer
        self.use_attention = use_attention
        # Capa d'entrada i capes GraphSAGE
        # Primera capa: de in_features a hidden_dim
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Linear(in_features, hidden_dim))
        # Usem Linear per fer la transformació inicial, després SAGEConv per la propagació (PyG requereix objecte Data, que tindrem al forward)
        # NOTA: Podríem fer servir torch_geometric.nn.SAGEConv directament; però per simplicitat apliquem un linear + agregació manual amb mitjana.
        # (Implementar SAGEConv manualment ens dona control per afegir residuals i norm.)
        # Capa de GraphSAGE: definim pesos per combinar node central i agregat de veïns
        self.W_self = nn.ModuleList()  # pes per la part pròpia
        self.W_nei = nn.ModuleList()   # pes per la part veïns
        for i in range(num_layers):
            # Cada capa tindrà dimensió hidden_dim -> hidden_dim
            self.W_self.append(nn.Linear(hidden_dim, hidden_dim))
            self.W_nei.append(nn.Linear(hidden_dim, hidden_dim))
        # Normalització i activació
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # Mòdul seqüencial (LSTM o GRU) per a la part temporal
        if use_lstm or use_gru:
            rnn_input_dim = hidden_dim
            rnn_hidden_dim = hidden_dim
            if use_lstm:
                self.rnn = nn.LSTM(rnn_input_dim, rnn_hidden_dim, batch_first=False)
            elif use_gru:
                self.rnn = nn.GRU(rnn_input_dim, rnn_hidden_dim, batch_first=False)
        else:
            self.rnn = None
        # Mòdul d'atenció temporal (multi-head) si es sol·licita
        if use_attention:
            # Fem servir MultiheadAttention de PyTorch: embed_dim = hidden_dim, num_heads = num_attention_heads
            self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_attention_heads, batch_first=False)
        else:
            self.attention = None
        # Capa de decodificació final: de hidden_dim (o 2*hidden_dim si concatenem context d'atenció) a out_features
        if use_attention:
            self.decoder = nn.Linear(hidden_dim * 2, out_features)
        else:
            self.decoder = nn.Linear(hidden_dim, out_features)
    
    def forward_graphsage(self, data: Data):
        """
        Aplica les capes d'agregació GraphSAGE al graf donat (data).
        Retorna un tensor de embeddings de mida [num_nodes, hidden_dim].
        """
        x = data.x  # features de nodes, mida [N, in_features]
        # Apliquem primer Linear d'entrada si hi és
        if len(self.conv_layers) > 0:
            x = self.conv_layers[0](x)
        N = x.size(0)
        # Obtenim llistes d'adjacència per agregació. 
        # Utilitzarem edge_index de data (esperem que sigui no dirigit o que conté parells en ambdues direccions).
        if hasattr(data, 'edge_index'):
            edge_index = data.edge_index
        else:
            # Si no hi ha arestes (cas extrem d'un sol node o dades no proporcionades)
            edge_index = torch.empty((2,0), dtype=torch.long, device=x.device)
        # Convertim edge_index a llistes de veïns per a facilitar l'agregació
        # edge_index té forma [2, E] on E és número d'arestes, amb edge_index[0] = origen, edge_index[1] = destí
        if edge_index.numel() > 0:
            neighbors = [[] for _ in range(N)]
            src, dst = edge_index
            # assumim que edge_index inclou edges en les dues direccions; si no, caldria assegurar veïns no dirigits
            for s, d in zip(src.tolist(), dst.tolist()):
                neighbors[d].append(s)
        else:
            neighbors = [[] for _ in range(N)]
        h = x  # inicialment, representació dels nodes
        for i in range(self.num_layers):
            new_h = torch.zeros_like(h)
            # Agreguem veïns: per a cada node, fem mitjana dels veïns
            # (GraphSAGE aggregator mean)
            agg_messages = torch.zeros_like(h)
            for node in range(N):
                if neighbors[node]:
                    nei_feats = h[neighbors[node]]
                    agg_messages[node] = nei_feats.mean(dim=0)
                else:
                    agg_messages[node] = torch.zeros(h.size(1), device=h.device)
            # Apliquem transformacions lineals i combinem
            h_self = self.W_self[i](h)
            h_nei = self.W_nei[i](agg_messages)
            combined = h_self + h_nei
            # Aplicar BatchNorm, activació i dropout
            combined = self.batch_norms[i](combined)
            combined = self.activation(combined)
            combined = self.dropout(combined)
            # Afegim connexió residual (skip connection)
            h = h + combined  # residual
        return h  # embeddings finals dels nodes
    
    def forward(self, data_sequence):
        """
        Execució endavant del model.
        data_sequence pot ser:
         - Una llista d'objectes Data (seqüència temporal).
         - Un únic objecte Data (cas sense seqüència, predicció instantània).
        Retorna:
         - Un tensor de prediccions de mida [num_nodes, out_features] per a l'últim instant de la seqüència (horitzó de predicció).
        """
        # Si data_sequence és un sol graf, el convertim a llista d'un element
        if isinstance(data_sequence, Data):
            data_sequence = [data_sequence]
        seq_length = len(data_sequence)
        device = next(self.parameters()).device  # obtenim device del model (CPU o CUDA)
        # Si no tenim component seqüencial i la seqüència conté múltiples grafs, només considerem l'últim com a input directe
        if self.rnn is None and seq_length > 1:
            data_seq_input = data_sequence[:-1]  # fins al penúltim
            data_target = data_sequence[-1]
            # Podríem ignorar tot i només fer servir l'últim input per predir sobre ell mateix.
            # Per consistència, agafem l'últim graf i farem forward estàtic.
            data_sequence = [data_sequence[-2], data_sequence[-1]] if seq_length >= 2 else [data_sequence[-1]]
            seq_length = len(data_sequence)
        # Filtrar nodes comuns a tota la seqüència per evitar problemes de dimensions
        common_ids = None
        for data in data_sequence:
            ids = list(data.ids) if hasattr(data, "ids") else list(range(data.x.size(0)))
            id_set = set(ids)
            if common_ids is None:
                common_ids = id_set
            else:
                common_ids &= id_set
        if common_ids is None:
            common_ids = set()
        # Si common_ids és menor que total, filtrem cada graf
        if common_ids and any(len(data.x) != len(common_ids) for data in data_sequence):
            common_ids = sorted(list(common_ids))
            common_idx_maps = []
            # Creem un mapping d'ID a nou index per als comuns
            id_to_new_idx = {id_val: idx for idx, id_val in enumerate(common_ids)}
            filtered_sequence = []
            for data in data_sequence:
                if hasattr(data, "ids"):
                    # indices dels nodes comuns en aquest graf
                    idx_keep = [i for i, id_val in enumerate(data.ids) if id_val in id_to_new_idx]
                else:
                    # si no hi ha ids, assumim mateix ordre i número
                    idx_keep = list(range(len(common_ids)))
                idx_keep_tensor = torch.tensor(idx_keep, dtype=torch.long)
                # Utilitzem subgraph de torch_geometric.utils per tallar el graf als idx seleccionats
                new_edge_index, new_edge_attr = subgraph(idx_keep_tensor, data.edge_index, getattr(data, 'edge_attr', None), relabel_nodes=True)
                new_x = data.x[idx_keep_tensor]
                # Construïm un nou Data
                new_data = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
                # Filtrar també la llista d'ids
                if hasattr(data, "ids"):
                    new_data.ids = [data.ids[i] for i in idx_keep]
                # Mantenim el timestamp si existeix
                if hasattr(data, "timestamp"):
                    new_data.timestamp = data.timestamp
                filtered_sequence.append(new_data)
            data_sequence = filtered_sequence
            # Actualitzem seq_length si s'ha modificat
            seq_length = len(data_sequence)
        # Llista per emmagatzemar embeddings de cada pas temporal
        node_embeddings_seq = []
        for t, data in enumerate(data_sequence):
            data = data.to(device)
            # Apliquem GraphSAGE al graf d'aquest timestep per obtenir embeddings
            node_emb = self.forward_graphsage(data)
            node_embeddings_seq.append(node_emb)
        # Convertim la llista a tensor de seqüència [seq_len, num_nodes, hidden_dim]
        # Assegurem que tots tenen la mateixa N (després del filtrat comú).
        node_embeddings_seq = torch.stack(node_embeddings_seq, dim=0)  # shape: (T, N, hidden_dim)
        # Si tenim RNN (LSTM/GRU), passem la seqüència pel RNN
        if self.rnn is not None:
            # Inicialitzem estat ocult (h0, c0) a zeros per a cada node com a seqüència independent
            # PyTorch ho fa automàticament si no proporcionem h0, per tant ho omitirem per claredat.
            # Executem RNN sobre la seqüència sencera
            rnn_out, hidden = self.rnn(node_embeddings_seq)
            # rnn_out té shape (T, N, hidden_dim). Agafem l'últim pas (T-1) com a representació final de cada node.
            last_out = rnn_out[-1]  # shape: (N, hidden_dim)
            if self.use_attention and self.attention is not None:
                # Apliquem atenció temporal: query = l'últim hidden, key = value = tota la seqüència de sortides RNN
                # Hem de donar shape (L, N, E) a key i value, i (1, N, E) a query.
                query = last_out.unsqueeze(0)        # (1, N, hidden_dim)
                key = value = rnn_out                # (T, N, hidden_dim)
                attn_output, attn_weights = self.attention(query, key, value)
                # attn_output shape: (1, N, hidden_dim)
                context = attn_output.squeeze(0)     # (N, hidden_dim)
                # Concatem l'últim hidden i el context d'atenció per formar la representació final
                combined = torch.cat([last_out, context], dim=1)  # (N, 2*hidden_dim)
                # Decodifiquem a sortida
                preds = self.decoder(combined)  # (N, out_features)
            else:
                # Sense atenció: fem servir directament l'últim estat com a representació
                preds = self.decoder(last_out)  # (N, out_features)
        elif self.use_transformer:
            # Opcional: si s'hagués implementat un transformer temporal en lloc de RNN.
            # (No completat per simplicitat; es podria afegir pos encodings i self-attention here)
            # Utilitzarem l'últim embedding directament per predir (model sense estat temporal explícit)
            last_emb = node_embeddings_seq[-1]  # (N, hidden_dim)
            preds = self.decoder(last_emb)      # (N, out_features)
        else:
            # Cas sense RNN i seqüència de longitud 1: simplement apliquem decoder al embedding del graf
            emb = node_embeddings_seq[-1]  # (N, hidden_dim)
            preds = self.decoder(emb)      # (N, out_features)
        return preds

def train_model(model: MeteoGraphSAGE, train_sequences: list, val_sequences: list = None,
                epochs: int = 50, learning_rate: float = 0.001, target_idx: int = 0):
    """
    Entrena el model amb les seqüències de train. Opcionalment, valida contra val_sequences.
    target_idx indica l'índex de la variable objectiu en les features (si out_features=1).
    Retorna el model entrenat i els històrics de pèrdues de train i val.
    """
    device = next(model.parameters()).device
    model.train()  # mode entrenament
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Funció de pèrdua: si out_features == 1 usem MSE sobre variable objectiu
    # si out_features > 1 (prediccions multivariables), sumem MSE de totes o es pot fer MSE global vectorial.
    criterion = nn.MSELoss()
    train_loss_history = []
    val_loss_history = []
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        for seq in train_sequences:
            # Prepara dades input (tota la seqüència menys l'últim) i label (últim valor real)
            if len(seq) < 2:
                # Seqüència massa curta per entrenar (necessitem almenys 1 input i 1 target)
                continue
            input_seq = seq[:-1]
            target_graph = seq[-1]
            # Forward
            preds = model(input_seq)
            # Obtenir valors objectiu reals
            # Si el model prediu una sola variable:
            if model.out_features == 1:
                # Fem servir target_idx per extreure la variable objectiu
                target_vals = target_graph.x[:, target_idx].to(device)
                # preds té shape (N,1), convertim a (N,)
                pred_vals = preds.view(-1)
            else:
                # Si prediu múltiples variables, comparem tot el vector de features
                target_vals = target_graph.x.to(device)
                pred_vals = preds
            # Calculem pèrdua (error mig quadràtic)
            loss = criterion(pred_vals, target_vals)
            # Backprop i optimització
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # Mètrica de pèrdua de training mitjana per seqüència
        if len(train_sequences) > 0:
            epoch_loss /= len(train_sequences)
        train_loss_history.append(epoch_loss)
        # Avaluació en validació
        val_loss = 0.0
        if val_sequences:
            model.eval()
            with torch.no_grad():
                for seq in val_sequences:
                    if len(seq) < 2:
                        continue
                    input_seq = seq[:-1]
                    target_graph = seq[-1]
                    preds = model(input_seq)
                    if model.out_features == 1:
                        true_vals = target_graph.x[:, target_idx].to(device)
                        pred_vals = preds.view(-1)
                    else:
                        true_vals = target_graph.x.to(device)
                        pred_vals = preds
                    loss_val = criterion(pred_vals, true_vals)
                    val_loss += loss_val.item()
                if len(val_sequences) > 0:
                    val_loss /= len(val_sequences)
            model.train()
        val_loss_history.append(val_loss)
        logging.info(f"Època {epoch}/{epochs} - Pèrdua train: {epoch_loss:.4f} - Pèrdua val: {val_loss:.4f}")
    return model, train_loss_history, val_loss_history

def evaluate_model(model: MeteoGraphSAGE, test_sequences: list, target_idx: int = 0):
    """
    Avalua el model sobre el conjunt de test. Retorna diccionari de mètriques (p. ex. RMSE).
    """
    device = next(model.parameters()).device
    model.eval()
    total_mse = 0.0
    count = 0
    # Podem calcular mètriques per node o globals; aquí fem global MSE i RMSE
    with torch.no_grad():
        for seq in test_sequences:
            if len(seq) < 2:
                continue
            input_seq = seq[:-1]
            target_graph = seq[-1]
            preds = model(input_seq)
            if model.out_features == 1:
                true_vals = target_graph.x[:, target_idx].to(device)
                pred_vals = preds.view(-1)
            else:
                true_vals = target_graph.x.to(device)
                pred_vals = preds
            # Suma de l'error quadràtic
            mse = ((pred_vals - true_vals) ** 2).mean().item()
            total_mse += mse
            count += 1
    avg_mse = total_mse / count if count > 0 else 0.0
    rmse = math.sqrt(avg_mse)
    logging.info(f"Resultats de test - MSE: {avg_mse:.4f} - RMSE: {rmse:.4f}")
    return {"MSE": avg_mse, "RMSE": rmse}

def predict_for_station(model: MeteoGraphSAGE, data_sequence: list, station_id: int,
                        target_idx: int = 0, horizon: int = 1):
    """
    Realitza prediccions per a una estació individual donada (station_id) a partir d'una seqüència històrica.
    Retorna una llista amb les prediccions successives (fins a 'horizon' passos endavant).
    """
    device = next(model.parameters()).device
    model.eval()
    preds_list = []
    # Creem una còpia de la seqüència per no modificar l'original
    seq_copy = [data.clone() for data in data_sequence]
    # Ens assegurem que la seqüència està al device correcte
    for data in seq_copy:
        data.to(device)
    current_sequence = seq_copy
    # Mapeig d'index del node de l'estació en cada graf (assumim id està present en tots)
    # Trobem l'índex de l'estació objectiu en el darrer graf de la seqüència
    station_index = None
    if hasattr(current_sequence[-1], "ids"):
        # Busquem l'estació en la llista d'ids
        if station_id in current_sequence[-1].ids:
            station_index = current_sequence[-1].ids.index(station_id)
    else:
        # Si no hi ha ids, assumim que station_id és un índex directe (cas fictici)
        station_index = station_id if station_id < current_sequence[-1].x.size(0) else None
    if station_index is None:
        logging.warning(f"L'estació {station_id} no es troba en l'últim graf de la seqüència. Predicció no disponible.")
        return preds_list
    with torch.no_grad():
        for h in range(horizon):
            # Utilitzem el model per predir el següent pas
            preds = model(current_sequence)
            if model.out_features == 1:
                pred_value = preds.view(-1)[station_index].item()
            else:
                pred_value = preds[station_index, target_idx].item()
            preds_list.append(pred_value)
            # Actualitzem la seqüència: afegim un nou Data amb aquesta predicció (com si fos el següent pas real)
            # Per crear el nou graf futur:
            last_graph = current_sequence[-1]
            new_graph = last_graph.clone()  # copiem l'últim i li canviem la variable objectiu
            # Actualitzem la variable objectiu de l'estació corresponent amb la predicció
            new_val = pred_value
            if model.out_features == 1:
                new_graph.x[station_index, target_idx] = new_val
            else:
                # Si són multivariables, només substituïm la que predim i deixem les altres igual (o es podria predir totes)
                new_graph.x[station_index, target_idx] = new_val
            # Afegim el nou graf a la seqüència (i llevem el primer si mantenim finestra fixa)
            current_sequence.append(new_graph)
            # Opcional: es podria treure current_sequence[0] si volem finestra lliscant de mida constant
    return preds_list

def predict_region_to_netcdf(model: MeteoGraphSAGE, data_sequence: list, grid_res: float = 0.1,
                             target_idx: int = 0, file_path: str = "prediction.nc"):
    """
    Utilitza el model per predir la variable objectiu al proper pas en totes les estacions, 
    després interpola espacialment aquests resultats a una graella regular i guarda en un fitxer NetCDF.
    """
    device = next(model.parameters()).device
    model.eval()
    if len(data_sequence) < 1:
        logging.error("Seqüència buida, no es pot generar predicció regional.")
        return
    # Assegurar que la seqüència és al device
    for data in data_sequence:
        data.to(device)
    # Predir per l'últim pas (seqüència completa com input)
    with torch.no_grad():
        preds = model(data_sequence)
    # Obtenir coordenades i prediccions de cada estació
    last_graph = data_sequence[-1]
    if not hasattr(last_graph, "pos") and hasattr(last_graph, "x"):
        # Si posicions no estan explícites, comprovem si lat i lon estan en features
        # (No es va guardar explicitament pos a toData, possiblement lat, lon són columnes extra)
        try:
            lat_idx = FEATURE_NAMES.index('lat')
            lon_idx = FEATURE_NAMES.index('lon')
            coords = [(last_graph.x[i, lat_idx].item(), last_graph.x[i, lon_idx].item()) for i in range(last_graph.x.size(0))]
        except:
            logging.error("No s'han trobat coordenades 'pos' o 'lat/lon' als nodes.")
            return
    else:
        coords = [(float(p[0]), float(p[1])) for p in last_graph.pos]  # assumim pos = [lat, lon] per node
    preds = preds.detach().cpu().numpy()
    if model.out_features == 1:
        pred_values = preds.reshape(-1)
    else:
        pred_values = preds[:, target_idx]
    # Convertim lat/lon de graus a rad si fem servir fórmula haversine (per ara no, usem dist euclidea approx)
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    # Definir graella
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    lat_grid = np.arange(min_lat, max_lat + 1e-9, grid_res)
    lon_grid = np.arange(min_lon, max_lon + 1e-9, grid_res)
    nlat = len(lat_grid)
    nlon = len(lon_grid)
    pred_grid = np.full((nlat, nlon), np.nan, dtype=np.float32)
    # Interpolació IDW
    for i, lat in enumerate(lat_grid):
        for j, lon in enumerate(lon_grid):
            # Calcular distàncies de (lat, lon) a cada estació
            # Convertim diferència en graus a distància aproximada en km per ponderar (1 grau lat ~ 111 km, 1 grau lon ~ cos(lat)*111 km)
            dists = []
            for (st_lat, st_lon) in coords:
                dlat = (st_lat - lat) * 111.0  # km approx
                dlon = (st_lon - lon) * 111.0 * math.cos(math.radians(lat))
                d = math.sqrt(dlat**2 + dlon**2)
                dists.append(d)
            dists = np.array(dists)
            # Evitem divisió per zero: si un punt coincideix exactament amb estació, prenem valor directe
            if np.any(dists < 1e-6):
                k = np.argmin(dists)
                pred_val = pred_values[k]
            else:
                # Pesos = invers de la distància
                w = 1.0 / (dists + 1e-6)
                w /= w.sum()
                pred_val = np.dot(w, pred_values)
            pred_grid[i, j] = pred_val
    # Crear fitxer NetCDF
    nc = Dataset(file_path, "w", format="NETCDF4")
    nc.createDimension("lat", nlat)
    nc.createDimension("lon", nlon)
    lat_var = nc.createVariable("lat", "f4", ("lat",))
    lon_var = nc.createVariable("lon", "f4", ("lon",))
    pred_var = nc.createVariable("prediction", "f4", ("lat", "lon"))
    lat_var.units = "degrees_north"
    lon_var.units = "degrees_east"
    pred_var.units = "units"  # es podria especificar unitat real de la variable predita, p. ex. "degC" si és temperatura
    lat_var[:] = lat_grid
    lon_var[:] = lon_grid
    pred_var[:, :] = pred_grid
    nc.title = "Predicció interpolada"
    nc.close()
    logging.info(f"Fitxer NetCDF de predicció regional guardat a {file_path}")

if __name__ == "__main__":
    # Configuració d'arguments (hiperparàmetres i opcions)
    parser = argparse.ArgumentParser(description="Entrenament i execució del model MeteoGraphSAGE")
    parser.add_argument("--data_dir", type=str, default="D:/DADES_METEO_PC_TO_DATA",
                        help="Directori amb els fitxers .pt de dades (grafs horaris i seqüències)")
    parser.add_argument("--group_by", type=str, choices=["none", "day", "month"], default="day",
                        help="Tipus d'agrupació de seqüències temporals (none=cap agrupar, day=dia, month=mes)")
    parser.add_argument("--history_length", type=int, default=None,
                        help="Longitud de la seqüència (finestra mòbil) si no s'agrupa per període fix")
    parser.add_argument("--target_variable", type=str, default="Temp",
                        help="Nom de la variable objectiu a predir (ha de ser una de FEATURE_NAMES)")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Horitzó de predicció (nombre de passos temporals endavant a predir)")
    parser.add_argument("--out_all_vars", action="store_true",
                        help="Si s'especifica, el model predirà totes les variables de cop en comptes d'una sola")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'èpoques d'entrenament")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimensionalitat oculta (nombre de neurones ocultes)")
    parser.add_argument("--graph_layers", type=int, default=2, help="Nombre de capes GraphSAGE")
    parser.add_argument("--use_lstm", action="store_true", help="Utilitzar LSTM per a la seqüència temporal")
    parser.add_argument("--use_gru", action="store_true", help="Utilitzar GRU per a la seqüència temporal")
    parser.add_argument("--use_transformer", action="store_true", help="Utilitzar atenció tipus Transformer (experimental)")
    parser.add_argument("--use_attention", action="store_true", help="Utilitzar mecanisme d'atenció temporal sobre RNN")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Taxa d'aprenentatge")
    parser.add_argument("--output_netcdf", action="store_true", help="Generar fitxer NetCDF amb predicció regional interpolada")
    parser.add_argument("--predict_station", type=int, default=None, help="ID d'una estació per fer una predicció individual de demostració")
    args = parser.parse_args()

    # Selecció de device (GPU si disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Carregar seqüències de dades
    sequences = create_sequences(args.data_dir, period=args.group_by, window_size=args.history_length)
    if not sequences:
        logging.error("No s'han trobat seqüències de dades per entrenar/predir.")
        exit(1)
    # Ordenar seqüències cronològicament (assumim que seqüències tenen atribut timestamp al primer element)
    sequences.sort(key=lambda seq: seq[0].timestamp if hasattr(seq[0], "timestamp") else "")
    # Dividir en train, val, test (80%-10%-10% per exemple)
    total_seq = len(sequences)
    train_end = int(total_seq * 0.8)
    val_end = int(total_seq * 0.9)
    train_sequences = sequences[:train_end]
    val_sequences = sequences[train_end:val_end] if val_end > train_end else []
    test_sequences = sequences[val_end:] if val_end < total_seq else []
    logging.info(f"Seqüències: Train={len(train_sequences)}, Val={len(val_sequences)}, Test={len(test_sequences)}")

    # Determinar dimensions de features i sortida
    example_data = sequences[0][0]
    in_features = example_data.x.shape[1]
    # Comprovem que la variable objectiu existeix
    if args.target_variable in FEATURE_NAMES:
        target_idx = FEATURE_NAMES.index(args.target_variable)
    else:
        logging.warning(f"Variable objectiu {args.target_variable} no reconeguda, per defecte s'agafarà {FEATURE_NAMES[0]}")
        target_idx = 0
    if args.out_all_vars:
        out_features = in_features
    else:
        out_features = 1
    # Crear model
    model = MeteoGraphSAGE(in_features=in_features, hidden_dim=args.hidden_dim, out_features=out_features,
                           num_layers=args.graph_layers, use_lstm=args.use_lstm, use_gru=args.use_gru,
                           use_transformer=args.use_transformer, use_attention=args.use_attention,
                           num_attention_heads=4, dropout=0.2).to(device)
    logging.info(f"Model MeteoGraphSAGE inicialitzat amb {sum(p.numel() for p in model.parameters())} paràmetres.")
    # Entrenar model
    model, train_hist, val_hist = train_model(model, train_sequences, val_sequences,
                                             epochs=args.epochs, learning_rate=args.learning_rate, target_idx=target_idx)
    # Avaluar model
    metrics = evaluate_model(model, test_sequences, target_idx=target_idx)
    logging.info(f"Mètriques finals en test: {metrics}")
    # Guardar gràfiques d'evolució de la pèrdua
    epochs_range = range(1, len(train_hist)+1)
    plt.figure()
    plt.plot(epochs_range, train_hist, label="Train Loss")
    if val_sequences:
        plt.plot(epochs_range, val_hist, label="Val Loss")
    plt.xlabel("Època")
    plt.ylabel("MSE")
    plt.title("Evolució de la pèrdua durant l'entrenament")
    plt.legend()
    plt.savefig("loss_evolution.png")
    # Opcional: visualització de prediccions vs reals per a una estació i variable
    if test_sequences:
        sample_seq = test_sequences[-1]  # última seqüència de test
        if len(sample_seq) > 1:
            # Agafem una estació (el primer node per exemple, o la indicada per --predict_station si existeix en la seqüència)
            vis_station_idx = 0
            vis_station_id = sample_seq[-1].ids[vis_station_idx] if hasattr(sample_seq[-1], "ids") else vis_station_idx
            true_series = [data.x[vis_station_idx, target_idx].item() for data in sample_seq]
            model.eval()
            pred_series = []
            # Predir pas a pas autoregressivament per visualitzar (utilitzem la funció predict_for_station per comoditat)
            pred_series = predict_for_station(model, sample_seq[:-1], vis_station_id, target_idx=target_idx, horizon=len(sample_seq)-1)
            # Plot
            plt.figure()
            plt.plot(range(len(true_series)), true_series, label="Real")
            plt.plot(range(len(true_series)), [None]+pred_series, label="Predicció", linestyle='--')
            plt.xlabel("Pas horari")
            plt.ylabel(args.target_variable)
            plt.title(f"Predicció vs Real - Estació {vis_station_id}")
            plt.legend()
            plt.savefig("prediction_vs_real.png")
    # Si es demana una predicció per una estació concreta (diferent de la de dalt)
    if args.predict_station is not None:
        # Seleccionem l'última seqüència de test si existeix, sinó de train
        seq_for_pred = test_sequences[-1] if test_sequences else train_sequences[0]
        preds_list = predict_for_station(model, seq_for_pred[:-1], args.predict_station, target_idx=target_idx, horizon=args.horizon)
        logging.info(f"Prediccions per a l'estació {args.predict_station} (variable {args.target_variable}): {preds_list}")
    # Si es demana generar NetCDF regional
    if args.output_netcdf:
        seq_for_map = test_sequences[-1] if test_sequences else train_sequences[-1]
        predict_region_to_netcdf(model, seq_for_map, grid_res=0.1, target_idx=target_idx, file_path="prediccio_region.nc")
