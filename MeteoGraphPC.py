#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeteoGraphPC.py
"""

from __future__ import annotations
import csv, glob, os, random, argparse
from datetime import datetime

import torch
from torch import nn, optim, Tensor
from typing import List
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.nn import GCNConv
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from torch_geometric_temporal.nn.recurrent import TGCN
import torch.nn.functional as F

from torch.amp import autocast
from torch.cuda.amp import GradScaler

from torch.utils.checkpoint import checkpoint
from contextlib import nullcontext

import torch.multiprocessing as tmp_mp
tmp_mp.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings("ignore",
                        category=FutureWarning,
                       message=".*weights_only=False.*")

#Eliminem warnings de scaler = GradScaler()
warnings.filterwarnings("ignore",
                        category=FutureWarning,
                        message=".*GradScaler.*")

# ───────────────────────────────── UTILITATS ──────────────────────────────────
def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate(batch):
    xs, eis, eas, masks, ids_seq, ys = [], [], [], [], [], []
    for x_seq, ei_seq, ea_seq, mask_seq, id_seq, y_seq in batch:
        xs.append(x_seq)
        eis.append(ei_seq)
        eas.append(ea_seq)
        masks.append(mask_seq)
        ids_seq.append(id_seq)
        ys.append(torch.stack(y_seq, dim=0))
    return xs, eis, eas, masks, ids_seq, torch.stack(ys)


def moving_average(prev_mean, prev_var_times_n, new_vec, n_seen):
    """Actualitza mitjana i var·n amb Welford (streaming)."""
    n = n_seen + 1
    delta = new_vec - prev_mean
    mean = prev_mean + delta / n
    var_times_n = prev_var_times_n + delta * (new_vec - mean)
    return mean, var_times_n, n

# ──────── funció helper a NIVELL DE MÒDUL (picklable) ────────
def is_valid_pt(fp: str) -> bool:
    # Obrim i tanquem explícitament el fitxer per evitar descriptors pendents
    with open(fp, "rb") as f:
        d = torch.load(f, map_location="cpu")
    # Descarta seqüències sense cap etiqueta a l'últim pas
    mask_last = d["y_mask_seq"][-1]               # tensor booleà de mida [N_global]
    if not mask_last.any().item():                # True si al menys un node està present
        return False
    # Comprova nan/inf per si de cas
    y = d["y_seq"][-1]
    return not (torch.isnan(y).any() or torch.isinf(y).any())

# ───────────────────────────────── DATASET ────────────────────────────────────
class GraphSeqDataset(Dataset):
    def __init__(self, seq_dir: str, num_workers: int | None = None, input_idx: list[int] = None, target_idx: list[int] = None):
        all_pt = sorted(glob.glob(os.path.join(seq_dir, "*.pt")))
        if not all_pt:
            raise RuntimeError("No hi ha .pt al directori!")

        workers = num_workers or max(multiprocessing.cpu_count() - 1, 1)
        print(f"Validant {len(all_pt):,} fitxers amb {workers} processos…")

        self.files: list[str] = []
        self.input_idx = input_idx
        self.target_idx = target_idx
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for fp, ok in tqdm(
                zip(all_pt, ex.map(is_valid_pt, all_pt)),
                total=len(all_pt),
                desc="Filtrant",
                unit="fitxer",
            ):
                if ok:
                    self.files.append(fp)

        if not self.files:
            raise RuntimeError("Cap fitxer vàlid! Revisa les seqüències.")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location="cpu")
        # Carregar tota la seqüència de futurs (llista de [N, F])
        y_seq_full = d["y_seq"]            # [W] llista de [N, F]
        if self.target_idx:
            # Filtrar només les columnes que volem predir
            y_seq = [ y_t[:, self.target_idx] for y_t in y_seq_full ]
        else:
            y_seq = y_seq_full

        # Reparem edge_index_seq perquè sempre sigui 2×E
        clean_ei_seq = []
        for ei in d["edge_index_seq"]:
            # si PyTorch ha aplatat o només n’hi ha 1 fila, el normalitzem a 2×0
            if ei.dim() == 1 or (ei.dim() == 2 and ei.size(0) == 1):
                ei = torch.empty((2, 0), dtype=torch.long, device=ei.device)
            clean_ei_seq.append(ei)

        # Selecció de features d'entrada
        x_seq = [
            x[:, self.input_idx] if self.input_idx else x
            for x in d["x_seq"]
        ]

        id_seq = d["id_seq"]
        ea_seq = d["edge_attr_seq"]
        mask_seq = d["mask_seq"]

        return x_seq, clean_ei_seq, ea_seq, mask_seq, id_seq, y_seq

    
# ───────────────────────────────── MODELS ──────────────────────────────────────
class TemporalConvCell(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, kernel_size: int = 3, dilations: list[int] = [1, 2, 4]):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        
        # Graph convolution for node features
        self.graph_conv = GCNConv(in_channels, hidden_size)
        
        # TCN layers with different dilations
        self.tcn_layers = nn.ModuleList()
        for d in dilations:
            # Each TCN layer is a dilated 1D convolution
            self.tcn_layers.append(nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * d // 2,
                dilation=d
            ))
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size * len(dilations), hidden_size)
        
    def forward(self, x, edge_index, edge_attr=None, h_prev=None):
        """
        x: [N, in_channels] - Node features at current step
        edge_index: [2, E] - Graph connectivity
        edge_attr: [E, edge_attr_dim] - Edge attributes (optional)
        h_prev: [N, kernel_size, hidden_size] - Previous hidden states for TCN
        """
        # First, apply graph convolution to get node embeddings
        h_graph = self.graph_conv(x, edge_index)
        h_graph = F.relu(h_graph)  # Apply non-linearity
        
        batch_size = h_graph.size(0)
        
        # Use the provided history if available, otherwise use zeros
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.kernel_size, self.hidden_size, device=x.device)
        
        # For each node, apply the TCN layers to its history
        tcn_outputs = []
        for i, tcn_layer in enumerate(self.tcn_layers):
            # Reshape for 1D convolution: [N, hidden_size, kernel_size]
            h_input = h_prev.transpose(1, 2)
            
            # Apply TCN layer
            h_tcn = tcn_layer(h_input)
            
            # Take the last output
            tcn_outputs.append(h_tcn[:, :, -1])
        
        # Concatenate TCN outputs
        h_tcn_concat = torch.cat(tcn_outputs, dim=1)  # [N, hidden_size * num_dilations]
        
        # Final projection
        h_out = self.out_proj(h_tcn_concat)  # [N, hidden_size]
        
        # Return the new hidden state
        return h_out

class DynTGCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, out_channels: int = None, horizon: int = None):
        super().__init__()
        # cèl·lula temporal TGCN que combina GCN + GRU
        self.tgcn_cell = TGCN(in_channels, hidden)
        self.hidden_size = hidden
        self.out_channels = out_channels or in_channels
        self.head = nn.Linear(hidden, self.out_channels)
        self.horizon = horizon

    def forward(self, x_seq, ei_seq, ea_seq, mask_seq, id_seq):
        """
        x_seq:    [T] llista de tensors [N_global x F]
        ei_seq:   [T] llista d'edge_index globals
        ea_seq:   [T] llista d'edge_attr globals
        mask_seq: [T] llista de booleans [N_global] (nodes presents)
        id_seq:   (ignored)
        """
        N_global = x_seq[0].size(0)
        H = self.hidden_size
        # Diccionari d’estats: index_global → Tensor([H])
        h_dict: dict[int, Tensor] = {}

        # 1) Encoding: per cada timestamp, passem el grafo sencer
        for x, ei, ea, mask in zip(x_seq, ei_seq, ea_seq, mask_seq):
            # 1.1) Construir h_prev global: [N_global, H]
            h_prev = x.new_zeros((N_global, H))
            # Omplir només per a nodes presents
            present = mask.nonzero(as_tuple=False).view(-1).tolist()
            if h_dict:
                idxs = [i for i in present if i in h_dict]
                if idxs:
                    h_prev_vals = torch.stack([h_dict[i] for i in idxs], dim=0)
                    h_prev[idxs] = h_prev_vals

            # 1.2) Extracció edge_weight 1D
            if ea is not None:
                edge_weight = ea[:, 0] if ea.dim() > 1 else ea
            else:
                edge_weight = None

            # 1.3) Càrrega a la cèl·lula TGCN
            h_new = self.tgcn_cell(x, ei, edge_weight, h_prev)

            # 1.4) Actualitzar h_dict sols pels nodes presents
            for i in present:
                h_dict[i] = h_new[i]

        # 2) Decodificació autoregressiva sobre el grafo sencer
        #    Primer h_prev = estat dels nodes presents al darrer pas
        h_prev = x_seq[-1].new_zeros((N_global, H))
        present = mask_seq[-1].nonzero(as_tuple=False).view(-1).tolist()
        idxs = [i for i in present if i in h_dict]
        if idxs:
            vals = torch.stack([h_dict[i] for i in idxs], dim=0)
            h_prev[idxs] = vals

        preds = []
        for _ in range(self.horizon):
            inp = preds[-1] if preds else x_seq[-1]
            if ea_seq[-1] is not None:
                ew = ea_seq[-1][:, 0] if ea_seq[-1].dim() > 1 else ea_seq[-1]
            else:
                ew = None
            h_new = self.tgcn_cell(inp, ei_seq[-1], ew, h_prev)
            y_hat = self.head(h_new)      # → [N_global × out_channels]
            preds.append(y_hat)
            # Per al següent pas, l’estat complet és aquest
            h_prev = h_new

        return torch.stack(preds, dim=0)


class DynTCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, out_channels: int = None, horizon: int = None, kernel_size: int = 3, dilations: list[int] = [1, 2, 4]):
        super().__init__()
        self.hidden_size = hidden
        self.kernel_size = kernel_size
        self.out_channels = out_channels or in_channels
        self.horizon      = horizon or 1

        # Definim explícitament la cèl·lula TCN (amb graph‐conv intern)
        self.tcn_cell = TemporalConvCell(
            in_channels=in_channels,
            hidden_size=hidden,
            kernel_size=kernel_size,
            dilations=dilations
        )

        # Encabezament per mapejar al nombre de sortida
        self.head = nn.Linear(hidden, self.out_channels)

    def forward(self, x_seq, edge_index_seq, edge_attr_seq, mask_seq, id_seq):
        """
        x_seq:    [T] llista de tensors [N_t x F] (features per node)
        edge_index_seq: [T] llista d'arestes de cada timestamp
        edge_attr_seq:  [T] llista d'atributs d'aresta (opcional)
        mask_seq:       [T] llista de màscares [N_t] indicant nodes presents
        id_seq:         [T] llista de lists amb els node-IDs de cada timestamp
        """
        # 1) Diccionari d’història: nid → llista de hidden states (per TCN)
        history: dict[int, List[Tensor]] = {}

        # 2) Processament de la seqüència d’entrada
        for t, (x_t, ei_t, ea_t, mask_t, ids_t) in enumerate(
                zip(x_seq, edge_index_seq, edge_attr_seq, mask_seq, id_seq)
        ):
            # 2.1) Construir el batch "local" per a la cèl·lula:
            #     recollim l’història de cada node i la seva feature actual
            h_prev_list = []
            x_local = []
            for nid, feat in zip(ids_t, x_t):
                # historial amb padding de zeros si no n’hi ha prou
                hist = history.get(nid, [])
                if len(hist) < self.tcn_cell.kernel_size:
                    pad = [x_t.new_zeros(self.tcn_cell.hidden_size)
                           for _ in range(self.tcn_cell.kernel_size - len(hist))]
                    hist = pad + hist
                h_prev_list.append(torch.stack(hist[-self.tcn_cell.kernel_size:], dim=0))
                x_local.append(feat)
            # tensor [N_t × kernel_size × hidden] i [N_t × F]
            h_prev = torch.stack(h_prev_list, dim=0)
            x_local = torch.stack(x_local, dim=0)

            # 2.2) Pas per la cèl·lula TCN (que inclou graph‐conv internament)
            #     retorna h_new: [N_t × hidden_size]
            h_new = self.tcn_cell(x_local, ei_t, ea_t, h_prev)

            # 2.3) Actualitzar l’historial per a cada node
            for i, nid in enumerate(ids_t):
                lst = history.get(nid, [])
                lst.append(h_new[i])
                # mantenim només el nombre de passos necessari
                history[nid] = lst[-self.tcn_cell.kernel_size:]

        # 3) Decodificació autoregressiva per a l’horitzó
        #    utilitzem l’últim timestamp com a input inicial
        last_ids = id_seq[-1]
        # Construïm h_prev per a l’últim timestamp
        h_prev_list = [history[nid][-1] for nid in last_ids]
        h_prev = torch.stack(h_prev_list, dim=0)
        # features d’entrada initial (última x_seq)
        inp = x_seq[-1]

        preds = []
        for _ in range(self.horizon):
            # utilitzem la mateixa cèl·lula per generar el pas següent
            h_new = self.tcn_cell(inp, edge_index_seq[-1],
                                  edge_attr_seq[-1], h_prev)
            y_hat = self.head(h_new)                     # [N_last × out_channels]
            preds.append(y_hat)
            inp = y_hat                                 # autoregressiu
            h_prev = h_new

        # Retornem [Horizon × N_last × out_channels]
        return torch.stack(preds, dim=0)

# ───────────────────────────────── ENTRENAMENT ────────────────────────────────
def split(ds):
    """
    Partició cronològica estricta:
      - train  → seqüències amb any d'inici ≤ 2022
      - val    → seqüències amb any d'inici == 2023
      - test   → seqüències amb any d'inici == 2024
    Si apareix any > 2024, avisem per si hi ha dades inesperades.
    """
    train_idx, val_idx, test_idx = [], [], []
    for i, fp in enumerate(sorted(ds.files)):
        fname = os.path.basename(fp)           # p.ex. '2024122918_2024123117.pt'
        start_str = fname.split('_')[0]        # '2024122918'
        year = int(start_str[:4])
        if year <= 2022:
            train_idx.append(i)
        elif year == 2023:
            val_idx.append(i)
        elif year == 2024:
            test_idx.append(i)
        else:
            # Si mai hi hagués seqüències de 2025 en endavant
            raise ValueError(f"Avis: seqüència inesperada amb any {year} → {fname}")
    return Subset(ds, train_idx), Subset(ds, val_idx), Subset(ds, test_idx)


def get_target_stats(loader):
    """Calcula mitjana i std (vectorials) de y en el conjunt d'entrenament."""
    mean = None
    var_times_n = None
    n_seen = 0
    for _, _, _, _, _, y_b in loader:
        for y in y_b:                      # recórrer un a un (y és [F])
            y = y
            if mean is None:
                mean = y.clone()
                var_times_n = torch.zeros_like(y)
                n_seen = 1
            else:
                mean, var_times_n, n_seen = moving_average(mean, var_times_n, y,
                                                           n_seen)
    std = torch.sqrt(var_times_n / max(n_seen - 1, 1)).clamp(min=STD_EPS)
    return mean, std

def rmse(pred, target):
    return torch.sqrt(nn.functional.mse_loss(pred, target))

def run(loader, model, crit, dev, opt, scheduler, mu, sigma, scaler, desc, args):
    train = opt is not None
    model.train() if train else model.eval()
    sum_loss = 0.0
    ys_true, ys_pred = [], []

    # ⇒ Context per entrenament vs. validació
    ctx = nullcontext() if train else torch.no_grad()
    # ⇒ Usar AMP en tots dos casos per eficàcia
    with ctx, autocast("cuda"):
        for xs_b, eis_b, ea_b, masks_b, ids_b, y_b in tqdm(loader, desc=desc, leave=False):
            # y_b: [batch, H, N, F_out]
            y_b = y_b.to(dev)
            # normalitzem objectiu
            y_norm = (y_b - mu) / sigma
            # preds_norm: list de [H, N, F_out]
            preds_list = []
            for xs_seq, ei_seq, ea_seq, mask_seq, ids_seq in zip(xs_b, eis_b, ea_b, masks_b, ids_b):
                # portar tot a GPU/CPU
                # portem a device
                xs_seq = [x.to(dev) for x in xs_seq]
                ei_seq = [ei.to(dev) for ei in ei_seq]
                # edge attributes opcional
                if args.use_edge_attr:
                    ea_seq_proc = [ea.to(dev) for ea in ea_seq]
                else:
                    ea_seq_proc = [None] * len(ei_seq)
                # mask opcional
                if args.use_mask:
                    mask_seq_proc = [m.to(dev) for m in mask_seq]
                else:
                    mask_seq_proc = [None] * len(ei_seq)
                # cridem el model amb llistes sempre del mateix tamany
                preds = model(xs_seq,
                            ei_seq,
                            ea_seq_proc,
                            mask_seq_proc,
                            ids_seq)

                preds_list.append(preds)

            preds_norm = torch.stack(preds_list)   # [batch, H, N, F_out]
            loss = crit(preds_norm, y_norm)

            if train:
                # Optimitzador només en entrenament
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()

                # Ajusta el learning rate si fas one-cycle
                if args.lr_scheduler == "onecycle":
                    scheduler.step()

            # mètriques en escala original
            preds = preds_norm * sigma + mu
            sum_loss += loss.item()
            # Emmagatzemar per càlcul global de R2 i MAPE
            ys_true.append(y_b.detach().cpu().numpy())
            ys_pred.append(preds.detach().cpu().numpy())

    n = len(loader)
    # Concatena totes les mostres (batch, horitzó, nodes, features)
    y_true = np.concatenate(ys_true, axis=0)   # [n_batches, H, N, F]
    y_pred = np.concatenate(ys_pred, axis=0)   # [n_batches, H, N, F]

    # Aplana batch, temps i nodes en un sol eix de mostra
    batches, H, N, F = y_true.shape
    y_true = y_true.reshape(batches * H * N, F)
    y_pred = y_pred.reshape(batches * H * N, F)

    rmse_global = np.sqrt(mean_squared_error(y_true, y_pred))

    mae_global  = mean_absolute_error(y_true, y_pred)
    r2_global   = r2_score(y_true, y_pred)
    denom = np.where(y_true == 0, STD_EPS, y_true)
    mape_global = np.mean(np.abs((y_true - y_pred) / denom)) * 100

    # Retornem pèrdua mitjana i mètriques globals
    return (sum_loss / n, rmse_global, mae_global, r2_global, mape_global)

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.epochs_no_improve = 0

    def step(self, metric):
        if metric < self.best - self.min_delta:
            self.best = metric
            self.epochs_no_improve = 0
            return False  # no stop
        self.epochs_no_improve += 1
        return self.epochs_no_improve >= self.patience
    
def save_checkpoint(model, optimizer, scheduler, epoch, best_val, save_dir):
    ckpt = {
        "epoch":           epoch,
        "model_state_dict":      model.state_dict(),
        "optimizer_state_dict":  optimizer.state_dict(),
        "best_val_loss":         best_val,
        "scheduler_state_dict":  scheduler.state_dict() if scheduler is not None else None,
    }
    ckpt_name = f"MeteoGraphPC_{datetime.now():%Y%m%d_%H%M%S}.pth"
    path = os.path.join(save_dir, ckpt_name)
    torch.save(ckpt, path)
    return path

# ───────────────────────────────── MAIN ───────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Entrena models basats en xarxes neuronals en grafs sobre seqüències generades per generate_seq_v7.py")
    parser.add_argument("--seq_dir", type=str, default="/fhome/nfarres/DADES_METEO_PC_generated_seqs_v6_ws48_str6_hh6", help="Directori amb les seqüències .pt generades")
    parser.add_argument("--batch_size", type=int, default=8, help="Mida del batch per al DataLoader")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre màxim d'èpoques")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate per l'optimitzador")
    parser.add_argument("--lr_scheduler", choices=["plateau", "onecycle"], default="onecycle", help="Tipus de scheduler per ajustar el learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Dimensió oculta del model")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Clip de gradient")
    parser.add_argument("--patience", type=int, default=15, help="Patience per early stopping")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Millora mínima per resetar patience")
    parser.add_argument("--device", type=str, default="cuda", help="Device per PyTorch ('cuda' o 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Semilla per a la reproducibilitat")
    parser.add_argument("--std_eps", type=float, default=1e-6, help="Petita constant per evitar divisions per zero en la normalització")
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directori on guardar el model entrenat')
    parser.add_argument('--log_csv', type=str, default=None, help='Fitxer CSV per desar el registre d\'entrenament')
    parser.add_argument("--model", choices=["dyntgcn", "dyntcn"], default="dyntgcn", help="Arquitectura: DynTGCN (GCN+GRU) o DynTCN (GCN+TCN dilatat)")
    parser.add_argument('--dl_num_workers', type=int, default=4, help="Nombre de processos/threads per al DataLoader")
    parser.add_argument('--input_indices', type=int, nargs='+', default=None, help="Índexs de columnes dins x_seq a usar com a features d'entrada")
    parser.add_argument('--target_indices', type=int, nargs='+', default=None, help="Índexs de columnes dins y_seq a preveure")
    parser.add_argument('--use_edge_attr', action='store_true', help="Inclou atributs d'arestes a l'entrada al model")
    parser.add_argument('--use_mask', action='store_true', help="Inclou la màscara de nodes inexistents a l'entrada al model")

    return parser.parse_args()

def main():
    args = parse_args()

    # Genera timestamp i crea nom de log si no s'ha passat amb --log_csv
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.log_csv:
        # Assegurem-nos que el directori existeix
        os.makedirs(args.save_dir, exist_ok=True)
        # Guardem el CSV dins de save_dir
        args.log_csv = os.path.join(args.save_dir, f"train_ws48_str6_hh6_log_{timestamp}.csv")

    # Assegura’t que el directori existeixi
    os.makedirs(args.save_dir, exist_ok=True)

    global GRAD_CLIP
    GRAD_CLIP = args.grad_clip

    # petita constant per evitar divisions per zero en la normalització
    global STD_EPS
    STD_EPS = args.std_eps

    set_seed(args.seed)

    ds = GraphSeqDataset(args.seq_dir, num_workers=args.dl_num_workers, input_idx=args.input_indices, target_idx=args.target_indices)
    # Usarem partició cronològica fixa: 2016–2022 train, 2023 val, 2024 test
    tr_ds, vl_ds, te_ds = split(ds)

    num_workers = args.dl_num_workers
    tr_dl = DataLoader(tr_ds, shuffle=True,  batch_size=args.batch_size, collate_fn=collate, num_workers=num_workers)
    vl_dl = DataLoader(vl_ds, shuffle=False, batch_size=args.batch_size, collate_fn=collate, num_workers=num_workers)
    te_dl = DataLoader(te_ds, shuffle=False, batch_size=args.batch_size, collate_fn=collate, num_workers=num_workers)
    
    print(f"Seqüències: total={len(ds)}, train={len(tr_ds)}, val={len(vl_ds)}, test={len(te_ds)}")
    print("Mostra algunes seqüències de test:", [os.path.basename(ds.files[i]) for i in te_ds.indices[:5]])

    # stats target
    mu, sigma = get_target_stats(tr_dl)
    mu, sigma = mu.to(args.device), sigma.to(args.device)

    F_in  = len(args.input_indices) if args.input_indices else ds[0][0][0].size(1)
    F_out = len(args.target_indices) if args.target_indices else F_in
    # Inferim horizon a partir del primer .pt de seq_dir
    sample = torch.load(ds.files[0], map_location="cpu")
    H = len(sample["y_seq"]) # nombre de passos futurs creats


    if args.model == "dyntgcn":
        model = DynTGCN(in_channels=F_in, hidden=args.hidden_dim, out_channels=F_out, horizon=H).to(args.device)
    elif args.model == "dyntcn":
        model = DynTCN( in_channels=F_in, hidden=args.hidden_dim, out_channels=F_out, horizon=H).to(args.device)
    else:
        raise ValueError(f"Model desconegut: {args.model}")

    #if torch.cuda.device_count() > 1:
    #    model = GeoDataParallel(model)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    scaler = GradScaler()

    if args.lr_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=args.patience, min_lr=1e-6)
    elif args.lr_scheduler == "onecycle":
        scheduler = OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=len(tr_dl), epochs=args.epochs, pct_start=0.3, div_factor=25.0, final_div_factor=1e4)

    crit  = nn.MSELoss()
    stopper = EarlyStopper(args.patience, args.min_delta)

    # Preparar fitxer de log
    with open(args.log_csv, "w", newline="") as f:
        w = csv.writer(f)
        # 1) Timestamp de la run
        w.writerow(["run_timestamp", timestamp])
        # 2) Hiperparàmetres
        w.writerow([
            "batch_size", args.batch_size,
            "lr", args.lr,
            "epochs", args.epochs,
            "hidden_dim", args.hidden_dim,
            "patience", args.patience,
            "min_delta", args.min_delta,
            "grad_clip", args.grad_clip,
            "std_eps", args.std_eps
        ])
        # 3) Mides dels conjunts
        w.writerow([
            "total_sequences", len(ds),
            "train_size", len(tr_ds),
            "val_size", len(vl_ds),
            "test_size", len(te_ds)
        ])
        # 4) Capçalera de les mètriques per època
        w.writerow(["epoch", "stage", "loss", "RMSE", "MAE", "R2", "MAPE"])

    for epoch in range(1, args.epochs+1):
            tr_loss, tr_rmse, tr_mae, tr_r2, tr_mape = run(tr_dl, model, crit, args.device, opt, scheduler, mu, sigma, scaler, desc=f"[{epoch}/{args.epochs}] Train", args=args)
            vl_loss, vl_rmse, vl_mae, vl_r2, vl_mape = run(vl_dl, model, crit, args.device, None, scheduler, mu, sigma, scaler, desc=f"[{epoch}/{args.epochs}] Val ", args=args)
            
            if args.lr_scheduler == "plateau":
                scheduler.step(vl_loss)

            current_lr = scheduler.get_last_lr()[0]
            print(f"[Epoch {epoch:2d}] Current LR: {current_lr:.2e}")

            print(f"Època {epoch:3d} | "
                  f"Train  (loss {tr_loss:.4f}, RMSE {tr_rmse:.4f}, MAE {tr_mae:.4f}) · "
                  f"Val  (loss {vl_loss:.4f}, RMSE {vl_rmse:.4f}, MAE {vl_mae:.4f})")

            # Guardem train i val amb tots els indicadors
            with open(args.log_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([epoch, "train", tr_loss, tr_rmse, tr_mae, tr_r2,  tr_mape])
                w.writerow([epoch, "val",   vl_loss, vl_rmse, vl_mae,   vl_r2,  vl_mape])

            # guardar millor model segons RMSE de validació desnormalitzada
            if vl_rmse < stopper.best - args.min_delta:
                # fem servir el helper per desar tot el checkpoint (model, opt, sched, epoch, best_val)
                best_ckpt_path = save_checkpoint(
                    model, opt, scheduler, epoch, vl_rmse, args.save_dir
                )
                print(f"  ↳ Millor val_RMSE! Checkpoint desat a: {best_ckpt_path}")

            if stopper.step(vl_rmse):
                print(f"Early stopping (sense millora {args.patience} èpoques).")
                break
            
            # Esborrem la memòria GPU per evitar errors de memòria
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 1) Carrega el millor model guardat
    checkpoint = torch.load(best_ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()


    # 2) Avaluació completa
    ys_true, ys_pred = [], []
    ys_persist, ys_climat = [], []

    with torch.no_grad():
        for xs_b, eis_b, ea_b, masks_b, ids_b, y_b in te_dl:
            y_b = y_b.to(args.device)
            preds_list = []
            for xs_seq, ei_seq, ea_seq, mask_seq, ids_seq in zip(xs_b, eis_b, ea_b, masks_b, ids_b):
                # 1) portem tot a device
                xs_seq   = [x.to(args.device)   for x in xs_seq]
                ei_seq   = [ei.to(args.device)  for ei in ei_seq]
                if args.use_edge_attr:
                    ea_seq_proc = [ea.to(args.device) for ea in ea_seq]
                else:
                    ea_seq_proc = [None] * len(ei_seq)

                if args.use_mask:
                    mask_seq_proc = [m.to(args.device) for m in mask_seq]
                else:
                    mask_seq_proc = [None] * len(ei_seq)

                preds = model(xs_seq,
                            ei_seq,
                            ea_seq_proc,
                            mask_seq_proc,
                            ids_seq)

                preds_list.append(preds)
                
            preds = torch.stack(preds_list) * sigma + mu
            ys_true.append(y_b.cpu().numpy())
            ys_pred.append(preds.cpu().numpy())

            # -- Baseline persistència i climatologia per a cada mostra del batch --
            for seq_x, y_true in zip(xs_b, y_b.cpu()):
                # — Persistència per node: l’últim estat tal qual —
                persist_pred = seq_x[-1].cpu().numpy()               # ara [N, F]
                ys_persist.append(persist_pred)
                
                # — Climatologia per node: repliquem la mitjana global —
                mu_np = mu.cpu().numpy()                        # [F]
                N_nodes = persist_pred.shape[0]
                climat_pred = np.tile(mu_np[None, :], (N_nodes, 1))  # [N, F]
                ys_climat.append(climat_pred)


    # ——————————— CONCATA I PREPARA PER TEST FINAL ———————————

    # 1) Concatena totes les seqüències de test    
    ys_true = np.concatenate(ys_true, axis=0)   # [S, H, N, F]
    ys_pred = np.concatenate(ys_pred, axis=0)   # [S, H, N, F]

    # Extreu dimensions
    S, H, N, F = ys_true.shape
    mu_np = mu.cpu().numpy()

    # Guardo prediccions i valors reals per a anàlisi posterior
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, "y_true_test.npy"), ys_true)
    np.save(os.path.join(args.save_dir, "y_pred_test.npy"), ys_pred)

    # 2) Concatena els baselines originals (un sol pas)  
    ys_persist = np.stack(ys_persist, axis=0)   # [S, N, F]

    # 4) Replica cada baseline per a tots els H passos de l’horitzó  
    persist_np = np.stack([
        np.repeat(p[None, ...], H, axis=0)       # de [N, F] -> [H, N, F]
        for p in ys_persist
    ], axis=0)                                  # [S, H, N, F]

    print("persist_np.shape =", persist_np.shape)

    # Aplanem tota la primera dimensió (seqüències × temps × nodes)
    y_true_flat  = ys_true.reshape(-1, ys_true.shape[-1])
    y_pred_flat  = ys_pred.reshape(-1, ys_pred.shape[-1])
    persist_flat = persist_np.reshape(-1, persist_np.shape[-1])
    # Aplanem les dues primeres dimensions i calculem mitjana per feature:
    mu_feat = mu_np.reshape(-1, mu_np.shape[-1]).mean(axis=0)  # shape (F,)
    # 3) Climatologia: repeteixo mu_np una fila per cada mostra de y_true_flat
    climat_flat = np.repeat(mu_feat[None, :], y_true_flat.shape[0], axis=0)

    # 6) Calcula les mètriques  
    rmse_persist = np.sqrt(mean_squared_error(y_true_flat, persist_flat))
    mae_persist  = mean_absolute_error   (y_true_flat, persist_flat)
    r2_persist   = r2_score              (y_true_flat, persist_flat)
    denom        = np.where(y_true_flat == 0, STD_EPS, y_true_flat)
    mape_persist = np.mean(np.abs((y_true_flat - persist_flat) / denom)) * 100

    rmse_clima   = np.sqrt(mean_squared_error(y_true_flat, climat_flat))
    mae_clima    = mean_absolute_error   (y_true_flat, climat_flat)
    r2_clima     = r2_score              (y_true_flat, climat_flat)
    mape_clima   = np.mean(np.abs((y_true_flat - climat_flat) / denom)) * 100

    rmse         = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae          = mean_absolute_error   (y_true_flat, y_pred_flat)
    r2           = r2_score              (y_true_flat, y_pred_flat)
    mape         = np.mean(np.abs((y_true_flat - y_pred_flat) / denom)) * 100

    # 7) Escriu al CSV i printeja per pantalla  
    with open(args.log_csv, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "test",         "", rmse,        mae,        r2,        mape])
        w.writerow(["", "persistència", "", rmse_persist, mae_persist, r2_persist, mape_persist])
        w.writerow(["", "climatologia",  "", rmse_clima,   mae_clima,   r2_clima,   mape_clima])

    print(f"Test metrics →       RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
    print(
        f"Baseline persistència -> "
        f"RMSE: {rmse_persist:.4f}, "
        f"MAE: {mae_persist:.4f}, "
        f"R²: {r2_persist:.4f}, "
        f"MAPE: {mape_persist:.2f}%"
    )
    print(
        f"Baseline climatologia  -> "
        f"RMSE: {rmse_clima:.4f}, "
        f"MAE: {mae_clima:.4f}, "
        f"R²: {r2_clima:.4f}, "
        f"MAPE: {mape_clima:.2f}%"
    )

    print(f"Entrenament acabat. Millor val_RMSE = {stopper.best:.4f}")
    print(f"Log complet al fitxer «{args.log_csv}».")

if __name__ == "__main__":
    main()
