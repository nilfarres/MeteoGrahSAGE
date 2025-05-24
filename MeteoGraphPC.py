#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeteoGraphPC.py
"""

from __future__ import annotations
import csv, glob, os, random, argparse
from datetime import datetime
import sys
import json

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
from torch_geometric.nn import NNConv, GATConv

from torch.amp import autocast
from torch.cuda.amp import GradScaler

from torch.utils.checkpoint import checkpoint
from contextlib import nullcontext

from torch_geometric.utils import softmax

import torch.multiprocessing as tmp_mp
tmp_mp.set_sharing_strategy('file_system')

import multiprocessing as mp

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
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True

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

# ───────────────────────────────── DATASET ────────────────────────────────────
class GraphSeqDataset(Dataset):
    def __init__(self, seq_dir: str, num_workers: int | None = None,
                 input_idx: list[int] = None, target_idx: list[int] = None):
        # Només carreguem els fitxers petits de metadades!
        self.meta_files = sorted(glob.glob(os.path.join(seq_dir, "chunk_*_meta.pt")))
        assert self.meta_files, f"No s'han trobat fitxers de metadades a {seq_dir}!"

        all_pt = glob.glob(os.path.join(seq_dir, "chunk_*.pt"))
        # Només els chunks reals, no els *_meta.pt
        self.chunk_files = sorted([f for f in all_pt if not f.endswith("_meta.pt")])
        assert self.chunk_files, f"No s'han trobat fitxers chunk a {seq_dir}!"

        self.indices = []
        self.filenames = []

        for i, meta_fp in enumerate(self.meta_files):
            meta = torch.load(meta_fp, map_location="cpu")
            chunk_fnames = meta.get("filenames", [None] * len(meta["filenames"]))
            normalized = [fn.replace('\\', '/').split('/')[-1] if fn is not None else None
                          for fn in chunk_fnames]
            self.filenames.extend(normalized)
            n = len(chunk_fnames)
            for j in range(n):
                self.indices.append((i, j))

        self.input_idx = input_idx
        self.target_idx = target_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        chunk_idx, seq_idx = self.indices[idx]
        # Ara només carreguem el chunk gran quan cal
        seqs = torch.load(self.chunk_files[chunk_idx], map_location="cpu")["sequences"]
        d = seqs[seq_idx]

        # Construcció de y_seq
        y_seq_full = d["y_seq"]
        if self.target_idx:
            y_seq = [y_t[:, self.target_idx] for y_t in y_seq_full]
        else:
            y_seq = y_seq_full

        # Neteja d'edge_index_seq
        clean_ei_seq = []
        for ei in d["edge_index_seq"]:
            if ei.dim() == 1 or (ei.dim() == 2 and ei.size(0) == 1):
                ei = torch.empty((2, 0), dtype=torch.long, device=ei.device)
            clean_ei_seq.append(ei)

        # X_seq i la resta
        x_seq = [x[:, self.input_idx] if self.input_idx else x for x in d["x_seq"]]
        id_seq = d["id_seq"]
        ea_seq = d["edge_attr_seq"]
        mask_seq = d["mask_seq"]

        return x_seq, clean_ei_seq, ea_seq, mask_seq, id_seq, y_seq


class TemporalConvCell(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        edge_dim: int,
        kernel_size: int = 3,
        dilations: list[int] = [1, 2, 4],
        p_dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        # Xarxa per transformar edge_attr en matriu de pesos per cada aresta
        nn_edge = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, in_channels * hidden_size)
        )

        # Graph convolution (NNConv, compatible amb edge_attr dinàmic)
        self.graph_conv = NNConv(
            in_channels,
            hidden_size,
            nn_edge,
            aggr='mean'    # pots fer servir 'add', 'mean', 'max' segons et convingui
        )

        # TCN layers amb diverses dilacions (memòria temporal multiescala)
        self.tcn_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * d // 2,
                dilation=d
            )
            for d in dilations
        ])
        self.out_proj = nn.Linear(hidden_size * len(dilations), hidden_size)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x, edge_index, edge_attr, h_prev=None):
        """
        Args:
            x:         [N, in_channels]         Node features
            edge_index:[2, E]                   Edge connectivity (dinàmic)
            edge_attr: [E, edge_dim]            Edge attributes (dinàmic)
            h_prev:    [N, kernel_size, hidden] Historial per node
        Returns:
            h_out:     [N, hidden]              Nous hidden states per node
            h_hist:    [N, kernel_size, hidden] Historial actualitzat
        """
        N = x.size(0)

        # 1. Graph convolution per a embedding espacial
        h_graph = self.graph_conv(x, edge_index, edge_attr)  # [N, hidden_size]
        h_graph = F.relu(h_graph)

        # 2. Prepara historial temporal per la TCN
        if h_prev is None:
            h_prev = torch.zeros(N, self.kernel_size, self.hidden_size, device=x.device)
        # Actualitza historial: desplaça i afegeix el nou embedding
        h_hist = torch.cat([h_prev[:, 1:], h_graph.unsqueeze(1)], dim=1)  # [N, kernel_size, hidden_size]

        # 3. Aplica TCN a cada node sobre l’historial
        tcn_outputs = []
        h_input = h_hist.transpose(1, 2)  # [N, hidden_size, kernel_size] (per Conv1d)
        for tcn_layer in self.tcn_layers:
            h_tcn = tcn_layer(h_input)    # [N, hidden_size, kernel_size]
            tcn_outputs.append(h_tcn[:, :, -1])  # [N, hidden_size] (últim pas temporal)

        # 4. Concatenació i projecció
        h_tcn_concat = torch.cat(tcn_outputs, dim=1)   # [N, hidden_size * num_dilations]
        h_out = self.out_proj(h_tcn_concat)            # [N, hidden_size]
        h_out = self.dropout(h_out)                    # Regularització

        return h_out, h_hist    # Retorna nou embedding i historial per continuar la seqüència
    
class TemporalDilatedCNN(nn.Module):
    """CNN temporal dilatada per captar dinàmica a llarg termini."""
    def __init__(self, d_in, d_hidden, num_layers=4, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2**i
            layers.append(nn.Conv1d(d_in if i==0 else d_hidden, d_hidden,
                                    kernel_size=kernel_size, padding='same', dilation=dilation))
            layers.append(nn.LayerNorm([d_hidden, -1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x, mask=None):
        # x: [B, T, N, F] → [B*N, F, T]
        B, T, N, F = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B*N, F, T)
        x = self.net(x)  # [B*N, H, T]
        x = x.permute(0, 2, 1).reshape(B, N, T, -1).permute(0, 2, 1, 3)  # [B, T, N, H]
        return x

class SpatialAttentionBlock(nn.Module):
    """Atenció espacial entre nodes, ponderant missatges de veïns."""
    def __init__(self, d_in, d_edge, d_hidden, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.lin_node = nn.Linear(d_in, d_hidden * heads, bias=False)
        self.lin_edge = nn.Linear(d_edge, d_hidden * heads, bias=False)
        self.lin_out = nn.Linear(d_hidden * heads, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_hidden)
    def forward(self, x, edge_index, edge_attr, mask=None):
        # x: [N, d_in], edge_index: [2, E], edge_attr: [E, d_edge]
        N = x.size(0)
        h = self.heads
        q = self.lin_node(x).view(N, h, -1)
        src, dst = edge_index
        e = self.lin_edge(edge_attr).view(-1, h, q.size(-1))
        alpha = (q[dst] * (q[src] + e)).sum(-1) / (q.size(-1)**0.5)
        alpha = softmax(alpha, dst)
        alpha = self.dropout(alpha)
        msg = q[src] + e
        out = torch.zeros_like(q)
        out.index_add_(0, dst, msg * alpha.unsqueeze(-1))
        out = out.view(N, -1)
        out = self.lin_out(out)
        out = self.ln(out + x)
        return out
    
class TemporalTransformerBlock(nn.Module):
    """Bloc transformer temporal per cada node."""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, key_padding_mask=None):
        # x: [B, T, N, F] → [B*N, T, F]
        B, T, N, F = x.shape
        x = x.permute(0,2,1,3).reshape(B*N, T, F)
        if key_padding_mask is not None:
            # mask: [B, T, N] → [B*N, T], True indica "ignora"
            mask = ~key_padding_mask.permute(0,2,1).reshape(B*N, T)
        else:
            mask = None
        out, _ = self.attn(x, x, x, key_padding_mask=mask)
        out = self.ln(x + self.dropout(out))
        # [B*N, T, F] → [B, T, N, F]
        out = out.reshape(B, N, T, F).permute(0,2,1,3)
        return out

class GATwithEdgeAttr(nn.Module):
    """Bloc GATConv millorada per considerar edge_attr a cada pas temporal."""
    def __init__(self, in_dim, out_dim, edge_dim, heads=2, dropout=0.1):
        super().__init__()
        # Hi ha GATv2Conv amb edge_attr, però la GATConv "bàsica" no l'usa.
        # Si vols afegir edge_attr, pots fer-ho com un "concatenat" a l'input
        # Alternativament, pots trobar GATv2Conv a torch_geometric>=2.4
        self.edge_proj = nn.Linear(edge_dim, in_dim)
        self.gat = GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=dropout)
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index, edge_attr):
        # edge_attr: [E, F_e] → [E, in_dim]
        edge_emb = self.edge_proj(edge_attr)
        # x: [N, in_dim]
        # Passa edge_emb com "edge_weight" (és un hack; pel GATv2Conv oficial pots passar-ho natiu)
        out = self.gat(x, edge_index, edge_emb)
        out = self.ln(self.dropout(out) + x)
        return out
    

# ───────────────────────────────── MODELS ──────────────────────────────────────


class MeteoGraphPC_v1(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, out_channels: int = None, horizon: int = None, p_dropout: float = 0.2):
        super().__init__()
        # cèl·lula temporal TGCN que combina GCN + GRU
        self.tgcn_cell = TGCN(in_channels, hidden)
        self.hidden_size = hidden
        self.out_channels = out_channels or in_channels
        self.dropout = nn.Dropout(p=p_dropout)
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
            h_new = self.dropout(h_new)
            y_hat = self.head(h_new)      # → [N_global × out_channels]
            preds.append(y_hat)
            # Per al següent pas, l’estat complet és aquest
            h_prev = h_new

        return torch.stack(preds, dim=0)


class MeteoGraphPC_v2(nn.Module):
    """
    Model seqüencial per a grafs dinàmics, amb TCN + GNN.
    Dissenyat per a predicció meteorològica multi-pas (multi-step forecasting)
    amb finestres d'entrada i grafs canviants.
    """
    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        out_channels: int = None,
        horizon: int = None,
        kernel_size: int = 3,
        dilations: list[int] = [1, 2, 4],
        edge_dim: int = 1,
        p_dropout: float = 0.2
    ):
        """
        Args:
            in_channels:   Nombre de features d'entrada per node.
            hidden:        Dimensió oculta per embedding.
            out_channels:  Nombre de variables a predir (per node).
            horizon:       Nombre de passos de temps a predir.
            kernel_size:   Longitud de la memòria temporal de la TCN.
            dilations:     Dilacions temporals de les convolucions.
            edge_dim:      Dimensió dels atributs d'aresta.
            p_dropout:     Dropout per regularització.
        """
        super().__init__()
        self.hidden_size = hidden
        self.kernel_size = kernel_size
        self.out_channels = out_channels or in_channels
        self.horizon = horizon or 1

        # Cèl·lula TCN + GNN recurrent
        self.tcn_cell = TemporalConvCell(
            in_channels=in_channels,
            hidden_size=hidden,
            edge_dim=edge_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            p_dropout=p_dropout
        )
        # Capçalera per convertir de hidden a out_channels
        self.head = nn.Linear(hidden, self.out_channels)

        # Mapping de node-ID -> índex global a inicialitzar en el primer forward
        self.node_id_to_idx: dict[int, int] | None = None

    def forward(
        self,
        x_seq: list[torch.Tensor],
        edge_index_seq: list[torch.Tensor],
        edge_attr_seq: list[torch.Tensor],
        mask_seq: list[torch.Tensor],
        id_seq: list[list[int]]
    ) -> torch.Tensor:
        """
        Args:
            x_seq:           Llista de tensors [N_u_global x F] d'entrada.
            edge_index_seq:  Llista de tensors [2 x E].
            edge_attr_seq:   Llista d'atributs d'aresta per timestamp.
            mask_seq:        Llista de màscares de nodes.
            id_seq:          Llista de llistes amb node-IDs per timestamp.
        Returns:
            Tensor de prediccions de mida [horizon x N_last x out_channels].
        """
        # Construïm mapping de node-ID -> idx global en el primer timestamp
        if self.node_id_to_idx is None:
            all_ids = id_seq[0]
            self.node_id_to_idx = {nid: i for i, nid in enumerate(all_ids)}

        # Historial de hidden states per node
        history: dict[int, list[torch.Tensor]] = {}

        # Passada d'encoding seqüencial
        for x_t, ei_t, ea_t, mask_t, ids_t in zip(
            x_seq, edge_index_seq, edge_attr_seq, mask_seq, id_seq
        ):
            h_prev_list = []
            x_local_list = []

            # Extraiem features locals i construïm historial per a cada node
            for nid in ids_t:
                idx_global = self.node_id_to_idx[nid]
                feat = x_t[idx_global]
                x_local_list.append(feat)

                hist = history.get(nid, [])
                if len(hist) < self.kernel_size:
                    pad = [
                        x_t.new_zeros(self.hidden_size)
                        for _ in range(self.kernel_size - len(hist))
                    ]
                    hist = pad + hist
                h_prev_list.append(torch.stack(hist[-self.kernel_size:], dim=0))

            # Convertim a tensors
            x_local = torch.stack(x_local_list, dim=0)   # [N_t, F]
            h_prev = torch.stack(h_prev_list, dim=0)    # [N_t, K, hidden]

            # Cèl·lula TCN + GNN
            h_new, _ = self.tcn_cell(x_local, ei_t, ea_t, h_prev)

            # Actualitzem historial de cada node
            for i, nid in enumerate(ids_t):
                lst = history.get(nid, [])
                lst.append(h_new[i])
                history[nid] = lst[-self.kernel_size:]

        # Decodificació autoregressiva
        last_ids = id_seq[-1]
        h_prev_list = [history[nid][-1] for nid in last_ids]
        h_prev = torch.stack(h_prev_list, dim=0)
        # Input inicial: features de l'últim timestamp per als last_ids
        inp = x_seq[-1][[self.node_id_to_idx[nid] for nid in last_ids]]

        preds = []
        for _ in range(self.horizon):
            h_new, _ = self.tcn_cell(inp, edge_index_seq[-1], edge_attr_seq[-1], h_prev)
            y_hat = self.head(h_new)           # [N_last, out_channels]
            preds.append(y_hat)
            inp = y_hat                         # autoregressiu
            h_prev = h_new

        # Retornem predictions [horizon, N_last, out_channels]
        return torch.stack(preds, dim=0)
    

class MeteoGraphPC_v3(nn.Module):
    """
    Model avançat: CNN temporal dilatada + GCN + atenció espacial + residuals
    100% compatible amb les teves seqüències i loader!
    """
    def __init__(self, n_feat, n_edge_feat, n_targets, hidden_dim=128, num_layers=3,
                 tcn_layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_feat, hidden_dim)
        self.temporal = TemporalDilatedCNN(hidden_dim, hidden_dim,
                                           num_layers=tcn_layers, dropout=dropout)
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.satt_blocks = nn.ModuleList([
            SpatialAttentionBlock(hidden_dim, n_edge_feat, hidden_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, n_targets)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq, edge_index_seq, edge_attr_seq, mask_seq=None):
        """
        x_seq: [B, T, N, F] — seqüència d'entrada (features)
        edge_index_seq: [T][2, E] — llista per pas temporal
        edge_attr_seq: [T][E, d_edge]
        mask_seq: [B, T, N] — màscara de nodes vàlids
        """
        B, T, N, F = x_seq.shape
        # 1. Projecció feature
        h = self.input_proj(x_seq)   # [B, T, N, H]
        # 2. CNN temporal per captar dependències llargues/curtes
        h = self.temporal(h)         # [B, T, N, H]
        # 3. Itera seqüència, graf per graf
        outs = []
        for t in range(T):
            ht = h[:, t]  # [B, N, H]
            # Fusiona GCN i atenció espacial
            for i, (gcn, attn) in enumerate(zip(self.gcn_layers, self.satt_blocks)):
                h_g = []
                for b in range(B):
                    # GCN
                    g = F.relu(gcn(ht[b], edge_index_seq[t]))
                    # Atenció espacial
                    g = attn(g, edge_index_seq[t], edge_attr_seq[t])
                    h_g.append(g)
                ht = torch.stack(h_g, dim=0)
                ht = self.dropout(ht)
            outs.append(ht)
        h_out = torch.stack(outs, dim=1)  # [B, T, N, H]
        h_out = self.ln(h_out)
        # 4. Output linear
        y_pred = self.head(h_out)  # [B, T, N, n_targets]
        # (Opcional: màscara per nodes absents)
        if mask_seq is not None:
            y_pred = y_pred * mask_seq.unsqueeze(-1)
        return y_pred

# ───────────────────────────────── ENTRENAMENT ────────────────────────────────
def split(ds): 
    """
    Cal modificar aquesta funció si es treballa experimentalment amb molt poques de només un any en concret. 
    La partició cronològica actual és la següent:
      - train  → seqüències amb any d'inici ≤ 2022
      - val    → seqüències amb any d'inici == 2023
      - test   → seqüències amb any d'inici == 2024
    """
    train_idx, val_idx, test_idx = [], [], []
    for i, fname in enumerate(ds.filenames):
        start_str = fname.split('_')[0]
        year = int(start_str[:4])
        if year <= 2022:
            train_idx.append(i)
        elif year == 2023:
            val_idx.append(i)
        elif year == 2024:
            test_idx.append(i)
        else:
            raise ValueError(f"Avis: seqüència inesperada amb any {year} → {fname}")
    return Subset(ds, train_idx), Subset(ds, val_idx), Subset(ds, test_idx)

class MeteoGraphPC_v4(nn.Module):
    """
    Model híbrid: Transformer temporal per node + GAT espacial amb edge_attr
    Totalment coherent amb la teva cadena de generació i entrenament!
    """
    def __init__(self,
                 n_feat,         # Input node features
                 n_edge_feat,    # Input edge features
                 n_targets,      # Output dimension
                 hidden_dim=128,
                 num_gat_layers=2,
                 t_transformer_layers=2,
                 n_heads=4,
                 dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_feat, hidden_dim)
        self.temporal_blocks = nn.ModuleList([
            TemporalTransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(t_transformer_layers)
        ])
        self.gat_blocks = nn.ModuleList([
            GATwithEdgeAttr(hidden_dim, hidden_dim, n_edge_feat, heads=2, dropout=dropout)
            for _ in range(num_gat_layers)
        ])
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, n_targets)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq, edge_index_seq, edge_attr_seq, mask_seq=None, id_seq=None):
        """
        x_seq: [B, T, N, F]
        edge_index_seq: [T][2, E_t]
        edge_attr_seq:  [T][E_t, F_e]
        mask_seq: [B, T, N]  (True=presència, False=absència)
        id_seq: Llista de llistes amb node-IDs per timestamp (opcional; s'ignora).
        """
        B, T, N, F = x_seq.shape
        # 1. Projecció features
        h = self.input_proj(x_seq)   # [B, T, N, H]
        # 2. Transformer temporal (per node)
        for block in self.temporal_blocks:
            h = block(h, key_padding_mask=mask_seq if mask_seq is not None else None)
        # 3. GAT espacial a cada timestamp (batch per separat)
        outs = []
        for t in range(T):
            ht = h[:, t]  # [B, N, H]
            edge_idx = edge_index_seq[t]
            edge_attr = edge_attr_seq[t]
            # Per lots (Batches), processem cada element del batch separat
            ht_out = []
            for b in range(B):
                x_b = ht[b]
                # Si mask_seq existeix, posem a zero els nodes absents
                if mask_seq is not None and not mask_seq[:,t].all():
                    x_b = x_b * mask_seq[b, t].unsqueeze(-1)
                for gat in self.gat_blocks:
                    x_b = gat(x_b, edge_idx, edge_attr)
                ht_out.append(x_b)
            outs.append(torch.stack(ht_out, dim=0))  # [B, N, H]
        h_out = torch.stack(outs, dim=1)  # [B, T, N, H]
        h_out = self.ln(h_out)
        y_pred = self.head(self.dropout(h_out))  # [B, T, N, n_targets]
        # Aplica la màscara (opcional)
        if mask_seq is not None:
            y_pred = y_pred * mask_seq.unsqueeze(-1)
        return y_pred


# ────────────────────────────────────────────────────────────────────


def get_target_stats(loader, device):
    """
    Calcula la mitjana i desviació estàndard per feature del target y,
    agregant totes les mostres del loader.

    Args:
        loader: DataLoader que retorna tuples on l'últim element és y_b de mida [B, H, N, F].
        device: Device on moure els tensors resultants.

    Retorna:
        mu: tensor de forma [1, F] amb la mitjana de cada feature.
        sigma: tensor de forma [1, F] amb la desviació estàndard (clamp a STD_EPS).
    """
    all_y = []
    for batch_idx, (*_, y_b) in enumerate(tqdm(loader, desc="Target stats", unit="batch", file=sys.stdout, flush=True, disable=False)):
        # y_b: [batch_size, H, N, F]
        B, H, N, F = y_b.shape
        # Aplana batch, temps i nodes en un sol eix: [B*H*N, F]
        all_y.append(y_b.reshape(-1, F).cpu())

    # Concatena totes les mostres: [M, F]
    all_y = torch.cat(all_y, dim=0)
    # Calcula mitjana i desviació estàndard per cada feature
    mu = all_y.mean(dim=0, keepdim=True).to(device)  # [1, F]
    sigma = all_y.std(dim=0, unbiased=True)
    sigma = sigma.clamp(min=STD_EPS).to(device)       # [1, F]

    return mu, sigma

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
        for xs_b, eis_b, ea_b, masks_b, ids_b, y_b in tqdm(loader, desc=desc, leave=False, file=sys.stdout, disable=False):
            # y_b: [batch, H, N, F_out]
            y_b = y_b.to(dev)
            # normalitzem objectiu
            y_norm = (y_b - mu) / sigma            # broadcasta correctament
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
    def __init__(self, patience=5, min_delta=1e-4):
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
    parser = argparse.ArgumentParser(description="Entrena models basats en xarxes neuronals en grafs sobre seqüències generades per generate_seq_v8.py")
    parser.add_argument("--seq_dir", type=str, default="/fhome/nfarres/All_Sequences_v8_ws48_str6_hh6_CHUNK", help="Directori amb chunks de seqüències .pt generades")
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
    parser.add_argument("--model", choices=["MeteoGraphPC_v1", "MeteoGraphPC_v2", "MeteoGraphPC_v3", "MeteoGraphPC_v4"], default="MeteoGraphPC_v1", help="Arquitectura: MeteoGraphPC_v1 (GCN+GRU), MeteoGraphPC_v2 (GCN+TCN dilatat), MeteoGraphPC_v3 (GCN+TCN+atenció) o MeteoGraphPC_v4 (Transformer+GAT)")
    parser.add_argument('--dl_num_workers', type=int, default=4, help="Nombre de processos/threads per al DataLoader")
    parser.add_argument('--input_indices', type=int, nargs='+', default=None, help="Índexs de columnes dins x_seq a usar com a features d'entrada")
    parser.add_argument('--target_indices', type=int, nargs='+', default=None, help="Índexs de columnes dins y_seq a preveure")
    parser.add_argument('--use_edge_attr', action='store_true', help="Inclou atributs d'arestes a l'entrada al model")
    parser.add_argument('--use_mask', action='store_true', help="Inclou la màscara de nodes inexistents a l'entrada al model")
    parser.add_argument('--norm_json', type=str, default=None, help="Fitxer JSON amb mean/std per a cada feature (si es vol carregar enlloc de recalcular)")

    return parser.parse_args()

def main():
    args = parse_args()

    # Genera timestamp i crea nom de log si no s'ha passat amb --log_csv
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.log_csv:
        # Assegurem-nos que el directori existeix
        os.makedirs(args.save_dir, exist_ok=True)
        # Guardem el CSV dins de save_dir
        args.log_csv = os.path.join(args.save_dir, f"train_log_{timestamp}.csv")

    # Assegura’t que el directori existeixi
    os.makedirs(args.save_dir, exist_ok=True)

    global GRAD_CLIP
    GRAD_CLIP = args.grad_clip

    # petita constant per evitar divisions per zero en la normalització
    global STD_EPS
    STD_EPS = args.std_eps

    set_seed(args.seed)

    all_sequences_path = args.seq_dir  # Ara aquest argument serà el directori de chunks
    print(f"Carregant seqüències de {all_sequences_path}...")
    ds = GraphSeqDataset(all_sequences_path, input_idx=args.input_indices, target_idx=args.target_indices)

    # Usarem partició cronològica fixa: 2016–2022 train, 2023 val, 2024 test
    tr_ds, vl_ds, te_ds = split(ds)

    num_workers = args.dl_num_workers
    print("Carregant DataLoaders...")
    tr_dl = DataLoader(tr_ds, shuffle=False,  batch_size=args.batch_size, collate_fn=collate, num_workers=num_workers, pin_memory=True, persistent_workers=False, prefetch_factor=2)
    vl_dl = DataLoader(vl_ds, shuffle=False, batch_size=args.batch_size, collate_fn=collate, num_workers=num_workers, pin_memory=True, persistent_workers=False, prefetch_factor=2)
    te_dl = DataLoader(te_ds, shuffle=False, batch_size=args.batch_size, collate_fn=collate, num_workers=num_workers, pin_memory=True, persistent_workers=False, prefetch_factor=2)
    
    print("Mostra algunes seqüències de test:", [ds.filenames[i] for i in te_ds.indices[:5]])

    # 1) Calcula F_in i F_out d’entrada
    F_in  = len(args.input_indices)  if args.input_indices  else ds[0][0][0].size(1)
    F_out = len(args.target_indices) if args.target_indices else F_in

    # 2) Si has passat JSON, carrega mu/sigma directament
    if args.norm_json:
        # Carrega i construeix mu/sigma des del JSON
        print(f"Carregant normalització de {args.norm_json}...")
        with open(args.norm_json, 'r') as f:
            norm = json.load(f)
        # Llista ordenada de noms de features (ha de coincidir amb el JSON!)
        feature_names = [
            "Temp","Humitat","Pluja","VentFor","Patm",
            "Alt_norm","VentDir_sin","VentDir_cos",
            "hora_sin","hora_cos","dia_sin","dia_cos",
            "cos_sza","DewPoint","PotentialTemp","Vent_u","Vent_v"
        ]
        means = [norm[n]["mean"] for n in feature_names]
        stds  = [norm[n]["std"]  for n in feature_names]
        mu    = torch.tensor(means, device=args.device).unsqueeze(0)
        sigma = torch.tensor(stds,  device=args.device).unsqueeze(0)
        assert mu.shape[1] == F_out, (
            f"JSON amb {mu.shape[1]} features, "
            f"esperava {F_out}."
        )
        print(f"Normalització carregada de {args.norm_json}")

    else:
        print("iniciant get_target_stats…")
        mu, sigma = get_target_stats(tr_dl, args.device)
        print("get_target_stats acabat")
        mu, sigma = mu.to(args.device), sigma.to(args.device)

    # Inferim horizon a partir del primer .pt de seq_dir
    # Troba a quin chunk i posició està la primera seqüència
    chunk_idx, seq_idx = ds.indices[0]
    sample = torch.load(ds.chunk_files[chunk_idx], map_location="cpu")["sequences"][seq_idx]


    H = len(sample["y_seq"]) # nombre de passos futurs creats


    if args.model == "MeteoGraphPC_v1":
        print ("Creant MeteoGraphPC_v1...")
        model = MeteoGraphPC_v1(in_channels=F_in, hidden=args.hidden_dim, out_channels=F_out, horizon=H).to(args.device)
        print ("MeteoGraphPC_v1 creat")
    elif args.model == "MeteoGraphPC_v2":
        print ("Creant MeteoGraphPC_v2...")
        model = MeteoGraphPC_v2( in_channels=F_in, hidden=args.hidden_dim, out_channels=F_out, horizon=H).to(args.device)
        print ("MeteoGraphPC_v2 creat")
    elif args.model == "MeteoGraphPC_v3":
        print ("Creant MeteoGraphPC_v3...")
        model = MeteoGraphPC_v3(n_feat=F_in, n_edge_feat=15, n_targets=F_out, hidden_dim=args.hidden_dim, num_layers=3, tcn_layers=4, heads=4, dropout=0.1).to(args.device)
        print ("MeteoGraphPC_v3 creat")
    elif args.model == "MeteoGraphPC_v4":
        print ("Creant MeteoGraphPC_v4...")
        model = MeteoGraphPC_v4(n_feat=F_in, n_edge_feat=15, n_targets=F_out, hidden_dim=args.hidden_dim, num_gat_layers=2, t_transformer_layers=2, n_heads=4, dropout=0.1).to(args.device)
        print ("MeteoGraphPC_v4 creat")
    else:
        raise ValueError(f"Model desconegut: {args.model}")

    #if torch.cuda.device_count() > 1:
    #    model = GeoDataParallel(model)

    print(f"Model creat amb {sum(p.numel() for p in model.parameters() if p.requires_grad):,} paràmetres entrenables.")
    print("Optimitzador escollit: Adam")
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    scaler = GradScaler()

    if args.lr_scheduler == "plateau":
        print("Scheduler escollit: ReduceLROnPlateau")
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=args.patience, min_lr=1e-6)
    elif args.lr_scheduler == "onecycle":
        print("Scheduler escollit: OneCycleLR")
        scheduler = OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=len(tr_dl), epochs=args.epochs, pct_start=0.5, div_factor=50.0, final_div_factor=1000, anneal_strategy='cos')

    crit  = nn.MSELoss()
    stopper = EarlyStopper(args.patience, args.min_delta)

    # Preparar fitxer de log
    print("Preparant fitxer de log CSV...")
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

    print("Iniciant entrenament (pot trigar hores)...")

    for epoch in range(1, args.epochs+1):
        print(f"\n\n=== Època {epoch}/{args.epochs} ===")
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
            #print(torch.cuda.memory_summary(device=0, abbreviated=True))

    print("Entrenament acabat.")
    print("Carregant millor model guardat...")
    # 1) Carrega el millor model guardat
    checkpoint = torch.load(best_ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()
    print("Model carregat.")

    print("Iniciant test...")
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
    ys_true = np.concatenate(ys_true, axis=0)   # → [S, H, N, F]
    ys_pred = np.concatenate(ys_pred, axis=0)   # → [S, H, N, F]

    # Extreu dimensions
    S, H, N, F = ys_true.shape
    mu_np = mu.cpu().numpy()

    # ← Guardo prediccions i valors reals per a anàlisi posterior
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, "y_true_test.npy"), ys_true)
    np.save(os.path.join(args.save_dir, "y_pred_test.npy"), ys_pred)

    # 2) Concatena els baselines originals (un sol pas)  
    ys_persist = np.stack(ys_persist, axis=0)   # → [S, N, F]

    # 4) Replica cada baseline per a tots els H passos de l’horitzó  
    persist_np = np.stack([
        np.repeat(p[None, ...], H, axis=0)       # de [N, F] → [H, N, F]
        for p in ys_persist
    ], axis=0)                                  # → [S, H, N, F]

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

    print(f"Metriques del test ->       RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
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

    print(f"Entrenament i test acabats! Millor val_RMSE = {stopper.best:.4f}")
    print(f"Log complet al fitxer «{args.log_csv}».")

if __name__ == "__main__":
    main()
