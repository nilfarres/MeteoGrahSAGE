#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import csv, glob, math, os, random, argparse, json
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

from torch_geometric.nn import DataParallel as GeoDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from torch_geometric_temporal.nn.recurrent import TGCN
import torch.nn.functional as F

import torch.multiprocessing as tmp_mp
tmp_mp.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings("ignore",
                        category=FutureWarning,
                       message=".*weights_only=False.*")


# ───────────────────────────────── UTILITATS ──────────────────────────────────
def set_seed(s: int) -> None:
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate(batch):
    xs, eis, ys = [], [], []
    for x_seq, ei_seq, y in batch:
        xs.append(x_seq); eis.append(ei_seq); ys.append(y)
    return xs, eis, torch.stack(ys)

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
    # Descarta seqüències on no hi ha etiqueta real a l'últim pas
    if not d["y_mask_seq"][-1]:
        return False
    # Comprova nan/inf per si de cas
    y = d["y_seq"][-1]
    return not (torch.isnan(y).any() or torch.isinf(y).any())

# ───────────────────────────────── DATASET ────────────────────────────────────
class GraphSeqDataset(Dataset):
    def __init__(self, seq_dir: str, num_workers: int | None = None):
        all_pt = sorted(glob.glob(os.path.join(seq_dir, "*.pt")))
        if not all_pt:
            raise RuntimeError("No hi ha .pt al directori!")

        workers = num_workers or max(multiprocessing.cpu_count() - 1, 1)
        print(f"Validant {len(all_pt):,} fitxers amb {workers} processos…")

        self.files: list[str] = []
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
        y_mat = d["y_seq"][-1]
        y = y_mat.mean(dim=0)

        # Reparem edge_index_seq perquè sempre sigui 2×E
        clean_ei_seq = []
        for ei in d["edge_index_seq"]:
            # si PyTorch ha aplatat o només n’hi ha 1 fila, el normalitzem a 2×0
            if ei.dim() == 1 or (ei.dim() == 2 and ei.size(0) == 1):
                ei = torch.empty((2, 0), dtype=torch.long, device=ei.device)
            clean_ei_seq.append(ei)

        return d["x_seq"], clean_ei_seq, y
    
# ───────────────────────────────── MODELS ──────────────────────────────────────
class DynGCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        #self.conv1 = GATConv(in_channels, hidden, heads=4, concat=False)
        #self.bn1   = nn.BatchNorm1d(hidden)
        self.conv2 = GCNConv(hidden,      hidden)
        #self.conv2 = GATConv(hidden,      hidden, heads=4, concat=False)
        #self.bn2   = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.gru   = nn.GRU(hidden, hidden, batch_first=True)
        self.head  = nn.Linear(hidden, in_channels)

    def forward(self, x_seq, ei_seq):
        g_vecs = []
        for x, ei in zip(x_seq, ei_seq):
            h = self.conv1(x, ei)
            #h = self.bn1(h)
            h = torch.relu(h)
            h = self.conv2(h, ei)
            #h = self.bn2(h)
            h = torch.relu(h)
            h = self.dropout(h)
            g = global_mean_pool(
                    h,
                    batch=torch.zeros(h.size(0), dtype=torch.long, device=h.device)
                ).squeeze(0)              # [hidden]
            g_vecs.append(g)
        g_stack = torch.stack(g_vecs).unsqueeze(0)   # [1,T,H]
        _, h_n  = self.gru(g_stack)                  # [1,1,H]
        return self.head(h_n.squeeze(0).squeeze(0))  # [F]
    
    
class DynTGCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        # cèl·lula temporal TGCN que combina GCN + GRU
        self.tgcn_cell = TGCN(in_channels, hidden)
        # capçalera multi‐output: projecta hidden → in_channels
        self.head = nn.Linear(hidden, in_channels)

    def forward(self, x_seq, ei_seq):
        h = None
        # recórrer seqüència horària
        for x, ei in zip(x_seq, ei_seq):
            # x: [N_t, in_channels], ei: [2, E_t]
            h = self.tgcn_cell(x, ei, h)  # h: [N_t, hidden]
        # agreguem informació de tots els nodes en un vector [hidden]
        # (poden usar-se global_mean_pool o simplement mitjana)
        g = global_mean_pool(h, batch=torch.zeros(h.size(0), dtype=torch.long, device=h.device)).squeeze(0) # [hidden]
        # ara retorna un vector [in_channels] compatible amb y (mitjana de nodes)
        return self.head(g)


class DynTCN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden: int = 128,
                 kernel_size: int = 3,
                 dilations: list[int] = [1, 2, 4]):
        super().__init__()
        # Convolucions espacials GCN
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        # Capes TCN amb diverses dilatacions
        self.tcn_layers = nn.ModuleList([
            nn.Conv1d(in_channels=hidden,
                      out_channels=hidden,
                      kernel_size=kernel_size,
                      dilation=d,
                      padding=(kernel_size - 1) * d)
            for d in dilations
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in dilations])
        # Capçalera per tornar a in_channels
        self.head = nn.Linear(hidden, in_channels)

    def forward(self, x_seq, ei_seq):
        # 1) Extracció de vectors g_t per cada hora
        g_vecs = []
        for x, ei in zip(x_seq, ei_seq):
            h = F.relu(self.conv1(x, ei))
            h = F.relu(self.conv2(h, ei))
            # Pooling global per obtenir vector [hidden] de cada graf
            g = global_mean_pool(
                h,
                batch=torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            ).squeeze(0)
            g_vecs.append(g)
        # 2) Crear tensor seq [1, hidden, T]
        seq = torch.stack(g_vecs).transpose(0, 1).unsqueeze(0)
        # 3) Passar per les capes TCN dilatades
        for conv, norm in zip(self.tcn_layers, self.norms):
            seq = F.relu(norm(conv(seq)))
        # 4) Agafar l'últim pas temporal i projectar
        t_last = seq[..., -1].squeeze(0)              # [hidden]
        return self.head(t_last)                     # [in_channels]

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
    for _, _, y_b in loader:
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

def run(loader, model, crit, dev, opt, mu, sigma, desc):
    train = opt is not None
    model.train() if train else model.eval()
    sum_loss = 0.0
    ys_true, ys_pred = [], []

    for xs_b, eis_b, y_b in tqdm(loader, desc=desc, leave=False):
        y_b = y_b.to(dev)
        # normalitzem objectiu
        y_norm = (y_b - mu) / sigma
        preds_norm = [
            model([x.to(dev) for x in xs], [ei.to(dev) for ei in eis])
            for xs, eis in zip(xs_b, eis_b)
        ]
        preds_norm = torch.stack(preds_norm)               # [B,F]
        loss = crit(preds_norm, y_norm)

        if train:
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        # mètriques en escala original
        preds = preds_norm * sigma + mu
        sum_loss += loss.item()
        # Emmagatzemar per càlcul global de R2 i MAPE
        ys_true.append(y_b.detach().cpu().numpy())
        ys_pred.append(preds.detach().cpu().numpy())

    n = len(loader)
    # Concatena totes les mostres
    y_true = np.concatenate(ys_true, axis=0)
    y_pred = np.concatenate(ys_pred, axis=0)
    # Càlcul global de mètriques
    rmse_global = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_global  = mean_absolute_error(y_true, y_pred)
    r2_global   = r2_score(y_true, y_pred)
    mape_global = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
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

# ───────────────────────────────── MAIN ───────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Entrena el model DynGCN sobre seqüències generades per generate_seq_v6.py")
    parser.add_argument("--seq_dir", type=str, default="D:/DADES_METEO_PC_generated_seqs_ws48_str6_hh6", help="Directori amb les seqüències .pt generades")
    parser.add_argument("--batch_size", type=int, default=8, help="Mida del batch per al DataLoader")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre màxim d'èpoques")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate per l'optimitzador")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Dimensió oculta del model")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Clip de gradient")
    parser.add_argument("--patience", type=int, default=15, help="Patience per early stopping")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Millora mínima per resetar patience")
    parser.add_argument("--device", type=str, default="cuda", help="Device per PyTorch ('cuda' o 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Semilla per a la reproducibilitat")
    parser.add_argument("--std_eps", type=float, default=1e-6, help="Petita constant per evitar divisions per zero en la normalització")
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directori on guardar el model entrenat')
    parser.add_argument('--log_csv', type=str, default=None, help='Fitxer CSV per desar el registre d\'entrenament')
    parser.add_argument("--model", choices=["dyngcn", "dyntgcn", "dyntcn"], default="dyngcn", help="Arquitectura: DynGCN (baseline), DynTGCN (GCN+GRU) o DynTCN (GCN+TCN dilatat)")
    parser.add_argument('--dl_num_workers', type=int, default=4, help="Nombre de processos/threads per al DataLoader")

    return parser.parse_args()

def main():
    args = parse_args()

    # Genera timestamp i crea nom de log si no s'ha passat amb --log_csv
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.log_csv:
        args.log_csv = f"train_log_{timestamp}.csv"

    # Assegura’t que el directori existeixi
    os.makedirs(args.save_dir, exist_ok=True)

    global GRAD_CLIP
    GRAD_CLIP = args.grad_clip

    # petita constant per evitar divisions per zero en la normalització
    global STD_EPS
    STD_EPS = args.std_eps

    set_seed(args.seed)

    ds = GraphSeqDataset(args.seq_dir, num_workers=args.dl_num_workers)
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

    F_in = ds[0][0][0].size(1)
    if args.model == "dyngcn":
        model = DynGCN(F_in, hidden=args.hidden_dim).to(args.device)
    elif args.model == "dyntgcn":
        model = DynTGCN(F_in, hidden=args.hidden_dim).to(args.device)
    elif args.model == "dyntcn":
        model = DynTCN(F_in, hidden=args.hidden_dim).to(args.device)
    else:
        raise ValueError(f"Model desconegut: {args.model}")

    #if torch.cuda.device_count() > 1:
    #    model = GeoDataParallel(model)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    #scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    scheduler = OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=len(tr_dl), epochs=args.epochs, pct_start=0.3, div_factor=25.0, final_div_factor=1e4)
    crit  = nn.MSELoss()
    stopper = EarlyStopper(args.patience, args.min_delta)
    ckpt_name = f"dyn_gcn_{datetime.now():%Y%m%d_%H%M%S}.pth"

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
            tr_loss, tr_rmse, tr_mae, tr_r2, tr_mape = run(tr_dl, model, crit, args.device, opt, mu, sigma, desc=f"[{epoch}/{args.epochs}] Train")
            vl_loss, vl_rmse, vl_mae, vl_r2, vl_mape = run(vl_dl, model, crit, args.device, None, mu, sigma, desc=f"[{epoch}/{args.epochs}] Val ")
            
            # Ajusta lr si no hi ha millora en val_loss
            #scheduler.step(vl_loss)
            scheduler.step()
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
                save_path = os.path.join(args.save_dir, ckpt_name)
                torch.save(model.state_dict(), save_path)
                print(f"Model desat a: {save_path}")

                print(f"  ↳ Millor val_RMSE! Model desat a {ckpt_name}")

            if stopper.step(vl_rmse):
                print(f"Early stopping (sense millora {args.patience} èpoques).")
                break
            
            # Esborrem la memòria GPU per evitar errors de memòria
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 1) Carrega el millor model
    best_path = os.path.join(args.save_dir, ckpt_name)
    state_dict = torch.load(best_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()


    # 2) Avaluació completa
    ys_true, ys_pred = [], []
    ys_persist, ys_climat = [], []

    with torch.no_grad():
        for xs_b, eis_b, y_b in te_dl:
            y_b = y_b.to(args.device)
            preds = [
                model([x.to(args.device) for x in xs],
                    [ei.to(args.device) for ei in eis])
                for xs, eis in zip(xs_b, eis_b)
            ]
            preds = torch.stack(preds) * sigma + mu
            ys_true.append(y_b.cpu().numpy())
            ys_pred.append(preds.cpu().numpy())

            # -- Baseline persistència i climatologia per a cada mostra del batch --
            for seq_x, y_true in zip(xs_b, y_b.cpu()):
                # Persistència: prenem l’últim snapshot i en fem la mitjana de nodes
                persist_pred = seq_x[-1].mean(dim=0).cpu().numpy()
                ys_persist.append(persist_pred)
                # Climatologia: sempre predictmu = mu (mitjana entrenament)
                ys_climat.append(mu.cpu().numpy())


    ys_true = np.concatenate(ys_true)
    ys_pred = np.concatenate(ys_pred)

    ys_persist = np.stack(ys_persist)
    ys_climat  = np.stack(ys_climat)

    # Importa d'entrada al fitxer:
    # from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse_persist = np.sqrt(mean_squared_error(ys_true, ys_persist))
    mae_persist  = mean_absolute_error(ys_true, ys_persist)
    r2_persist   = r2_score(ys_true, ys_persist)
    mape_persist = np.mean(np.abs((ys_true - ys_persist) / ys_true)) * 100

    rmse_clima  = np.sqrt(mean_squared_error(ys_true, ys_climat))
    mae_clima   = mean_absolute_error(ys_true, ys_climat)
    r2_clima    = r2_score(ys_true, ys_climat)
    mape_clima  = np.mean(np.abs((ys_true - ys_climat) / ys_true)) * 100

    rmse = np.sqrt(mean_squared_error(ys_true, ys_pred))
    mae  = mean_absolute_error   (ys_true, ys_pred)
    r2   = r2_score              (ys_true, ys_pred)
    mape = np.mean(np.abs((ys_true - ys_pred) / ys_true)) * 100

    with open(args.log_csv, "a", newline="") as f:
        w = csv.writer(f)
        # posem epoch="" perquè no és una època sinó el test final
        w.writerow(["", "test", "", rmse, mae, r2, mape])

    print(f"Test metrics → RMSE: {rmse:.4f}, MAE: {mae:.4f}, "
        f"R²: {r2:.4f}, MAPE: {mape:.2f}%")
    
    print(f"Baseline persistència → RMSE: {rmse_persist:.4f}, MAE: {mae_persist:.4f}, "
      f"R²: {r2_persist:.4f}, MAPE: {mape_persist:.2f}%")
    print(f"Baseline climatologia → RMSE: {rmse_clima:.4f}, MAE: {mae_clima:.4f}, "
        f"R²: {r2_clima:.4f}, MAPE: {mape_clima:.2f}%")

    with open(args.log_csv, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "persistència", "", rmse_persist, mae_persist, r2_persist, mape_persist])
        w.writerow(["", "climatologia",   "", rmse_clima,   mae_clima,   r2_clima,   mape_clima])


    print(f"Entrenament acabat. Millor val_RMSE = {stopper.best:.4f}")
    print(f"Log complet al fitxer «{args.log_csv}».")

if __name__ == "__main__":
    main()
