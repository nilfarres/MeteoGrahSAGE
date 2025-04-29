#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
toData.py ― v2 (abril 2025)

Converteix els CSV preprocessats (prep_GPU_parallel.py) a Data objects
de torch_geometric amb:
  • Radi adaptatiu local (r_i = scale·d_k)        ➜ menys arestes innecessàries
  • Backbone Delaunay 2-D planari (lon·cos lat)   ➜ veïns naturals
  • Filtre d'altitud en metres reals              ➜ evita ponts entre vessants
  • Atributs de pendent/orientació a edge_attr
  • connectivitat limitada (≤120 km i ∆z ≤ max)
  • coalesce + remove_self_loops
  • Pes exponencial distància (opc.)
  • Metadades de grau i radi efectiu + sanity check

Autor original: Nil Farrés Soler · Modificat: 21-04-2025
"""

# --------------------------------------------------------------------------- #
# Imports                                                                     #
# --------------------------------------------------------------------------- #
import os, re, glob, math, calendar, argparse, logging, json
from datetime import datetime
from collections import defaultdict
from typing import Tuple

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import coalesce, remove_self_loops
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import networkx as nx
from scipy.spatial import Delaunay   # Backbone planari

# --------------------------------------------------------------------------- #
# Paràmetres per defecte                                                      #
# --------------------------------------------------------------------------- #
DEFAULT_INPUT_ROOT               = "D:/DADES_METEO_PC_PREPROCESSADES_GPU_PARALLEL"
DEFAULT_OUTPUT_ROOT              = "D:/DADES_METEO_PC_TO_DATA_v4"
DEFAULT_MAX_WORKERS              = 8

DEFAULT_K_NEIGHBORS              = 4     # k_base per al radi adaptatiu
DEFAULT_RADIUS_QUANTILE          = 0.08  # només per multiescala
DEFAULT_MULTISCALE_RADIUS_QUANT  = 0.65

DEFAULT_EDGE_DISTANCE_SCALE      = 100.0  # km
DEFAULT_EDGE_DECAY_LENGTH        = 75.0   # km
DEFAULT_PRESSURE_REF             = 1013.0 # hPa

DEFAULT_MAX_ALT_DIFF             = 0.15   # km ≅ 150 m
MAX_CONN_RADIUS_KM               = 80.0  # límit per ensure_connectivity_limited

REQUIRED_COLUMNS = ['id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt',
                    'VentDir', 'VentFor', 'Patm', 'lat', 'lon']

FEATURE_COLUMNS = ['Temp', 'Humitat', 'Pluja', 'VentFor', 'Patm', 'Alt_norm',
                   'VentDir_sin', 'VentDir_cos', 'hora_sin', 'hora_cos',
                   'dia_sin', 'dia_cos', 'cos_sza', 'DewPoint', 'PotentialTemp']

TEMPORAL_FEATURES = ['hora_sin', 'hora_cos', 'dia_sin', 'dia_cos', 'cos_sza']

YEARS_FOR_INTERPOLATION = list(range(2015, 2025))
PROCESSED_YEARS         = [y for y in YEARS_FOR_INTERPOLATION if y >= 2016]

GROUP_BY_PERIOD_CHOICES = ["none", "day", "month"]

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
os.makedirs("logs", exist_ok=True)
logfile = os.path.join("logs",
                       f"toData_v4_{datetime.now():%Y%m%d_%H%M%S}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(logfile, encoding="utf-8")]  # <─ afegeix encoding
)

np.seterr(divide="ignore")  # evita warnings /0

# --------------------------------------------------------------------------- #
# Utils bàsics                                                                #
# --------------------------------------------------------------------------- #
def extract_timestamp_from_filename(path: str) -> str:
    ts = datetime.strptime(os.path.basename(path)[:10], "%Y%m%d%H")
    return ts.strftime("%Y-%m-%d %H:%M:%S")

# -------------- Features de temps i sol ------------------------------------ #
def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if not np.issubdtype(df['Timestamp'].dtype, np.datetime64):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors="coerce")
    df['hora_sin'] = np.sin(2*np.pi*df['Timestamp'].dt.hour/24)
    df['hora_cos'] = np.cos(2*np.pi*df['Timestamp'].dt.hour/24)

    days_in_year = df['Timestamp'].dt.year.apply(
        lambda y: 366 if calendar.isleap(y) else 365)
    df['dia_sin'] = np.sin(2*np.pi*(df['Timestamp'].dt.dayofyear-1)/days_in_year)
    df['dia_cos'] = np.cos(2*np.pi*(df['Timestamp'].dt.dayofyear-1)/days_in_year)
    return df

def add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    if not np.issubdtype(df['Timestamp'].dtype, np.datetime64):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors="coerce")
    lat_rad = np.deg2rad(df['lat'])
    doy     = df['Timestamp'].dt.dayofyear
    dec_rad = np.deg2rad(23.44*np.sin(2*np.pi*(284+doy)/365))
    hour_loc = (df['Timestamp'].dt.hour + df['Timestamp'].dt.minute/60
                + df['lon']/15) % 24
    hra_rad = np.deg2rad((hour_loc-12)*15)
    df['cos_sza'] = (np.sin(lat_rad)*np.sin(dec_rad)
                     + np.cos(lat_rad)*np.cos(dec_rad)*np.cos(hra_rad))
    return df

# -------------- Variables derivades ---------------------------------------- #
def add_potential_temperature(df: pd.DataFrame, p0: float=DEFAULT_PRESSURE_REF):
    df['PotentialTemp'] = (df['Temp'] *
                           (p0/df['Patm'])**0.286)
    return df

def add_dew_point(df: pd.DataFrame):
    a, b      = 17.27, 237.7
    T_c       = df['Temp'] - 273.15
    alpha     = np.log(df['Humitat']/100) + (a*T_c)/(b+T_c)
    dew_c     = (b*alpha)/(a-alpha)
    df['DewPoint'] = dew_c + 273.15
    return df

def encode_wind_direction(df: pd.DataFrame, add_components: bool=False):
    df['VentDir'] = df['VentDir'].astype(float)
    df['VentDir_sin'] = np.sin(np.deg2rad(df['VentDir']))
    df['VentDir_cos'] = np.cos(np.deg2rad(df['VentDir']))
    if add_components:
        df['Vent_u'] = -df['VentFor']*df['VentDir_sin']
        df['Vent_v'] = -df['VentFor']*df['VentDir_cos']
    df.drop(columns=['VentDir'], inplace=True)
    return df

# --------------------------------------------------------------------------- #
# Normalització de features                                                  #
# --------------------------------------------------------------------------- #
def custom_normalize_features(x: torch.Tensor, names: list,
                              exclude: list, params: dict=None):
    x_norm = x.clone()
    if params is None:
        params = {}
        for i, n in enumerate(names):
            if n in exclude: continue
            mean, std = x[:,i].mean().item(), x[:,i].std().item() or 1
            x_norm[:,i] = (x[:,i]-mean)/std
            params[n]   = {"mean": mean, "std": std}
    else:
        for i, n in enumerate(names):
            if n in exclude: continue
            x_norm[:,i] = (x[:,i]-params[n]['mean'])/params[n]['std']
    return x_norm, params

# --------------------------------------------------------------------------- #
# Posicions                                                                   #
# --------------------------------------------------------------------------- #
def convert_to_cartesian(pos: torch.Tensor) -> torch.Tensor:
    R = 6371.0
    lat, lon, alt = pos[:,0], pos[:,1], pos[:,2]/1000
    lat_r, lon_r  = torch.deg2rad(lat), torch.deg2rad(lon)
    r_tot         = R + alt
    x = r_tot*torch.cos(lat_r)*torch.cos(lon_r)
    y = r_tot*torch.cos(lat_r)*torch.sin(lon_r)
    z = r_tot*torch.sin(lat_r)
    return torch.stack((x,y,z), dim=1)

def create_position_tensor(df: pd.DataFrame, metric: bool) -> torch.Tensor:
    pos = torch.tensor(df[['lat','lon','Alt']].values, dtype=torch.float)
    return convert_to_cartesian(pos) if metric else pos

# --------------------------------------------------------------------------- #
# Backbone Delaunay + Radi adaptatiu                                          #
# --------------------------------------------------------------------------- #
def _lon_scaled(lon: torch.Tensor, lat: torch.Tensor,
                lat0: float=41.7) -> torch.Tensor:
    return lon*math.cos(math.radians(lat0))

def build_delaunay_backbone(pos: torch.Tensor) -> torch.Tensor:
    lon_adj = _lon_scaled(pos[:,1], pos[:,0]).cpu().numpy()
    pts     = np.column_stack((lon_adj, pos[:,0].cpu().numpy()))
    tri     = Delaunay(pts)
    edges   = {tuple(sorted((a,b))) for simplex in tri.simplices
               for a,b in [(simplex[i], simplex[(i+1)%3]) for i in range(3)]}
    und     = torch.tensor(list(edges), dtype=torch.long).t()
    return torch.cat([und, und[[1,0]]], dim=1)  # dirigit

def adaptive_radius_edges(pos: torch.Tensor, k_base:int=DEFAULT_K_NEIGHBORS,
                          scale: float=1.1, metric: bool=False)->torch.Tensor:
    dmat = torch.cdist(pos, pos) if metric else torch.cdist(
        convert_to_cartesian(pos), convert_to_cartesian(pos))
    dmat.fill_diagonal_(float('inf'))
    kth,_ = torch.topk(dmat, k_base, largest=False)
    radii = kth[:,-1]*scale
    src,dst = [],[]
    for i, r in enumerate(radii):
        neigh = torch.nonzero(dmat[i]<=r).squeeze(1)
        src.extend([i]*len(neigh)); dst.extend(neigh.tolist())
    return torch.tensor([src,dst], dtype=torch.long)

# --------------------------------------------------------------------------- #
# Edge attributes helpers                                                     #
# --------------------------------------------------------------------------- #
def compute_haversine(pos_src, pos_dst):
    lat1, lon1 = torch.deg2rad(pos_src[:,0]), torch.deg2rad(pos_src[:,1])
    lat2, lon2 = torch.deg2rad(pos_dst[:,0]), torch.deg2rad(pos_dst[:,1])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = torch.sin(dlat/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2)**2
    return 6371.0*2*torch.asin(torch.sqrt(a+1e-8))  # km

def compute_bearing(lat1, lon1, lat2, lon2):
    lat1_r, lat2_r = torch.deg2rad(lat1), torch.deg2rad(lat2)
    dlon_r         = torch.deg2rad(lon2-lon1)
    x = torch.sin(dlon_r)*torch.cos(lat2_r)
    y = (torch.cos(lat1_r)*torch.sin(lat2_r)
         - torch.sin(lat1_r)*torch.cos(lat2_r)*torch.cos(dlon_r))
    return (torch.rad2deg(torch.atan2(x,y))+360)%360  # graus

def edge_attr_from_pairs(src, dst, pos, x, dist_scale, metric):
    # Distància
    dist = (torch.norm(pos[src]-pos[dst], dim=1)
            if metric else compute_haversine(pos[src], pos[dst]))
    dist_scaled = (dist/dist_scale).unsqueeze(1)

    # Diferències signades i absolutes
    diff = lambda col: (x[src,col]-x[dst,col]).unsqueeze(1)
    abs_alt = torch.abs(x[src,5]-x[dst,5]).unsqueeze(1)

    # Bearing & slope
    if metric:
        bearing_sin = bearing_cos = torch.zeros_like(diff(0))
    else:
        b = compute_bearing(pos[src,0], pos[src,1],
                            pos[dst,0], pos[dst,1])
        b_rad        = torch.deg2rad(b)
        bearing_sin  = torch.sin(b_rad).unsqueeze(1)
        bearing_cos  = torch.cos(b_rad).unsqueeze(1)

    horiz_km = dist
    slope    = torch.where(horiz_km>0, abs_alt.squeeze(1)/horiz_km,
                           torch.zeros_like(horiz_km)).unsqueeze(1)

    fwd = torch.cat([dist_scaled,
                     diff(0), diff(1), diff(2), diff(3), diff(4),
                     abs_alt, slope,
                     diff(6), diff(7), diff(13), diff(14),
                     bearing_sin, bearing_cos], dim=1)

    rev = torch.cat([dist_scaled,
                     -diff(0), -diff(1), -diff(2), -diff(3), -diff(4),
                     abs_alt, slope,
                     -diff(6), -diff(7), -diff(13), -diff(14),
                     -bearing_sin, -bearing_cos], dim=1)
    return fwd, rev

# --------------------------------------------------------------------------- #
# Connectivitat limitada                                                      #
# --------------------------------------------------------------------------- #
def ensure_connectivity_limited(edge_index, pos, max_alt_km, max_rad_km,
                                metric: bool):
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())
    n_nodes = pos.size(0)
    for n in range(n_nodes):
        if G.degree[n] > 0: continue
        dist = torch.norm(pos[n]-pos, dim=1) if metric else compute_haversine(
            pos[n].unsqueeze(0), pos).squeeze(0)
        alt  = torch.abs(pos[:,2]-pos[n,2])/1000
        mask = (dist<=max_rad_km)&(alt<=max_alt_km)&(dist>0)
        if mask.any():
            j = torch.argmin(dist.masked_fill(~mask, float('inf'))).item()
            edge_index = torch.cat([edge_index,
                                    torch.tensor([[n,j],[j,n]],dtype=torch.long)], dim=1)
    return edge_index

# --------------------------------------------------------------------------- #
# Edge builder principal                                                      #
# --------------------------------------------------------------------------- #
def create_edge_index_and_attr(pos, x, k_neighbors, radius_quantile,
                               dist_scale, add_multiscale, ms_quant,
                               max_alt_km, add_weight, decay_len,
                               metric):
    # 1) Backbone + radi adaptatiu
    edge_back = build_delaunay_backbone(pos)
    edge_rad  = adaptive_radius_edges(pos, k_neighbors, 1.3, metric)
    dir_edge  = torch.cat([edge_back, edge_rad], dim=1)

    # 2) Atributs
    src, dst  = dir_edge
    fwd, rev  = edge_attr_from_pairs(src, dst, pos, x, dist_scale, metric)
    edge_attr = torch.cat([fwd, rev], dim=0)
    edge_idx  = torch.cat([dir_edge, dir_edge[[1,0]]], dim=1)

    # 3) Multiescala opcional
    if add_multiscale:
        ms_edge = radius_graph(pos, r=torch.quantile(
            torch.cdist(pos,pos), ms_quant).item(), loop=False)
        ms_src, ms_dst = ms_edge
        m_fwd, m_rev   = edge_attr_from_pairs(ms_src, ms_dst, pos, x,
                                              dist_scale, metric)
        edge_idx  = torch.cat([edge_idx, ms_edge, ms_edge[[1,0]]], dim=1)
        edge_attr = torch.cat([edge_attr, m_fwd, m_rev], dim=0)

    # 4) Filtre ∆z
    alt_diff = torch.abs(pos[edge_idx[0],2]-pos[edge_idx[1],2])/1000
    keep     = alt_diff <= max_alt_km
    edge_idx, edge_attr = edge_idx[:,keep], edge_attr[keep]

    # 5) Connectivitat limitada
    edge_idx = ensure_connectivity_limited(edge_idx, pos,
                                           max_alt_km, MAX_CONN_RADIUS_KM,
                                           metric)

    # Re‑calcula attrs per possibles noves arestes
    added = edge_attr.size(0) != edge_idx.size(1)
    if added:
        new_src = edge_idx[0][-2:]   # les que s’han afegit (parell dirigit)
        new_dst = edge_idx[1][-2:]
        n_fwd, n_rev = edge_attr_from_pairs(new_src, new_dst, pos, x,
                                            dist_scale, metric)
        edge_attr = torch.cat([edge_attr, n_fwd, n_rev], dim=0)

    # 6) coalesce + remove_self_loops
    edge_idx, edge_attr = remove_self_loops(edge_idx, edge_attr)
    edge_idx, edge_attr = coalesce(edge_idx, edge_attr,
                                num_nodes=pos.size(0))


    # 7) Pes exponencial opcional
    if add_weight:
        w = torch.exp(-edge_attr[:,0]/decay_len).unsqueeze(1)
        edge_attr = torch.cat([edge_attr, w], dim=1)

    return edge_idx, edge_attr

# --------------------------------------------------------------------------- #
# Node features                                                               #
# --------------------------------------------------------------------------- #
def create_node_features(df: pd.DataFrame, excl_temp_norm: bool,
                         add_wind_comp: bool, p_ref: float,
                         log_pluja: bool,
                         norm_params: dict=None) -> Tuple[torch.Tensor, dict]:

    df.columns = df.columns.str.strip()
    if 'Timestamp' in df.columns:
        df = add_cyclical_time_features(df)
        df = add_solar_features(df)

    df = encode_wind_direction(df, add_wind_comp)

    # Kelvin, %➜[0,1], log1p pluja
    df['Temp']    += 273.15
    df['Humitat'] /= 100.0
    if log_pluja:
        df['Pluja'] = np.log1p(np.maximum(
            pd.to_numeric(df['Pluja'], errors='coerce').fillna(0), 0))
    df['Patm']   -= p_ref

    # DewPoint & PotentialTemp
    df = add_dew_point(df)
    df = add_potential_temperature(df, p_ref)

    # Alt_norm si cal
    if 'Alt_norm' not in df.columns:
        alt_mean, alt_std = 454.3, 175.61
        df['Alt_norm'] = (df['Alt']-alt_mean)/alt_std

    # Selecció de columnes & tensor
    x = torch.tensor(df[FEATURE_COLUMNS].values, dtype=torch.float)
    excl = TEMPORAL_FEATURES if excl_temp_norm else []
    x, params = custom_normalize_features(x, FEATURE_COLUMNS, excl, norm_params)
    return x, params

# --------------------------------------------------------------------------- #
# Sanity i metadades                                                          #
# --------------------------------------------------------------------------- #
def compute_graph_metadata(data: Data):
    n = data.x.size(0)
    e = data.edge_index.size(1)//2
    mean_deg = (e*2)/n if n else 0
    eff_rad  = data.edge_attr[:,0].mean().item() if data.edge_attr.size(0) else 0
    return {"mean_degree": mean_deg, "effective_radius": eff_rad}

def sanity_check_node(data: Data, idx: int=0, k: int=5):
    neigh = data.edge_index[1][data.edge_index[0]==idx]
    logging.info(f"Node {idx} té {neigh.numel()} veïns.")
    for n in neigh[:k]:
        d = data.edge_attr[(data.edge_index[0]==idx)&(data.edge_index[1]==n)][0,0]
        logging.info(f" -> {n.item()} dist={d*DEFAULT_EDGE_DISTANCE_SCALE:.1f} km")

# --------------------------------------------------------------------------- #
# Processament d’un fitxer                                                    #
# --------------------------------------------------------------------------- #
def process_file(csv_path: str, input_root: str, output_root: str,
                 k_neighbors: int, radius_q: float, dist_scale: float,
                 metric_pos: bool, excl_temp_norm: bool, gpu: str,
                 add_multiscale: bool, ms_q: float, max_alt_km: float,
                 add_weight: bool, decay_len: float, p_ref: float,
                 log_pluja: bool, add_wind_comp: bool,
                 include_year: bool, norm_params: dict=None):

    try:
        df = pd.read_csv(csv_path)
        df['VentFor'] = pd.to_numeric(df['VentFor'], errors='coerce').fillna(0)/3.6
        if 'Timestamp' not in df.columns:
            df.insert(0, 'Timestamp', extract_timestamp_from_filename(csv_path))
        if not set(REQUIRED_COLUMNS).issubset(df.columns):
            logging.error(f"{csv_path} incomplet.")
            return None

        device = torch.device(gpu)
        x, nparams = create_node_features(df, excl_temp_norm, add_wind_comp,
                                          p_ref, log_pluja, norm_params)
        pos = create_position_tensor(df, metric_pos).to(device)
        x   = x.to(device)

        if x.size(0) < 2:
            logging.warning(f"{csv_path} només té 1 node, s'omet.")
            return None

        edge_idx, edge_attr = create_edge_index_and_attr(
            pos, x, k_neighbors, radius_q, dist_scale,
            add_multiscale, ms_q, max_alt_km, add_weight,
            decay_len, metric_pos)

        data = Data(x=x.cpu(), pos=pos.cpu(),
                    edge_index=edge_idx.cpu(), edge_attr=edge_attr.cpu())
        data.ids        = list(df['id'])
        data.fonts      = list(df['Font'])
        data.timestamp  = df['Timestamp'].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
        if include_year: data.year = int(df['Timestamp'].dt.year.iloc[0])
        data.norm_params = nparams
        data.meta        = compute_graph_metadata(data)
        sanity_check_node(data, 0, 3)

        # Path de sortida
        rel = os.path.relpath(csv_path, input_root).replace("dadesPC_utc.csv","pt")
        out = os.path.join(output_root, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save(data, out)
        logging.info(f"{csv_path} -> {out}")
        return True
    except Exception as e:
        logging.error(f"{csv_path} ERR -> {e}")
        return False

# --------------------------------------------------------------------------- #
# Batch processing                                                            #
# --------------------------------------------------------------------------- #
def assign_gpu(idx:int, gpus:list)->str: return gpus[idx%len(gpus)]

def process_all_files(input_root, output_root, max_workers,
                      k_neighbors, radius_q, dist_scale, metric_pos,
                      excl_temp_norm, gpus, add_multiscale, ms_q,
                      max_alt_km, add_weight, decay_len, p_ref,
                      log_pluja, add_wind_comp, include_year,
                      group_by, make_seq, coverage, norm_params):

    files = [os.path.join(r,f) for r,_,fs in os.walk(input_root)
             for f in fs if f.endswith("dadesPC_utc.csv")]
    files = [f for f in files
             if int(re.match(r'(\d{4})', os.path.basename(f)).group(1))
             in PROCESSED_YEARS]
    logging.info(f"{len(files)} fitxers a processar")

    ok, ko = 0, 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(process_file, fp, input_root, output_root,
                         k_neighbors, radius_q, dist_scale, metric_pos,
                         excl_temp_norm, assign_gpu(i,gpus), add_multiscale,
                         ms_q, max_alt_km, add_weight, decay_len, p_ref,
                         log_pluja, add_wind_comp, include_year,
                         norm_params): fp
               for i,fp in enumerate(files)}
        for r in tqdm(as_completed(fut), total=len(fut), unit="fitxer"):
            ok += r.result() is True
            ko += r.result() is False
    logging.info(f"Fi: {ok} OK · {ko} KO")

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args():
    P = argparse.ArgumentParser("CSV -> Data MeteoGraphSAGE")
    P.add_argument("--input_root",  default=DEFAULT_INPUT_ROOT)
    P.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    P.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    P.add_argument("--k_neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    P.add_argument("--radius_quantile", type=float, default=DEFAULT_RADIUS_QUANTILE)
    P.add_argument("--edge_distance_scale", type=float, default=DEFAULT_EDGE_DISTANCE_SCALE)
    P.add_argument("--use_metric_pos", action="store_true")
    P.add_argument("--exclude_temporal_norm", action="store_true")
    P.add_argument("--gpu_devices", default="cuda:0")
    P.add_argument("--add_multiscale", action="store_true")
    P.add_argument("--multiscale_radius_quantile", type=float,
                   default=DEFAULT_MULTISCALE_RADIUS_QUANT)
    P.add_argument("--max_alt_diff", type=float, default=DEFAULT_MAX_ALT_DIFF)
    P.add_argument("--add_edge_weight", action="store_true")
    P.add_argument("--edge_decay_length", type=float, default=DEFAULT_EDGE_DECAY_LENGTH)
    P.add_argument("--pressure_ref", type=float, default=DEFAULT_PRESSURE_REF)
    P.add_argument("--log_transform_pluja", action="store_true")
    P.add_argument("--add_wind_components", action="store_true")
    P.add_argument("--include_year_feature", action="store_true")
    P.add_argument("--group_by_period", choices=GROUP_BY_PERIOD_CHOICES, default="none")
    P.add_argument("--generate_sequence", action="store_true")
    P.add_argument("--node_coverage_analysis", action="store_true")
    P.add_argument("--PC_norm_params", help="JSON amb paràmetres globals")
    return P.parse_args()

def main():
    A = parse_args()
    norm = json.load(open(A.PC_norm_params)) if A.PC_norm_params else None
    gpus = [g.strip() for g in A.gpu_devices.split(",") if g.strip()]
    process_all_files(A.input_root, A.output_root, A.max_workers,
                      A.k_neighbors, A.radius_quantile, A.edge_distance_scale,
                      A.use_metric_pos, A.exclude_temporal_norm, gpus,
                      A.add_multiscale, A.multiscale_radius_quantile,
                      A.max_alt_diff, A.add_edge_weight, A.edge_decay_length,
                      A.pressure_ref, A.log_transform_pluja,
                      A.add_wind_components, A.include_year_feature,
                      A.group_by_period, A.generate_sequence,
                      A.node_coverage_analysis, norm)

if __name__ == "__main__":
    main()
