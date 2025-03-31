#!/usr/bin/env python3
"""
toData_GPU_parallel_v4.py

Script per transformar fitxers CSV preprocessats en objectes Data de torch_geometric,
optimitzat per alimentar el model MeteoGraphSAGE.

Millores respecte a la versió toData_GPU_parallel_v3.py:
  - Normalització personalitzada que exclou les features temporals (hora/dia) per preservar el seu valor cíclic.
  - Possibilitat de convertir les coordenades (lat, lon, Alt) a coordenades mètriques (km) abans de construir el graf.
  - Escalament de la distància d’aresta per ajustar la seva magnitud.
  - Inclusió de la diferència de pressió (Patm) com a atribut d’aresta.
  - Possibilitat de definir el dispositiu GPU a utilitzar (per evitar saturació amb múltiples processos).
  - Paràmetres configurables via línia de comandes.

Autor: Nil Farrés Soler
"""

import os, glob, re, argparse, logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_undirected, remove_duplicate_edges
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuració per defecte
DEFAULT_INPUT_ROOT = "D:/DADES_METEO_PC_PREPROCESSADES_GPU_PARALLEL"
DEFAULT_OUTPUT_ROOT = "D:/DADES_METEO_PC_TO_DATA"
DEFAULT_MAX_WORKERS = 8
DEFAULT_K_NEIGHBORS = 5
DEFAULT_RADIUS_QUANTILE = 0.75
DEFAULT_EDGE_DISTANCE_SCALE = 100.0

# Columnes requerides i fonts fiables
REQUIRED_COLUMNS = ['id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt', 'VentDir', 'VentFor', 'Patm', 'lat', 'lon', 'Timestamp']
OFFICIAL_SOURCES = ["Aemet", "METEOCAT", "METEOCAT_WEB", "Meteoclimatic", "Vallsdaneu",
                    "SAIH", "avamet", "Meteoprades", "MeteoPirineus", "WLINK_DAVIS"]

# Llista de columnes de features finals
FEATURE_COLUMNS = ['Temp', 'Humitat', 'Pluja', 'VentFor', 'Patm', 'Alt', 
                   'VentDir_sin', 'VentDir_cos', 'hora_sin', 'hora_cos', 'dia_sin', 'dia_cos']
# Llista de features temporals que volem EXCLÒURE de la normalització
TEMPORAL_FEATURES = ['hora_sin', 'hora_cos', 'dia_sin', 'dia_cos']

# Anys per a interpolació i preprocessament
YEARS_FOR_INTERPOLATION = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
PROCESSED_YEARS = [year for year in YEARS_FOR_INTERPOLATION if year >= 2016]

# Configurar logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def custom_normalize_features(x: torch.Tensor, feature_names: list, exclude_names: list) -> torch.Tensor:
    """
    Normalitza les columnes de x corresponents a les features no incloses en exclude_names.
    Les columnes a excloure es deixen sense canvis.
    """
    x_norm = x.clone()
    for i, name in enumerate(feature_names):
        if name in exclude_names:
            continue
        col = x[:, i]
        mean = col.mean()
        std = col.std()
        if std == 0:
            std = 1
        x_norm[:, i] = (col - mean) / std
    return x_norm


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Afegeix a df les característiques temporals cícliques:
      - hora_sin, hora_cos basades en l'hora del Timestamp.
      - dia_sin, dia_cos basades en el dia de l'any.
    """
    if not np.issubdtype(df['Timestamp'].dtype, np.datetime64):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['hora_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.hour / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.hour / 24)
    df['dia_sin'] = np.sin(2 * np.pi * (df['Timestamp'].dt.dayofyear - 1) / 365)
    df['dia_cos'] = np.cos(2 * np.pi * (df['Timestamp'].dt.dayofyear - 1) / 365)
    return df


def encode_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma la columna 'VentDir' textual en dues columnes numèriques: 'VentDir_sin' i 'VentDir_cos',
    convertint la direcció en graus (seguint un mapping) a sin i cos.
    """
    mapping = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    df['VentDir'] = df['VentDir'].map(mapping)
    df['VentDir_sin'] = np.sin(np.deg2rad(df['VentDir']))
    df['VentDir_cos'] = np.cos(np.deg2rad(df['VentDir']))
    df.drop(columns=['VentDir'], inplace=True)
    return df


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    """
    Normalitza el tensor x perquè cada característica tingui mitjana 0 i desviació estàndard 1.
    """
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    std[std == 0] = 1
    return (x - mean) / std


def compute_geodesic_distance(pos_src: torch.Tensor, pos_dst: torch.Tensor) -> torch.Tensor:
    """
    Calcula la distància geodèsica 3D entre dos vectors de posicions:
      - pos[:,0]: latitud en graus
      - pos[:,1]: longitud en graus
      - pos[:,2]: altitud en metres
    Es retorna la distància en km, combinant la distància horitzontal (Haversine)
    i la diferència d'altitud (convertida a km).
    """
    lat1 = torch.deg2rad(pos_src[:, 0])
    lon1 = torch.deg2rad(pos_src[:, 1])
    lat2 = torch.deg2rad(pos_dst[:, 0])
    lon2 = torch.deg2rad(pos_dst[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.asin(torch.sqrt(a + 1e-8))
    R = 6371.0  # Radi de la Terra en km
    horizontal_distance = R * c
    alt1 = pos_src[:, 2] / 1000.0
    alt2 = pos_dst[:, 2] / 1000.0
    alt_diff = alt2 - alt1
    distance = torch.sqrt(horizontal_distance**2 + alt_diff**2)
    return distance.unsqueeze(1)


def convert_to_metric(pos: torch.Tensor) -> torch.Tensor:
    """
    Converteix les coordenades (lat, lon, Alt) a coordenades mètriques en km.
    Utilitza una projecció local amb latitud mitjana.
    """
    lat = pos[:, 0]
    lon = pos[:, 1]
    alt = pos[:, 2]  # en metres
    # Convertir altitud a km
    alt_km = alt / 1000.0
    # Calcular la latitud mitjana en radians per a la correcció de longitud
    lat_mean = torch.deg2rad(lat.mean())
    # Aproximadament, 1 grau de latitud ≈ 111 km; per longitud, 111 * cos(lat_mean)
    x = lon * 111.0 * torch.cos(lat_mean)
    y = lat * 111.0
    z = alt_km
    return torch.stack((x, y, z), dim=1)


def create_node_features(df: pd.DataFrame, exclude_temporal_norm: bool) -> torch.Tensor:
    """
    Crea el tensor de característiques dels nodes a partir del dataframe.
    Incorpora la codificació trigonomètrica per VentDir i les features temporals.
    Si exclude_temporal_norm és True, les columnes temporals no es normalitzen.
    """
    if 'Timestamp' in df.columns:
        df = add_cyclical_time_features(df)
    if 'VentDir' in df.columns:
        df = encode_wind_direction(df)
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        logging.error(f"Columns missing: {missing}")
        raise ValueError("Missing feature columns in dataframe.")
    x = torch.tensor(df[FEATURE_COLUMNS].values, dtype=torch.float)
    if exclude_temporal_norm:
        x = custom_normalize_features(x, FEATURE_COLUMNS, TEMPORAL_FEATURES)
    else:
        x = normalize_features(x)
    return x


def create_position_tensor(df: pd.DataFrame, use_metric: bool) -> torch.Tensor:
    """
    Converteix les columnes ['lat', 'lon', 'Alt'] en un tensor de posicions.
    Si use_metric és True, es converteixen a coordenades mètriques en km.
    """
    pos = torch.tensor(df[['lat', 'lon', 'Alt']].values, dtype=torch.float)
    if use_metric:
        pos = convert_to_metric(pos)
    return pos


def create_edge_index_and_attr(pos: torch.Tensor, x: torch.Tensor, k_neighbors: int, radius_quantile: float, edge_distance_scale: float) -> (torch.Tensor, torch.Tensor):
    """
    Crea l'edge_index i edge_attr del graf.
    Si num_nodes < 50, s'utilitza knn_graph; en cas contrari, s'utilitza radius_graph amb radi definit pel quantil.
    El graf es converteix a no dirigit.
    
    edge_attr inclou:
      1. Distància geodèsica 3D (km) escalada per edge_distance_scale.
      2. Diferència absoluta en Temp (índex 0 de x).
      3. Diferència absoluta en Humitat (índex 1 de x).
      4. Diferència absoluta en VentFor (índex 3 de x).
      5. Diferència absoluta en Patm (índex 4 de x).
    """
    num_nodes = pos.size(0)
    if num_nodes < 2:
        raise ValueError("No hi ha suficients nodes per construir el graf.")
    if num_nodes < 50:
        edge_index = knn_graph(pos, k=k_neighbors, loop=False)
    else:
        sample_size = min(50, num_nodes)
        indices = torch.randperm(num_nodes)[:sample_size]
        sample_pos = pos[indices]
        dists = torch.cdist(sample_pos, sample_pos)
        dists_flat = dists[dists > 0]
        radius = torch.quantile(dists_flat, radius_quantile).item()
        edge_index = radius_graph(pos, r=radius, loop=False)
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_duplicate_edges(edge_index)
    
    src, dst = edge_index
    edge_attr_dist = compute_geodesic_distance(pos[src], pos[dst])
    # Escalar la distància per edge_distance_scale
    edge_attr_dist = edge_attr_dist / edge_distance_scale
    diff_temp = torch.abs(x[src, 0] - x[dst, 0]).unsqueeze(1)
    diff_humitat = torch.abs(x[src, 1] - x[dst, 1]).unsqueeze(1)
    diff_ventFor = torch.abs(x[src, 3] - x[dst, 3]).unsqueeze(1)
    diff_patm = torch.abs(x[src, 4] - x[dst, 4]).unsqueeze(1)
    edge_attr = torch.cat([edge_attr_dist, diff_temp, diff_humitat, diff_ventFor, diff_patm], dim=1)
    return edge_index, edge_attr


def process_file(file_path: str, input_root: str, output_root: str, k_neighbors: int, radius_quantile: float, edge_distance_scale: float, use_metric: bool, exclude_temporal_norm: bool, gpu_device: str):
    """
    Processa un fitxer CSV per transformar-lo en un objecte Data.
    Inclou lectura, filtratge, creació de features, construcció del graf i desament.
    """
    try:
        df = pd.read_csv(file_path)
        df = df[REQUIRED_COLUMNS]
        df = df[df['Font'].isin(OFFICIAL_SOURCES)]
        if df.empty:
            logging.info(f"El fitxer {file_path} no conté dades oficials. S'omet.")
            return
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        device = torch.device(gpu_device)
        x = create_node_features(df, exclude_temporal_norm).to(device)
        pos = create_position_tensor(df, use_metric).to(device)
        num_nodes = x.size(0)
        if num_nodes < 2:
            logging.info(f"El fitxer {file_path} té menys d'una connexió possible (num_nodes={num_nodes}). S'omet.")
            return
        edge_index, edge_attr = create_edge_index_and_attr(pos, x, k_neighbors, radius_quantile, edge_distance_scale)
        
        # Portar resultats a CPU
        x = x.cpu()
        pos = pos.cpu()
        edge_index = edge_index.cpu()
        edge_attr = edge_attr.cpu()
        
        data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
        data.ids = list(df['id'])
        data.fonts = list(df['Font'])
        if 'Timestamp' in df.columns:
            data.timestamp = df['Timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
        
        rel_path = os.path.relpath(file_path, input_root)
        rel_path_pt = rel_path.replace("dadesPC_utc.csv", "pt")
        output_file = os.path.join(output_root, rel_path_pt)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save(data, output_file)
        logging.info(f"Conversió correcta: {file_path} -> {output_file}")
    except Exception as e:
        logging.error(f"Error en processar {file_path}: {e}")


def process_all_files(input_root: str, output_root: str, max_workers: int, k_neighbors: int, radius_quantile: float, edge_distance_scale: float, use_metric: bool, exclude_temporal_norm: bool, gpu_device: str):
    """
    Recorre tots els fitxers CSV dins d'input_root (seguint criteris d'any i nom) i processa cada fitxer en paral·lel.
    """
    files_to_process = []
    for root_dir, dirs, files in os.walk(input_root):
        print("Processant directori:", root_dir)
        dirs[:] = [d for d in dirs if not any(sub.lower() in d.lower() for sub in [
            "tauladades", "vextrems", "admin_estacions", "clima", "clima meteo",
            "error_var", "html", "png", "var_vextrems"]) and not re.search(r'\d+_old', d)]
        for file in files:
            if file.endswith("dadesPC_utc.csv"):
                year_match = re.match(r'(\d{4})', file)
                if year_match:
                    year_int = int(year_match.group(1))
                    if PROCESSED_YEARS and (year_int not in PROCESSED_YEARS):
                        continue
                    file_path = os.path.join(root_dir, file)
                    files_to_process.append(file_path)
    print(f"Total fitxers a processar: {len(files_to_process)}")
    if not files_to_process:
        print("No s'ha trobat cap fitxer que compleixi els criteris.")
        return
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, fp, input_root, output_root, k_neighbors, radius_quantile, edge_distance_scale, use_metric, exclude_temporal_norm, gpu_device): fp for fp in files_to_process}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processant fitxers", unit="fitxer"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processant {futures[future]}: {e}")
    logging.info("Processament finalitzat.")
    print("Processament finalitzat.")


def parse_args():
    parser = argparse.ArgumentParser(description="Converteix fitxers CSV preprocessats en objectes Data per entrenar MeteoGraphSAGE.")
    parser.add_argument("--input_root", type=str, default=DEFAULT_INPUT_ROOT, help="Directori d'entrada dels fitxers CSV preprocessats.")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Directori de sortida per als fitxers .pt.")
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS, help="Nombre màxim de treballadors per a paral·lelització.")
    parser.add_argument("--k_neighbors", type=int, default=DEFAULT_K_NEIGHBORS, help="Nombre de veïns per a knn_graph si num_nodes < 50.")
    parser.add_argument("--radius_quantile", type=float, default=DEFAULT_RADIUS_QUANTILE, help="Quantil per calcular el radi en radius_graph si num_nodes >= 50.")
    parser.add_argument("--edge_distance_scale", type=float, default=DEFAULT_EDGE_DISTANCE_SCALE, help="Factor per escalar la distància d'aresta (ex: dividir per 100 km).")
    parser.add_argument("--use_metric_pos", action="store_true", help="Si s'activa, converteix les coordenades (lat, lon, Alt) a coordenades mètriques en km per construir el graf.")
    parser.add_argument("--exclude_temporal_norm", action="store_true", help="Si s'activa, les features temporals (hora, dia) no es normalitzen per conservar el seu valor cíclic.")
    parser.add_argument("--gpu_device", type=str, default="cuda:0", help="Dispositiu GPU a utilitzar (per exemple, 'cuda:0').")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.info(f"Iniciant processament amb input_root={args.input_root} i output_root={args.output_root}")
    process_all_files(args.input_root, args.output_root, args.max_workers, args.k_neighbors,
                      args.radius_quantile, args.edge_distance_scale, args.use_metric_pos,
                      args.exclude_temporal_norm, args.gpu_device)

if __name__ == "__main__":
    main()
