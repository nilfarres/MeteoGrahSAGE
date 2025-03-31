#!/usr/bin/env python3
"""
toData_GPU_parallel_v3.py

Script per transformar fitxers CSV preprocessats en objectes Data de torch_geometric,
optimitzat per alimentar el model MeteoGraphSAGE basat en GraphSAGE.
 
Millores implementades respecte toData_GPU_parallel_v2.py:
  - Ús d'una única normalització (mitjançant la funció normalize_features).
  - Codificació trigonomètrica per a la direcció del vent (VentDir_sin, VentDir_cos).
  - Inclusió de característiques temporals cícliques (hora i dia) com a part dels features dels nodes.
  - Càlcul de la distància geodèsica 3D basada en la fòrmula de Haversine (latitud/longitud en km i altitud convertida a km).
  - Construcció de graf no dirigit (concatenant edge_index amb la seva inversa i eliminant duplicats).
  - Modularitat i configurabilitat a través d'arguments de línia de comandes.
  - Logging detallat per facilitar depuració i manteniment.

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

# Columnes requerides i fonts fiables
REQUIRED_COLUMNS = ['id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt', 'VentDir', 'VentFor', 'Patm', 'lat', 'lon', 'Timestamp']
OFFICIAL_SOURCES = ["Aemet", "METEOCAT", "METEOCAT_WEB", "Meteoclimatic", "Vallsdaneu",
                    "SAIH", "avamet", "Meteoprades", "MeteoPirineus", "WLINK_DAVIS"]

# Llista de columnes de features finals (afegint codificació per VentDir i features temporals)
FEATURE_COLUMNS = ['Temp', 'Humitat', 'Pluja', 'VentFor', 'Patm', 'Alt', 
                   'VentDir_sin', 'VentDir_cos', 'hora_sin', 'hora_cos', 'dia_sin', 'dia_cos']

# Anys per a interpolació i preprocessament
YEARS_FOR_INTERPOLATION = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
PROCESSED_YEARS = [year for year in YEARS_FOR_INTERPOLATION if year >= 2016]

# Configurar logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    df['dia_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.dayofyear / 365)
    df['dia_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.dayofyear / 365)
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
    std[std == 0] = 1  # Evita divisió per zero
    return (x - mean) / std

def compute_geodesic_distance(pos_src: torch.Tensor, pos_dst: torch.Tensor) -> torch.Tensor:
    """
    Calcula la distància geodèsica 3D entre dos vectors de posicions:
      - pos[:,0]: latitud en graus
      - pos[:,1]: longitud en graus
      - pos[:,2]: altitud en metres
    Es retorna la distància en kilòmetres, combinant la distància horitzontal (fòrmula de Haversine)
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
    # Convertir altitud de metres a km
    alt1 = pos_src[:, 2] / 1000.0
    alt2 = pos_dst[:, 2] / 1000.0
    alt_diff = alt2 - alt1
    distance = torch.sqrt(horizontal_distance**2 + alt_diff**2)
    return distance.unsqueeze(1)

def create_node_features(df: pd.DataFrame) -> torch.Tensor:
    """
    Crea el tensor de característiques dels nodes a partir del dataframe.
    Incorpora la codificació trigonomètrica per VentDir i les features temporals.
    """
    if 'Timestamp' in df.columns:
        df = add_cyclical_time_features(df)
    if 'VentDir' in df.columns:
        df = encode_wind_direction(df)
    # Comprovar que totes les columnes de FEATURE_COLUMNS estan presents
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        logging.error(f"Columns missing: {missing}")
        raise ValueError("Missing feature columns in dataframe.")
    x = torch.tensor(df[FEATURE_COLUMNS].values, dtype=torch.float)
    x = normalize_features(x)
    return x

def create_position_tensor(df: pd.DataFrame) -> torch.Tensor:
    """
    Converteix les columnes ['lat', 'lon', 'Alt'] en un tensor de posicions.
    """
    pos = torch.tensor(df[['lat', 'lon', 'Alt']].values, dtype=torch.float)
    return pos

def create_edge_index_and_attr(pos: torch.Tensor, x: torch.Tensor, k_neighbors: int, radius_quantile: float) -> (torch.Tensor, torch.Tensor):
    """
    Crea el 'edge_index' i 'edge_attr' del graf basant-se en la posició dels nodes i les seves features.
    Si num_nodes < 50 s'utilitza knn_graph, en cas contrari, s'aplica radius_graph amb radi definit pel quantil.
    Finalment, es fa que el graf sigui no dirigit.
    
    edge_attr conté:
      1. Distància geodèsica 3D entre nodes (km).
      2. Diferència absoluta en Temp (índex 0 de x).
      3. Diferència absoluta en Humitat (índex 1 de x).
      4. Diferència absoluta en VentFor (índex 3 de x).
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
    # Convertir el graf en no dirigit
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_duplicate_edges(edge_index)
    
    src, dst = edge_index
    edge_attr_dist = compute_geodesic_distance(pos[src], pos[dst])
    diff_temp = torch.abs(x[src, 0] - x[dst, 0]).unsqueeze(1)
    diff_humitat = torch.abs(x[src, 1] - x[dst, 1]).unsqueeze(1)
    diff_ventFor = torch.abs(x[src, 3] - x[dst, 3]).unsqueeze(1)
    edge_attr = torch.cat([edge_attr_dist, diff_temp, diff_humitat, diff_ventFor], dim=1)
    return edge_index, edge_attr

def process_file(file_path: str, input_root: str, output_root: str, k_neighbors: int, radius_quantile: float):
    """
    Processa un fitxer CSV per transformar-lo en un objecte Data de PyTorch Geometric.
    Aquest procés inclou:
      - Lectura del CSV.
      - Filtrat per fonts fiables.
      - Extracció del Timestamp i creació de features temporals.
      - Creació de les característiques dels nodes amb codificació trigonomètrica per VentDir.
      - Creació del tensor de posicions.
      - Construcció del graf amb edge_index i edge_attr.
      - Desament del Data object replicant l'estructura original.
    """
    try:
        df = pd.read_csv(file_path)
        df = df[REQUIRED_COLUMNS]
        df = df[df['Font'].isin(OFFICIAL_SOURCES)]
        if df.empty:
            logging.info(f"El fitxer {file_path} no conté dades oficials. S'omet.")
            return
        # Convertir Timestamp (extret del nom de fitxer prèviament) ja existeix en df.
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        # Crear features dels nodes i codificar VentDir
        x = create_node_features(df).to('cuda')
        pos = create_position_tensor(df).to('cuda')
        num_nodes = x.size(0)
        if num_nodes < 2:
            logging.info(f"El fitxer {file_path} té menys d'una connexió possible (num_nodes={num_nodes}). S'omet.")
            return
        edge_index, edge_attr = create_edge_index_and_attr(pos, x, k_neighbors, radius_quantile)
        
        # Traslladar tensors a CPU per guardar
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

def process_all_files(input_root: str, output_root: str, max_workers: int, k_neighbors: int, radius_quantile: float):
    """
    Recorre tots els fitxers CSV a processar dins d'input_root (filtrant per anys i nom de fitxer),
    i processa cada fitxer en paral·lel.
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
        futures = {executor.submit(process_file, fp, input_root, output_root, k_neighbors, radius_quantile): fp for fp in files_to_process}
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
    return parser.parse_args()

def main():
    args = parse_args()
    logging.info(f"Iniciant processament amb input_root={args.input_root} i output_root={args.output_root}")
    process_all_files(args.input_root, args.output_root, args.max_workers, args.k_neighbors, args.radius_quantile)

if __name__ == "__main__":
    main()
