#!/usr/bin/env python3
"""
toData_GPU_parallel.py

Script per convertir fitxers CSV preprocessats (per prep_GPU_parallel.py)
en objectes Data de torch_geometric per alimentar el model MeteoGraphSAGE (GraphSAGE).

Exemples d'execució:
  python toData_GPU_parallel.py --input_root "/path/to/input" --output_root "/path/to/output" \
    --gpu_devices "cuda:0,cuda:1" --max_workers 4 --group_by_period "day" --node_coverage_analysis

Autor: Nil Farrés Soler
"""

import os, glob, re, argparse, logging, math
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import remove_duplicate_edges
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import networkx as nx
import calendar
import json
from typing import Tuple

# Configuració per defecte
DEFAULT_INPUT_ROOT = "D:/DADES_METEO_PC_PREPROCESSADES_GPU_PARALLEL"
DEFAULT_OUTPUT_ROOT = "D:/DADES_METEO_PC_TO_DATA"
DEFAULT_MAX_WORKERS = 8

DEFAULT_K_NEIGHBORS = 5

DEFAULT_RADIUS_QUANTILE = 0.15
DEFAULT_MULTISCALE_RADIUS_QUANTILE = 0.65

DEFAULT_EDGE_DISTANCE_SCALE = 100.0
DEFAULT_EDGE_DECAY_LENGTH = 50.0
DEFAULT_PRESSURE_REF = 1013.0

DEFAULT_MAX_ALT_DIFF = 0.8  # 800 m convertit a km

# Columnes requerides i fonts fiables
REQUIRED_COLUMNS = ['id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt', 'VentDir', 'VentFor', 'Patm', 'lat', 'lon', 'Timestamp']
OFFICIAL_SOURCES = ["Aemet", "METEOCAT", "METEOCAT_WEB", "Meteoclimatic", "Vallsdaneu",
                    "SAIH", "avamet", "Meteoprades", "MeteoPirineus", "WLINK_DAVIS"]

# Features finals dels nodes
FEATURE_COLUMNS = ['Temp', 'Humitat', 'Pluja', 'VentFor', 'Patm', 'Alt_norm',
                   'VentDir_sin', 'VentDir_cos', 'hora_sin', 'hora_cos', 'dia_sin', 
                   'dia_cos', 'cos_sza', 'DewPoint', 'PotentialTemp']
TEMPORAL_FEATURES = ['hora_sin', 'hora_cos', 'dia_sin', 'dia_cos', 'cos_sza']

YEARS_FOR_INTERPOLATION = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
PROCESSED_YEARS = [year for year in YEARS_FOR_INTERPOLATION if year >= 2016]

# Opcions per agrupar resultats i anàlisi de cobertura de nodes
GROUP_BY_PERIOD_CHOICES = ["none", "day", "month"]

os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if not np.issubdtype(df['Timestamp'].dtype, np.datetime64):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['hora_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.hour / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.hour / 24)
    # Calcular el nombre de dies de l'any per cada timestamp
    df['days_in_year'] = df['Timestamp'].dt.year.apply(lambda y: 366 if calendar.isleap(y) else 365)
    df['dia_sin'] = np.sin(2 * np.pi * (df['Timestamp'].dt.dayofyear - 1) / df['days_in_year'])
    df['dia_cos'] = np.cos(2 * np.pi * (df['Timestamp'].dt.dayofyear - 1) / df['days_in_year'])
    df.drop(columns=['days_in_year'], inplace=True)
    return df

def add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    if not np.issubdtype(df['Timestamp'].dtype, np.datetime64):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    # Convertir latitud a radians
    lat_rad = np.deg2rad(df['lat'])
    
    # Obtenir el dia de l'any per cada fila
    day_of_year = df['Timestamp'].dt.dayofyear

    # Calcular la declinació solar en graus i després convertir-la a radians
    dec_deg = 23.44 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
    dec_rad = np.deg2rad(dec_deg)
    
    # Calcular l'angle hora (HRA)
    # Aquí, assumim que el Timestamp és en UTC, per la qual cosa ajustem afegint lon/15 per obtenir l'hora solar local.
    # A més, s'aplica el mòdul 24 per mantenir l'hora dins del rang [0,24).
    hour_local = (df['Timestamp'].dt.hour + df['Timestamp'].dt.minute / 60.0 + df['lon'] / 15.0) % 24
    # HRA en graus: cada hora (fora del migdia) representa 15°
    HRA_deg = (hour_local - 12) * 15
    HRA_rad = np.deg2rad(HRA_deg)
    
    # Calcular cos(SZA)
    df['cos_sza'] = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(HRA_rad)
    
    return df

def add_potential_temperature(df: pd.DataFrame, pressure_ref: float) -> pd.DataFrame:
    """
    Calcula la temperatura potencial en Kelvin utilitzant Temp ja convertida a Kelvin.
    La fórmula és: θ = T * (P₀ / P)^(R/cₚ), amb R/cₚ ≈ 0.286.
    """
    # Suposem que df['Patm'] encara conté la pressió mesurada (sense la referència restada)
    # i Temp ja és en Kelvin
    df['PotentialTemp'] = df['Temp'] * (pressure_ref / df['Patm'])**0.286
    return df

def add_dew_point(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el punt de rosada a partir de la temperatura (Temp) i la humitat relativa (Humitat).
    S'assumeix que Temp està en Kelvin i Humitat en %.
    Utilitza la fórmula de Magnus per obtenir el punt de rosada en °C i, a continuació, el converteix a Kelvin.
    """
    a = 17.27
    b = 237.7
    # Converteix Temp de Kelvin a Celsius
    T_c = df['Temp'] - 273.15
    # Calcula alpha segons la fórmula de Magnus
    alpha = np.log(df['Humitat'] / 100.0) + (a * T_c) / (b + T_c)
    # Calcula el punt de rosada en Celsius
    dewpoint_c = (b * alpha) / (a - alpha)
    # Converteix el punt de rosada a Kelvin
    df['DewPoint'] = dewpoint_c + 273.15
    return df

def encode_wind_direction(df: pd.DataFrame, add_components: bool=False) -> pd.DataFrame:
    """
    Converteix 'VentDir' a 'VentDir_sin' i 'VentDir_cos'.
    Si VentFor és 0, s'estableixen VentDir_sin i VentDir_cos a 0 per indicar direcció indefinida.
    Opcionalment, afegeix components zonal/meridional.
    """
    if df['VentDir'].dtype == object:
        mapping = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
            'Calma': 0
        }
        df['VentDir'] = df['VentDir'].map(mapping)
    df['VentDir_sin'] = np.sin(np.deg2rad(df['VentDir']))
    df['VentDir_cos'] = np.cos(np.deg2rad(df['VentDir']))
    
    # Si la velocitat del vent és zero, estableix les components de direcció a 0
    df.loc[df['VentFor'] == 0, 'VentDir_sin'] = 0.0
    df.loc[df['VentFor'] == 0, 'VentDir_cos'] = 0.0
    
    if add_components:
        df['Vent_u'] = df['VentFor'] * df['VentDir_cos']
        df['Vent_v'] = df['VentFor'] * df['VentDir_sin']
    # Eliminar la columna original
    df.drop(columns=['VentDir'], inplace=True)
    return df


def custom_normalize_features(x: torch.Tensor, feature_names: list, exclude_names: list, norm_params: dict = None):
    """
    Normalitza les columnes de x que no estan a exclude_names.
    Si 'norm_params' està definit, utilitza els valors proporcionats;
    en cas contrari, calcula la mitjana i desviació a partir de 'x'.
    Retorna el tensor normalitzat i el diccionari de paràmetres.
    """
    x_norm = x.clone()
    computed_params = {}

    if norm_params is None:
        # Calcular per cada fitxer (no recomanat per a training final)
        for i, name in enumerate(feature_names):
            if name in exclude_names:
                continue
            col = x[:, i]
            mean = col.mean().item()
            std = col.std().item()
            if std == 0:
                std = 1
            x_norm[:, i] = (col - mean) / std
            computed_params[name] = {'mean': mean, 'std': std}
        return x_norm, computed_params
    else:
        # Utilitzar els paràmetres globals precomputats
        for i, name in enumerate(feature_names):
            if name in exclude_names:
                continue
            mean = norm_params[name]['mean']
            std = norm_params[name]['std']
            x_norm[:, i] = (x[:, i] - mean) / std
        return x_norm, norm_params



def create_node_features(df: pd.DataFrame, exclude_temporal_norm: bool, add_wind_components: bool,
                         pressure_ref: float, log_transform_pluja: bool, add_station_id_feature: bool, 
                         precomputed_norm_params: dict = None) -> Tuple[torch.Tensor, dict]:
    """
    Crea el tensor de features dels nodes.
    
    - Afegeix features temporals (hora_sin, hora_cos, dia_sin, dia_cos, cos_sza).
    - Codifica la direcció del vent en components trigonomètriques i, opcionalment, afegeix les components zonal/meridional.
    - Converteix la temperatura (Temp) de °C a Kelvin.
    - Calcula indicadors derivats: el punt de rosada (DewPoint) i la temperatura potencial (PotentialTemp).
    - Aplica la transformació logarítmica a 'Pluja' si s'indica, i normalitza la pressió restant-hi pressure_ref.
    - Opcionalment, afegeix una característica basada en l'ID d'estació.
    
    Retorna el tensor de features i els paràmetres de normalització.
    """

    if 'Timestamp' in df.columns:
        df = add_cyclical_time_features(df)
        df = add_solar_features(df)

    df = encode_wind_direction(df, add_wind_components)

    # Converteix la temperatura de ºC a Kelvin
    df['Temp'] = df['Temp'] + 273.15

    # Afegeix el punt de rosada (DewPoint) i la temperatura potencial (PotentialTemp)
    # (La funció add_dew_point assumeix que Temp està en Kelvin i retorna el valor en Kelvin)
    df = add_dew_point(df)
    # La funció add_potential_temperature assumeix que Temp ja està en Kelvin
    df = add_potential_temperature(df, pressure_ref)

    if log_transform_pluja:
        df['Pluja'] = np.log1p(df['Pluja'])
        
    df['Patm'] = df['Patm'] - pressure_ref

    # Si s'ha activat add_wind_components, afegim Vent_u i Vent_v a FEATURE_COLUMNS
    if add_wind_components:
        if 'Vent_u' not in df.columns:
            # En principi, ja s'hauria calculat dins de encode_wind_direction,
            # però en cas contrari, es pot calcular així:
            df['Vent_u'] = df['VentFor'] * df['VentDir_cos']
        if 'Vent_v' not in df.columns:
            df['Vent_v'] = df['VentFor'] * df['VentDir_sin']
        # Assegurar-se que FEATURE_COLUMNS inclou aquestes noves columnes
        if 'Vent_u' not in FEATURE_COLUMNS:
            FEATURE_COLUMNS.append('Vent_u')
        if 'Vent_v' not in FEATURE_COLUMNS:
            FEATURE_COLUMNS.append('Vent_v')

    # Assegurem que FEATURE_COLUMNS inclou les noves variables derivades
    if 'DewPoint' not in FEATURE_COLUMNS:
        FEATURE_COLUMNS.append('DewPoint')
    if 'PotentialTemp' not in FEATURE_COLUMNS:
        FEATURE_COLUMNS.append('PotentialTemp')

    if 'Alt_norm' not in df.columns:
        # Definim uns valors de referència per l'altitud:
        # Mitjana
        # Altitud mitjana de Catalunya: 700m
        # Altitud mitjana País Valencià: 363m
        # Altitud mitjana Illes Balears: 300m
        # 700 + 363 + 300 = 1363 / 3 = 454.3m -> Aquesta és la mitjana d'altitud dels Països Catalans que utilitzarem
        # Desviació estàndard
        # Primer calculem la diferència de cada valor respecte a la mitjana i elevem al quadrat.
        # Catalunya: 700 - 454 = 246 -> 246^2 = 60516
        # País Valencià: 363 - 454 = -91 -> (-91)^2 = 8281
        # Illes Balears: 300 - 454 = -154 -> (-154)^2 = 23716
        # Suma dels quadrats: 60516 + 8281 + 23716 = 92513
        # La variància és la suma dels quadrats dividida pel nombre de valors (3): 92513 / 3 = 30837.66667
        # La desviació estàndard és l'arrel quadrada de la variància: sqrt(30837.66667) = 175.61 -> Aquesta és la desviació estàndard que utilitzarem
        alt_mean = 454.3
        alt_std = 175.61
        df['Alt_norm'] = (df['Alt'] - alt_mean) / alt_std
    
    if add_station_id_feature:
        # Crear una característica numèrica basada en l'ID; per exemple, mapear l'ID a un enter únic
        # Aquí simplement assignem un index segons l'ordre d'aparició
        df['StationID'] = pd.factorize(df['id'])[0].astype(np.float32)
        # Afegir aquesta nova columna a FEATURE_COLUMNS si no hi és
        if 'StationID' not in FEATURE_COLUMNS:
            FEATURE_COLUMNS.append('StationID')

    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        logging.error(f"Columns missing: {missing}")
        raise ValueError("Missing feature columns in dataframe.")
    
    x = torch.tensor(df[FEATURE_COLUMNS].values, dtype=torch.float)

    if exclude_temporal_norm:
        x_norm, norm_params = custom_normalize_features(x, FEATURE_COLUMNS, TEMPORAL_FEATURES, precomputed_norm_params)
    else:
        x_norm, norm_params = custom_normalize_features(x, FEATURE_COLUMNS, [], precomputed_norm_params)
    return x_norm, norm_params


def convert_to_cartesian(pos: torch.Tensor) -> torch.Tensor:
    """
    Converteix (lat, lon, Alt) a coordenades cartesianes 3D en km respecte al centre de la Terra.
    Fórmula:
      x = (R + alt_km) * cos(lat_rad) * cos(lon_rad)
      y = (R + alt_km) * cos(lat_rad) * sin(lon_rad)
      z = (R + alt_km) * sin(lat_rad)
    On R ≈ 6371 km.
    """
    R = 6371.0
    lat = pos[:, 0]
    lon = pos[:, 1]
    alt = pos[:, 2] / 1000.0
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)
    r_total = R + alt
    x = r_total * torch.cos(lat_rad) * torch.cos(lon_rad)
    y = r_total * torch.cos(lat_rad) * torch.sin(lon_rad)
    z = r_total * torch.sin(lat_rad)
    return torch.stack((x, y, z), dim=1)


def create_position_tensor(df: pd.DataFrame, use_metric: bool) -> torch.Tensor:
    """
    Converteix ['lat', 'lon', 'Alt'] en un tensor de posicions.
    Si use_metric és True, les converteix a coordenades cartesianes 3D en km.
    """
    pos = torch.tensor(df[['lat', 'lon', 'Alt']].values, dtype=torch.float)
    if use_metric:
        pos = convert_to_cartesian(pos)
    return pos


def compute_geodesic_distance(pos_src: torch.Tensor, pos_dst: torch.Tensor) -> torch.Tensor:
    """
    Calcula la distància geodèsica 3D entre dos vectors de posicions utilitzant Haversine
    i la diferència d’altitud, retornant la distància en km.
    """
    lat1 = torch.deg2rad(pos_src[:, 0])
    lon1 = torch.deg2rad(pos_src[:, 1])
    lat2 = torch.deg2rad(pos_dst[:, 0])
    lon2 = torch.deg2rad(pos_dst[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.asin(torch.sqrt(a + 1e-8))
    R = 6371.0
    horizontal_distance = R * c
    alt1 = pos_src[:, 2] / 1000.0
    alt2 = pos_dst[:, 2] / 1000.0
    alt_diff = alt2 - alt1
    distance = torch.sqrt(horizontal_distance**2 + alt_diff**2)
    return distance.unsqueeze(1)

def compute_euclidean_distance(pos_src: torch.Tensor, pos_dst: torch.Tensor) -> torch.Tensor:
    """
    Calcula la distància euclidiana entre dos vectors de posicions (assumint que estan en un espai 3D euclidià).
    Retorna la distància en km com un tensor de mida (N, 1).
    """
    distance = torch.norm(pos_src - pos_dst, p=2, dim=1, keepdim=True)
    return distance

def ensure_connectivity(edge_index: torch.Tensor, pos: torch.Tensor, k_neighbors: int) -> torch.Tensor:
    """
    Comprova els graus dels nodes i, per aquells sense connexió, afegeix una aresta amb el seu veí més proper.
    """
    num_nodes = pos.size(0)
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    for i in edge_index[0]:
        degrees[i] += 1
    new_edges = []
    for node in range(num_nodes):
        if degrees[node] == 0:
            knn_edge = knn_graph(pos, k=1, loop=False)
            mask = (knn_edge[0] == node)
            if mask.sum() > 0:
                idx = torch.nonzero(mask)[0]
                new_edges.append(knn_edge[:, idx])
    if new_edges:
        new_edges = torch.cat(new_edges, dim=1)
        edge_index = torch.cat([edge_index, new_edges], dim=1)
    return edge_index


def add_multiscale_edges(pos: torch.Tensor, x: torch.Tensor, multiscale_radius_quantile: float, 
                         edge_distance_scale: float, use_metric: bool):
    """
    Crea un segon conjunt d’arestes amb quantil superior per connexions regionals i duplica per obtenir
    informació en ambdues direccions.
    """
    num_nodes = pos.size(0)
    if num_nodes < 2:
        raise ValueError("No hi ha suficients nodes per al graf multiescala.")
    sorted_indices = torch.argsort(pos[:, 0])
    sample_size = min(50, num_nodes)
    sample_indices = sorted_indices[:sample_size]
    sample_pos = pos[sample_indices]
    dists = torch.cdist(sample_pos, sample_pos)
    dists_flat = dists[dists > 0]
    radius = torch.quantile(dists_flat, multiscale_radius_quantile).item()
    directed_edge_index = radius_graph(pos, r=radius, loop=False)
    
    src = directed_edge_index[0]
    dst = directed_edge_index[1]

    if use_metric:
        edge_attr_dist = compute_euclidean_distance(pos[src], pos[dst]) / edge_distance_scale
    else:
        edge_attr_dist = compute_geodesic_distance(pos[src], pos[dst]) / edge_distance_scale

    diff_temp = (x[src, 0] - x[dst, 0]).unsqueeze(1)
    diff_humitat = (x[src, 1] - x[dst, 1]).unsqueeze(1)
    diff_pluja = (x[src, 2] - x[dst, 2]).unsqueeze(1)
    diff_ventFor = (x[src, 3] - x[dst, 3]).unsqueeze(1)
    diff_patm = (x[src, 4] - x[dst, 4]).unsqueeze(1)
    diff_ventDir_sin = (x[src, 6] - x[dst, 6]).unsqueeze(1)
    diff_ventDir_cos = (x[src, 7] - x[dst, 7]).unsqueeze(1)
    edge_attr = torch.cat([edge_attr_dist, diff_temp, diff_humitat, diff_pluja, diff_ventFor, diff_patm, diff_ventDir_sin, diff_ventDir_cos], dim=1)
    rev_edge_index = directed_edge_index[[1, 0]]
    rev_edge_attr = torch.cat([edge_attr_dist, -diff_temp, -diff_humitat, -diff_pluja, -diff_ventFor, -diff_patm, -diff_ventDir_sin, -diff_ventDir_cos], dim=1)
    full_edge_index = torch.cat([directed_edge_index, rev_edge_index], dim=1)
    full_edge_attr = torch.cat([edge_attr, rev_edge_attr], dim=0)
    return full_edge_index, full_edge_attr


def create_edge_index_and_attr(pos: torch.Tensor, x: torch.Tensor, k_neighbors: int,
                               radius_quantile: float, edge_distance_scale: float,
                               add_multiscale: bool, multiscale_radius_quantile: float,
                               max_alt_diff: float, add_edge_weight: bool, edge_decay_length: float,
                               use_metric: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construeix l'edge_index i edge_attr del graf amb les millores:
      - Selecció determinista (ordenació per latitud) per radius o knn.
      - Duplicitat d'arestes amb gradients signats.
      - Garantir connectivitat.
      - Opcionalment, afegir arestes multiescala.
      - Opcionalment, filtrar per diferència d'altitud.
      - Opcionalment, afegir edge weighting.
    """
    num_nodes = pos.size(0)
    if num_nodes < 2:
        raise ValueError("No hi ha suficients nodes per construir el graf.")
    
    sorted_indices = torch.argsort(pos[:, 0])
    pos_ordered = pos[sorted_indices]
    if num_nodes < 50:
        directed_edge_index = knn_graph(pos_ordered, k=k_neighbors, loop=False)
    else:
        sample_size = min(50, num_nodes)
        sample_indices = sorted_indices[:sample_size]
        sample_pos = pos[sample_indices]
        dists = torch.cdist(sample_pos, sample_pos)
        dists_flat = dists[dists > 0]
        radius = torch.quantile(dists_flat, radius_quantile).item()
        directed_edge_index = radius_graph(pos_ordered, r=radius, loop=False)
    
    mapping = torch.zeros(num_nodes, dtype=torch.long)
    mapping[sorted_indices] = torch.arange(num_nodes)
    directed_edge_index = mapping[directed_edge_index]
    
    src = directed_edge_index[0]
    dst = directed_edge_index[1]

    if use_metric:
        edge_attr_dist = compute_euclidean_distance(pos[src], pos[dst]) / edge_distance_scale
    else:
        edge_attr_dist = compute_geodesic_distance(pos[src], pos[dst]) / edge_distance_scale

    diff_temp = (x[src, 0] - x[dst, 0]).unsqueeze(1)
    diff_humitat = (x[src, 1] - x[dst, 1]).unsqueeze(1)
    diff_pluja = (x[src, 2] - x[dst, 2]).unsqueeze(1)
    diff_ventFor = (x[src, 3] - x[dst, 3]).unsqueeze(1)
    diff_patm = (x[src, 4] - x[dst, 4]).unsqueeze(1)
    abs_diff_alt = torch.abs(x[src, 5] - x[dst, 5]).unsqueeze(1) # Diferència absoluta d’altitud
    diff_ventDir_sin = (x[src, 6] - x[dst, 6]).unsqueeze(1)
    diff_ventDir_cos = (x[src, 7] - x[dst, 7]).unsqueeze(1)
    edge_attr = torch.cat([edge_attr_dist, diff_temp, diff_humitat, diff_pluja, diff_ventFor, diff_patm, abs_diff_alt, diff_ventDir_sin, diff_ventDir_cos], dim=1)
    
    rev_edge_index = directed_edge_index[[1, 0]]
    rev_edge_attr = torch.cat([edge_attr_dist, -diff_temp, -diff_humitat, -diff_pluja, -diff_ventFor, -diff_patm, abs_diff_alt, -diff_ventDir_sin, -diff_ventDir_cos], dim=1)
    full_edge_index = torch.cat([directed_edge_index, rev_edge_index], dim=1)
    full_edge_attr = torch.cat([edge_attr, rev_edge_attr], dim=0)
    
    full_edge_index = ensure_connectivity(full_edge_index, pos, k_neighbors)
    
    if add_multiscale:
        ms_edge_index, ms_edge_attr = add_multiscale_edges(pos, x, multiscale_radius_quantile, edge_distance_scale)
        full_edge_index = torch.cat([full_edge_index, ms_edge_index], dim=1)
        full_edge_attr = torch.cat([full_edge_attr, ms_edge_attr], dim=0)
    
    if max_alt_diff is not None:
        src = full_edge_index[0]
        dst = full_edge_index[1]
        alt_diff = torch.abs(pos[src][:, 2] - pos[dst][:, 2]) / 1000.0
        mask = alt_diff <= max_alt_diff
        full_edge_index = full_edge_index[:, mask]
        full_edge_attr = full_edge_attr[mask]
    
    if add_edge_weight:
        weight = torch.exp(-full_edge_attr[:, 0] / edge_decay_length).unsqueeze(1)
        full_edge_attr = torch.cat([full_edge_attr, weight], dim=1)
    
    return full_edge_index, full_edge_attr


def compute_graph_metadata(data):
    # Grau mitjà: nombre total d'arestes (dividit per 2, ja que cada aresta apareix dues vegades) dividit pel nombre de nodes.
    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1) // 2  # Ja que s'han concatenat les arestes inverses.
    mean_degree = (num_edges * 2) / num_nodes  if num_nodes > 0 else 0
    
    # Radi efectiu: mitjana de la primera columna d'edge_attr (assumint que és la distància geodèsica escalada).
    effective_radius = data.edge_attr[:, 0].mean().item() if data.edge_attr.size(0) > 0 else 0
    
    return {"mean_degree": mean_degree, "effective_radius": effective_radius}


def sanity_check_node(data, node_index, num_neighbors=5):
    """
    Realitza un control de qualitat per a un node donat.
    Llista els primers 'num_neighbors' veïns del node i mostra informació rellevant.
    """
    # Trobar els indices on el node és origen
    mask = data.edge_index[0] == node_index
    neighbors = data.edge_index[1][mask]
    
    # Si no té veïns, notifica-ho
    if neighbors.numel() == 0:
        print(f"El node {node_index} no té veïns.")
        return
    
    print(f"Node {node_index} té {neighbors.numel()} veïns. Mostrant els primers {num_neighbors}:")
    for i, neighbor in enumerate(neighbors[:num_neighbors]):
        # Mostrar informació bàsica: id del node veí i la distància de l'aresta
        edge_mask = (data.edge_index[0] == node_index) & (data.edge_index[1] == neighbor)
        edge_attr = data.edge_attr[edge_mask]
        # Suposant que la primera columna d'edge_attr és la distància geodèsica escalada
        distance = edge_attr[0, 0].item() if edge_attr.size(0) > 0 else None
        print(f"  Veí {i+1}: Node {neighbor.item()} amb distància: {distance:.2f}")


def assign_gpu_device(file_idx: int, gpu_devices: list) -> str:
    """
    Assigna un dispositiu GPU per a cada fitxer basant-se en l'índex del fitxer (round-robin).
    """
    return gpu_devices[file_idx % len(gpu_devices)]


def process_file(file_path: str, input_root: str, output_root: str, k_neighbors: int,
                 radius_quantile: float, edge_distance_scale: float, use_metric: bool,
                 exclude_temporal_norm: bool, gpu_device: str, add_multiscale: bool,
                 multiscale_radius_quantile: float, max_alt_diff: float, add_edge_weight: bool,
                 edge_decay_length: float, pressure_ref: float, log_transform_pluja: bool,
                 add_wind_components: bool, include_year_feature: bool, 
                 precomputed_norm_params: dict = None):
    """
    Processa un fitxer CSV preprocessat per convertir-lo en un objecte Data.
    S'assumeix que el CSV ja està net.
    """
    try:
        df = pd.read_csv(file_path)
        if not set(REQUIRED_COLUMNS).issubset(df.columns):
            logging.error(f"El fitxer {file_path} no conté totes les columnes requerides.")
            return {"status": "fail", "file": file_path}
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        device = torch.device(gpu_device)
        x, norm_params = create_node_features(df, exclude_temporal_norm, add_wind_components, pressure_ref, log_transform_pluja, 
                                              add_station_id_feature=False, precomputed_norm_params=precomputed_norm_params)
        pos = create_position_tensor(df, use_metric).to(device)
        x = x.to(device)
        num_nodes = x.size(0)
        if num_nodes < 2:
            logging.info(f"El fitxer {file_path} té menys d'una connexió possible (num_nodes={num_nodes}). S'omet.")
            return {"status": "skip", "file": file_path}
        edge_index, edge_attr = create_edge_index_and_attr(pos, x, k_neighbors, radius_quantile,
                                                           edge_distance_scale, add_multiscale, multiscale_radius_quantile,
                                                           max_alt_diff, add_edge_weight, edge_decay_length, use_metric)
        data = Data(x=x.cpu(), pos=pos.cpu(), edge_index=edge_index.cpu(), edge_attr=edge_attr.cpu())
        data.ids = list(df['id'])
        data.fonts = list(df['Font'])
        if 'Timestamp' in df.columns:
            data.timestamp = df['Timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
            if include_year_feature:
                data.year = int(df['Timestamp'].dt.year.iloc[0])
        data.norm_params = norm_params

        # Afegir metadades del graf per documentar-lo
        meta = compute_graph_metadata(data)
        data.meta = meta
        logging.info(f"Metadades del graf: Grau mitjà = {meta['mean_degree']:.2f}, Radi efectiu = {meta['effective_radius']:.2f}")

        # Opcional: Realitzar una validació de sanity check per a un node específic (exemple: primer node)
        sanity_check_node(data, node_index=0, num_neighbors=5)
        
        # Afegir, opcionalment, l'índex de node (per futurs embeddings)
        rel_path = os.path.relpath(file_path, input_root)
        rel_path_pt = rel_path.replace("dadesPC_utc.csv", "pt")
        output_file = os.path.join(output_root, rel_path_pt)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save(data, output_file)
        logging.info(f"Conversió correcta: {file_path} -> {output_file}")
        return {"status": "success", "file": file_path}
    except Exception as e:
        logging.error(f"Error en processar {file_path}: {e}")
        return {"status": "fail", "file": file_path, "error": str(e)}


def process_all_files(input_root: str, output_root: str, max_workers: int, k_neighbors: int,
                      radius_quantile: float, edge_distance_scale: float, use_metric: bool,
                      exclude_temporal_norm: bool, gpu_devices: list, add_multiscale: bool,
                      multiscale_radius_quantile: float, max_alt_diff: float, add_edge_weight: bool,
                      edge_decay_length: float, pressure_ref: float, log_transform_pluja: bool,
                      add_wind_components: bool, include_year_feature: bool, group_by_period: str,
                      generate_sequence: bool, node_coverage_analysis: bool,
                      precomputed_norm_params: dict = None):
    """
    Processa tots els fitxers CSV preprocessats dins d'input_root en paral·lel i
    opcionalment agrupa els resultats per període o genera seqüències temporals.
    També realitza una anàlisi de cobertura de nodes si s'activa node_coverage_analysis.
    Retorna un resum amb els fitxers processats correctament i els fallits.
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
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, fp in enumerate(files_to_process):
            assigned_gpu = assign_gpu_device(idx, gpu_devices)
            futures[executor.submit(process_file, fp, input_root, output_root, k_neighbors,
                                      radius_quantile, edge_distance_scale, use_metric,
                                      exclude_temporal_norm, assigned_gpu, add_multiscale,
                                      multiscale_radius_quantile, max_alt_diff, add_edge_weight,
                                      edge_decay_length, pressure_ref, log_transform_pluja,
                                      add_wind_components, include_year_feature,
                                      precomputed_norm_params)] = fp
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processant fitxers", unit="fitxer"):
            results.append(future.result())
    # Resum final
    success = [r for r in results if r and r.get("status") == "success"]
    failed = [r for r in results if r and r.get("status") == "fail"]
    logging.info(f"Processament finalitzat: {len(success)} èxit, {len(failed)} fallits.")
    print(f"Processament finalitzat: {len(success)} èxit, {len(failed)} fallits.")
    
    # Opcional: Agrupar resultats per període
    if group_by_period in ["day", "month"]:
        group_processed_files(output_root, group_by_period)
    # Opcional: Generar seqüència temporal (per exemple, agrupant per dia)
    if generate_sequence:
        generate_sequence_files(output_root, group_by_period)
    # Opcional: Anàlisi de cobertura de nodes
    if node_coverage_analysis:
        analyze_node_coverage(output_root)


def group_processed_files(output_root: str, period: str):
    """
    Agrupa els fitxers .pt processats en un únic fitxer per període (dia o mes).
    """
    grouped = {}
    for root, dirs, files in os.walk(output_root):
        for file in files:
            if file.endswith(".pt"):
                full_path = os.path.join(root, file)
                # Suposem que la ruta conté la data; extreure-la del nom del directori
                # Per exemple, /.../2016/07/15/xx.pt -> "2016-07-15" o "2016-07" segons el període
                parts = os.path.normpath(full_path).split(os.sep)
                try:
                    year = parts[-4]
                    month = parts[-3]
                    day = parts[-2]
                    if period == "day":
                        key = f"{year}-{month}-{day}"
                    elif period == "month":
                        key = f"{year}-{month}"
                    else:
                        key = "all"
                except Exception as e:
                    key = "all"
                grouped.setdefault(key, []).append(full_path)
    for key, files in grouped.items():
        data_list = []
        for fp in files:
            data = torch.load(fp)
            data_list.append(data)
        out_file = os.path.join(output_root, f"group_{key}.pt")
        torch.save(data_list, out_file)
        logging.info(f"Grup {key} guardat amb {len(data_list)} gràfics a {out_file}")


def generate_sequence_files(output_root: str, period: str):
    """
    Genera un fitxer que conté una llista de Data objects ordenats temporalment per el període especificat.
    Aquesta funció pot ajudar a fer forecast seqüencial.
    """
    sequences = {}
    for root, dirs, files in os.walk(output_root):
        for file in files:
            if file.endswith(".pt") and not file.startswith("group_"):
                full_path = os.path.join(root, file)
                data = torch.load(full_path)
                # Extreure data del camp timestamp
                if hasattr(data, "timestamp"):
                    ts = datetime.strptime(data.timestamp, "%Y-%m-%d %H:%M:%S")
                    if period == "day":
                        key = ts.strftime("%Y-%m-%d")
                    elif period == "month":
                        key = ts.strftime("%Y-%m")
                    else:
                        key = "all"
                    sequences.setdefault(key, []).append((ts, data))
    for key, seq in sequences.items():
        seq.sort(key=lambda x: x[0])
        seq_data = [d for t, d in seq]
        out_file = os.path.join(output_root, f"sequence_{key}.pt")
        torch.save(seq_data, out_file)
        logging.info(f"Seqüència {key} guardada amb {len(seq_data)} gràfics a {out_file}")


def analyze_node_coverage(output_root: str):
    """
    Analitza la cobertura de nodes en tots els fitxers .pt processats, informant la freqüència d'aparició per cada ID.
    """
    coverage = defaultdict(int)
    for root, dirs, files in os.walk(output_root):
        for file in files:
            if file.endswith(".pt") and not file.startswith("group_") and not file.startswith("sequence_"):
                full_path = os.path.join(root, file)
                data = torch.load(full_path)
                for node_id in data.ids:
                    coverage[node_id] += 1
    total = sum(coverage.values())
    logging.info("Anàlisi de cobertura de nodes:")
    for node_id, count in sorted(coverage.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"ID: {node_id} apareix {count} vegades ({(count/total)*100:.2f}% del total)")


def parse_gpu_devices(gpu_str: str) -> list:
    """Converteix una cadena separada per comes en una llista de dispositius GPU."""
    return [d.strip() for d in gpu_str.split(",") if d.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Converteix fitxers CSV preprocessats en objectes Data per entrenar MeteoGraphSAGE.")
    parser.add_argument("--input_root", type=str, default=DEFAULT_INPUT_ROOT, help="Directori d'entrada dels CSV preprocessats.")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Directori de sortida per als fitxers .pt.")
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS, help="Nombre màxim de treballadors.")
    parser.add_argument("--k_neighbors", type=int, default=DEFAULT_K_NEIGHBORS, help="Nombre de veïns per knn_graph si num_nodes < 50.")
    parser.add_argument("--radius_quantile", type=float, default=DEFAULT_RADIUS_QUANTILE, help="Quantil per calcular el radi en radius_graph si num_nodes >= 50.")
    parser.add_argument("--edge_distance_scale", type=float, default=DEFAULT_EDGE_DISTANCE_SCALE, help="Factor per escalar la distància d'aresta.")
    parser.add_argument("--use_metric_pos", action="store_true", help="Si s'activa, converteix posicions a coordenades cartesianes 3D en km.")
    parser.add_argument("--exclude_temporal_norm", action="store_true", help="Si s'activa, les features temporals no es normalitzen.")
    parser.add_argument("--gpu_devices", type=str, default="cuda:0", help="Llista de dispositius GPU separats per comes (ex: 'cuda:0,cuda:1').")
    parser.add_argument("--add_multiscale", action="store_true", help="Si s'activa, s'afegeixen arestes multiescala.")
    parser.add_argument("--multiscale_radius_quantile", type=float, default=DEFAULT_MULTISCALE_RADIUS_QUANTILE, help="Quantil per arestes multiescala.")
    parser.add_argument("--max_alt_diff", type=float, default=DEFAULT_MAX_ALT_DIFF, help="Màxim diferència d'altitud (en km) per permetre una aresta.")
    parser.add_argument("--add_edge_weight", action="store_true", help="Si s'activa, s'afegeix un pes a cada aresta.")
    parser.add_argument("--edge_decay_length", type=float, default=DEFAULT_EDGE_DECAY_LENGTH, help="Longitud de decaïment per edge weighting (en km).")
    parser.add_argument("--pressure_ref", type=float, default=DEFAULT_PRESSURE_REF, help="Valor de referència per normalitzar la pressió (hPa).")
    parser.add_argument("--log_transform_pluja", action="store_true", help="Si s'activa, s'aplica log(Pluja+1) a la precipitació.")
    parser.add_argument("--add_wind_components", action="store_true", help="Si s'activa, s'afegeixen les components zonal i meridional del vent.")
    parser.add_argument("--include_year_feature", action="store_true", help="Si s'activa, s'afegeix l'any com a atribut global del Data object.")
    parser.add_argument("--group_by_period", type=str, choices=GROUP_BY_PERIOD_CHOICES, default="none", help="Agrupa els resultats per període ('day' o 'month').")
    parser.add_argument("--generate_sequence", action="store_true", help="Si s'activa, genera seqüències temporals (una per període) amb els Data objects.")
    parser.add_argument("--node_coverage_analysis", action="store_true", help="Si s'activa, analitza la cobertura de nodes i genera un informe.")
    parser.add_argument("--PC_norm_params", type=str, default=None, help="Path al fitxer JSON amb paràmetres dels Països Catalans de normalització.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Carregar els parametres dels Paisos Catalans si s'ha proporcionat
    PC_norm_params = None
    if args.global_norm_params is not None:
        with open(args.PC_norm_params, 'r') as f:
            PC_norm_params = json.load(f)

    logging.info(f"Iniciant processament amb input_root={args.input_root} i output_root={args.output_root}")
    gpu_devices = parse_gpu_devices(args.gpu_devices)
    process_all_files(args.input_root, args.output_root, args.max_workers, args.k_neighbors,
                      args.radius_quantile, args.edge_distance_scale, args.use_metric_pos,
                      args.exclude_temporal_norm, gpu_devices, args.add_multiscale,
                      args.multiscale_radius_quantile, args.max_alt_diff, args.add_edge_weight,
                      args.edge_decay_length, args.pressure_ref, args.log_transform_pluja,
                      args.add_wind_components, args.include_year_feature, args.group_by_period,
                      args.generate_sequence, args.node_coverage_analysis,
                      precomputed_norm_params=PC_norm_params)


if __name__ == "__main__":
    main()
