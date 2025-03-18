#!/usr/bin/env python3
"""
Script per a l'extracció i el preprocessament dels fitxers CSV amb dades meteorològiques.

Aquest codi:
  - Llegeix els fitxers originals sense modificar-los (fes servir 'utf-8' i, si falla, 'latin-1').
  - Extreu la data i l'hora del nom del fitxer per generar la columna 'Timestamp'
    (format: "YYYY-MM-DD HH:00:00"); si l'hora extreta és ≥ 24, s'omet el fitxer.
  - Filtra les dades per mantenir només les fonts oficials (segons la llista FONTS_OFICIALS).
  - Selecciona les columnes d'interès:
       * Per als fitxers dels anys 2016-2023: ['id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt',
         'VentDir', 'VentFor', 'Patm', 'lat', 'lon']
       * Per als fitxers de l'any 2015 (que poden no tenir 'Patm'): ['id', 'Font', 'Temp', 'Humitat',
         'Pluja', 'Alt', 'VentDir', 'VentFor', 'lat', 'lon']
  - Converteix les columnes numèriques.
  - Processa la variable "Pluja": calcula la pluja real per hora com la diferència
    entre el valor acumulat actual i el del fitxer de l'hora anterior (si és la primera hora, es considera 0).
  - Imputa els valors nuls de les variables 'Temp', 'Humitat', 'VentDir', 'VentFor' i 'Patm' (si existeix)
    fent interpolació amb dades vàlides dels fitxers adjacents:
      • Es cerca primer dins d'un marge de 8 hores (per hores immediates anterior i posterior);
      • Si no es troben, es comprova la mateixa hora del dia anterior (o posterior).
      • Si encara no es troben dades vàlides, el valor quedarà com a NaN.
  - Abans de desar el fitxer processat, s'eliminen les estacions (files) que contenen algun NaN
    en alguna variable requerida.
  - Ajusta els intervals de 'Humitat' (0-100%), 'VentFor' (0-200km/h) i 'VentDir' (0-360º).
  - Desa els fitxers processats en un directori separat, mantenint l'estructura original.
  - Mostra una barra de progrés per saber quants fitxers s'han processat.

Les dades originals, en fitxers amb format csv, romanen intactes. 
Les dades preprocessades es guarden en un nou directori amb la mateixa estructura de carpetes.

A més, aquest codi fa ús de diverses millores per tal d'accelerar l'execució mitjançant una GPU:
  - Processament paral·lel de fitxers amb ProcessPoolExecutor.
  - Ús de Cupy per a càlculs numèrics crítics (interpolació i càlcul de la pluja).
  - Manteniment de la lògica original: lectura, filtratge, extracció de Timestamp, conversió,
    interpolació, ajust de valors i desada dels fitxers processats.
"""

import os
import re
import pandas as pd
import numpy as np
import cupy as cp
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configurar el logger per desar els logs en un fitxer
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"processament_dades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename)]
)

# Llista de fonts fiables d'estacions meteorològiques
FONTS_OFICIALS = [
    "Aemet", "METEOCAT", "METEOCAT_WEB", "Meteoclimatic", "Vallsdaneu",
    "SAIH", "avamet", "Meteoprades", "MeteoPirineus", "WLINK_DAVIS"
]

# Anys per a interpolació i anys que es pre-processaran
YEARS_FOR_INTERPOLATION = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
PROCESSED_YEARS = [year for year in YEARS_FOR_INTERPOLATION if year >= 2016]

# Diccionari per emmagatzemar en caché els fitxers llegits (per cada procés)
file_cache = {}

def get_file_path_for_timestamp(root_directory: str, timestamp: datetime) -> str:
    any_str = f"{timestamp.year}"
    mes_str = f"{timestamp.month:02d}"
    dia_str = f"{timestamp.day:02d}"
    hora_str = f"{timestamp.hour:02d}"
    filename = f"{timestamp.year}{timestamp.month:02d}{timestamp.day:02d}{timestamp.hour:02d}dadesPC_utc.csv"
    return os.path.join(root_directory, any_str, mes_str, dia_str, hora_str, filename)

def load_file(file_path: str) -> pd.DataFrame:
    if file_path in file_cache:
        return file_cache[file_path]
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path, encoding='utf-8', na_values=['', ' '],
                         quotechar='"', sep=',', engine='python', on_bad_lines='skip')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin-1', na_values=['', ' '],
                             quotechar='"', sep=',', engine='python', on_bad_lines='skip')
            logging.warning(f"Fitxer {file_path} llegit amb 'latin-1'.")
        except Exception as e:
            logging.error(f"Error llegint {file_path} amb 'latin-1': {e}")
            return None
    except Exception as e:
        logging.error(f"Error llegint {file_path}: {e}")
        return None
    file_cache[file_path] = df
    return df

def get_station_value(root_directory: str, timestamp: datetime, station_id, variable: str):
    file_path = get_file_path_for_timestamp(root_directory, timestamp)
    df = load_file(file_path)
    if df is None or df.empty:
        return None
    if 'Font' in df.columns:
        df = df[df['Font'].isin(FONTS_OFICIALS)]
        if df.empty:
            return None
    else:
        return None
    if 'id' not in df.columns:
        return None
    df_station = df[df['id'] == station_id]
    if df_station.empty:
        return None
    if variable not in df_station.columns:
        return None
    try:
        value = pd.to_numeric(df_station.iloc[0][variable], errors='coerce')
    except Exception as e:
        logging.error(f"Error convertint {variable} per l'estació {station_id} al fitxer {file_path}: {e}")
        return None
    if variable == 'Pluja' and pd.isna(value):
        value = 0.0
    return value

def get_neighbor_value(root_directory: str, current_timestamp: datetime, station_id, variable: str, direction: str):
    delta = -1 if direction == 'backward' else 1
    for offset in range(1, 9):
        candidate_time = current_timestamp + timedelta(hours=delta * offset)
        value = get_station_value(root_directory, candidate_time, station_id, variable)
        if value is not None and not pd.isna(value):
            return value, candidate_time
    candidate_time = current_timestamp + timedelta(days=delta)
    value = get_station_value(root_directory, candidate_time, station_id, variable)
    if value is not None and not pd.isna(value):
        return value, candidate_time
    return None

def interpolate_value(root_directory: str, current_timestamp: datetime, station_id, variable: str):
    backward = get_neighbor_value(root_directory, current_timestamp, station_id, variable, 'backward')
    forward = get_neighbor_value(root_directory, current_timestamp, station_id, variable, 'forward')
    if backward is not None and forward is not None:
        value_back, time_back = backward
        value_forward, time_forward = forward
        total_seconds = (time_forward - time_back).total_seconds()
        if total_seconds == 0:
            logging.warning(f"Temps zero per interpolació en {variable} per l'estació {station_id} al {current_timestamp}.")
            return cp.nan
        fraction = (current_timestamp - time_back).total_seconds() / total_seconds
        # Ús de Cupy per fer el càlcul numèric
        interpolated = cp.array(value_back) + (cp.array(value_forward) - cp.array(value_back)) * fraction
        return float(interpolated.get()) if hasattr(interpolated, 'get') else float(interpolated)
    else:
        return cp.nan

def preprocess_csv(file_path: str, root_directory: str) -> pd.DataFrame:
    # Llegeix el fitxer sense modificar-lo (prova amb 'utf-8' i, si falla, amb 'latin-1')
    if file_path is None or not os.path.exists(file_path):
        logging.error(f"El fitxer {file_path} no existeix.")
        return None
    try:
        data = pd.read_csv(file_path, encoding='utf-8', na_values=['', ' '],
                           quotechar='"', sep=',', engine='python', on_bad_lines='skip')
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(file_path, encoding='latin-1', na_values=['', ' '],
                               quotechar='"', sep=',', engine='python', on_bad_lines='skip')
            logging.warning(f"Fitxer {file_path} llegit amb 'latin-1'.")
        except Exception as e:
            logging.error(f"Error llegint {file_path} amb 'latin-1': {e}")
            return None
    except Exception as e:
        logging.error(f"Error llegint {file_path}: {e}")
        return None

    if data.empty or data.columns.size == 0:
        logging.warning(f"Fitxer {file_path} llegit sense columnes. Es retorna DataFrame buit.")
        return pd.DataFrame()

    # Filtrar per fonts oficials
    if 'Font' in data.columns:
        data = data[data['Font'].isin(FONTS_OFICIALS)]
        if data.empty:
            logging.warning(f"Fitxer {file_path} no conté fonts oficials després del filtratge.")
            return pd.DataFrame()
    else:
        logging.warning(f"La columna 'Font' no està present a {file_path}.")
        return None

    # Extreure el Timestamp a partir del nom del fitxer
    base_filename = os.path.basename(file_path)
    match = re.match(r'(\d{4})(\d{2})(\d{2})(\d{2})dadesPC_utc\.csv', base_filename)
    if match:
        year_str, month, day, hour = match.groups()
        hour_int = int(hour)
        if hour_int >= 24:
            logging.info(f"Fitxer {file_path} té hora invàlida ({hour_int}). S'omet el processament.")
            return None
        timestamp_str = f"{year_str}-{month}-{day} {hour}:00:00"
        try:
            ts = pd.to_datetime(timestamp_str, format='%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logging.error(f"Error convertint el Timestamp de {file_path}: {e}")
            return None
        data['Timestamp'] = ts.strftime('%Y-%m-%d %H:%M:%S')
    else:
        logging.error(f"No s'ha pogut extreure el Timestamp de {file_path}.")
        return None

    year = ts.year
    # Si l'any no és 2015, la columna 'Patm' ha d'estar present
    if year != 2015 and 'Patm' not in data.columns:
        logging.error(f"El fitxer {file_path} per a l'any {year} no conté la columna 'Patm' requerida.")
        return None

    # Definir l'ordre de columnes segons si hi ha 'Patm'
    if 'Patm' in data.columns:
        column_order = ['id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt', 'VentDir', 'VentFor', 'Patm', 'lat', 'lon']
    else:
        column_order = ['id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt', 'VentDir', 'VentFor', 'lat', 'lon']

    cols_present = [col for col in column_order if col in data.columns]
    if 'Timestamp' not in data.columns:
        cols_present.append('Timestamp')
    data = data[cols_present]

    # Converteix les columnes numèriques
    numeric_columns = ['Temp', 'Humitat', 'Alt', 'VentFor', 'lat', 'lon', 'Pluja']
    if 'Patm' in data.columns:
        numeric_columns.append('Patm')
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['Pluja'] = data['Pluja'].fillna(0)

    # Processar 'VentDir': mapegem les direccions a graus
    if 'VentDir' in data.columns:
        mapping_direccions = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
            'Calma': np.nan, 'Variable': np.nan, 'Null': np.nan, '': np.nan
        }
        data['VentDir'] = data['VentDir'].map(mapping_direccions)

    imputer_columns = [col for col in numeric_columns if col in data.columns]
    files_antes = data.shape[0]
    data = data.dropna(subset=imputer_columns, how='all')
    if data.shape[0] < files_antes:
        logging.info(f"S'han eliminat {files_antes - data.shape[0]} files buides de {file_path}.")
    if data.empty:
        logging.warning(f"Després del preprocessament, {file_path} queda buit.")
        return data

    current_timestamp = ts

    # Processar la variable "Pluja": càlcul de la diferència respecte l'hora anterior
    pluja_values = []
    for idx, row in data.iterrows():
        station_id = row['id']
        current_pluja = row['Pluja'] if not pd.isna(row['Pluja']) else 0.0
        if current_timestamp.hour == 0:
            previous_pluja = 0.0
        else:
            previous_timestamp = current_timestamp - timedelta(hours=1)
            previous_pluja = get_station_value(root_directory, previous_timestamp, station_id, 'Pluja')
            if previous_pluja is None:
                previous_pluja = 0.0
        # Ús de Cupy per calcular la diferència
        cp_current = cp.array(current_pluja)
        cp_previous = cp.array(previous_pluja)
        pluja_real = cp_current - cp_previous
        pluja_real_val = float(pluja_real.get()) if hasattr(pluja_real, 'get') else float(pluja_real)
        if pluja_real_val < 0:
            pluja_real_val = 0.0
        pluja_values.append(pluja_real_val)
    data['Pluja'] = pluja_values

    # Imputar les altres variables amb interpolació (excepte 'Pluja' i 'Alt')
    variables_interp = ['Temp', 'Humitat', 'VentDir', 'VentFor']
    if 'Patm' in data.columns:
        variables_interp.append('Patm')
    for idx, row in data.iterrows():
        station_id = row['id']
        for var in variables_interp:
            if pd.isna(row[var]):
                interpolated = interpolate_value(root_directory, current_timestamp, station_id, var)
                data.at[idx, var] = interpolated

    n_antes = data.shape[0]
    data = data.dropna(subset=variables_interp, how='any')
    if data.shape[0] < n_antes:
        logging.info(f"S'han eliminat {n_antes - data.shape[0]} estacions per falta de dades interpolades.")
    if 'Humitat' in data.columns:
        data['Humitat'] = data['Humitat'].clip(0, 100)
    if 'VentFor' in data.columns:
        data['VentFor'] = data['VentFor'].clip(0, 200)
    if 'VentDir' in data.columns:
        data['VentDir'] = data['VentDir'].clip(0, 360)

    return data

def process_file(file_path: str, root_directory: str, processed_directory: str):
    df = preprocess_csv(file_path, root_directory)
    if df is not None and not df.empty:
        relative_path = os.path.relpath(os.path.dirname(file_path), root_directory)
        output_dir = os.path.join(processed_directory, relative_path)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.basename(file_path))
        try:
            df.to_csv(output_file, index=False)
            logging.info(f"Fitxer processat desat a: {output_file}")
        except Exception as e:
            logging.error(f"Error desant {output_file}: {e}")

def process_all_csvs_parallel(root_directory: str, processed_directory: str, max_workers: int = 8):
    files_to_process = []
    for root_dir, dirs, files in os.walk(root_directory):
        print("Processant directori:", root_dir)
        # Filtrar directoris segons subcadenes excloses
        dirs[:] = [d for d in dirs if not any(sub.lower() in d.lower() for sub in [
            "tauladades", "vextrems", "Admin_Estacions", "Clima", "Clima METEOCAT",
            "error_VAR", "html", "png", "var_vextrems"]) and not re.search(r'\d+_old', d)]
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
        futures = {executor.submit(process_file, file_path, root_directory, processed_directory): file_path for file_path in files_to_process}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processant fitxers", unit="fitxer"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processant {futures[future]}: {e}")

if __name__ == "__main__":
    root_directory = 'D:/DADES_METEO_PC'
    processed_directory = 'D:/DADES_METEO_PC_PREPROCESSADES_GPU_PARALLEL'
    process_all_csvs_parallel(root_directory, processed_directory, max_workers=8)
    logging.info("Processament finalitzat.")
    print("Processament finalitzat.")
