#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
compute_PC_norm_params.py

Script per calcular els paràmetres de normalització globals dels Països Catalans
(mitjana i desviació estàndard) a partir dels fitxers CSV meteorològics preprocessats amb "prep.py".

FUNCIONALITATS PRINCIPALS:
  - Llegeix tots els fitxers CSV de dades meteorològiques preprocessades (sortida de "prep.py").
  - Aplica el mateix processament que "toData.py" per generar totes les columnes derivades necessàries:
      VentDir_sin, VentDir_cos, hora_sin, hora_cos, dia_sin, dia_cos, cos_sza, DewPoint, PotentialTemp, Vent_u, Vent_v.
  - Calcula la mitjana i la desviació estàndard de cada feature definida a FEATURE_COLUMNS.
  - Desa aquests paràmetres globals en un fitxer JSON anomenat "PC_norm_params.json".
  - Genera i desa un histograma per a cada feature al directori "histogrames" (en format PNG).

INSTRUCCIONS D'ÚS:
  1. Important: cal haver executat primer l'script "prep.py" per obtenir els fitxers preprocessats.
  2. Assegura't que la variable "input_root" apunta al directori amb els fitxers preprocessats.
  3. Executa aquest script. El procés pot trigar depenent del volum de dades.
  4. Trobaràs el fitxer "PC_norm_params.json" amb les estadístiques globals i, opcionalment, els histogrames de cada variable al directori "histogrames".

REQUISITS:
  - Python 3.x
  - Llibreries: pandas, numpy, matplotlib, tqdm

AUTOR: Nil Farrés Soler
==============================================================================
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import calendar
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"compute_PC_norm_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename)]
)

TRAIN_YEARS = set(range(2016, 2023))
ADD_WIND_COMPONENTS = True

# Definició de FEATURE_COLUMNS que s'utilitzaran a toData.py
FEATURE_COLUMNS = [
    'Temp', 'Humitat', 'Pluja', 'VentFor', 'Patm', 'Alt_norm',
    'VentDir_sin', 'VentDir_cos', 'hora_sin', 'hora_cos', 'dia_sin', 
    'dia_cos', 'cos_sza', 'DewPoint', 'PotentialTemp', 'Vent_u', 'Vent_v'
]

# Ruta d'entrada:
input_root = "D:/DADES_METEO_PC_PREPROCESSADES"
# Ruta de sortida per als paràmetres dels Països Catalans
output_norm_params = "PC_norm_params.json"

# Suprimeix els warnings de divisió per zero
np.seterr(divide='ignore')

def extract_timestamp_from_filename(file_path: str) -> str:
    """
    Extreu el timestamp del nom del fitxer, que ha de tenir el format 'YYYYMMDDHHdadesPC_utc.csv'.
    Retorna el timestamp en format 'YYYY-MM-DD HH:MM:SS'.
    """
    base = os.path.basename(file_path)
    ts_str = base[:10]  # 'YYYYMMDDHH'
    ts = datetime.strptime(ts_str, "%Y%m%d%H")
    return ts.strftime("%Y-%m-%d %H:%M:%S")

##########################################
# Funcions derivades per generar features #
##########################################

def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Afegeix les features cícliques basades en Timestamp: hora_sin, hora_cos, dia_sin i dia_cos."""
    if not np.issubdtype(df['Timestamp'].dtype, np.datetime64):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['hora_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.hour / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.hour / 24)
    # Calcular el nombre de dies de l'any
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

def encode_wind_direction(df: pd.DataFrame, add_components: bool = False) -> pd.DataFrame:
    """
    Converteix 'VentDir' a 'VentDir_sin' i 'VentDir_cos'.
    Si s'especifica, afegeix les components zonal i meridional del vent.
    """
    try:
        # Convertir la columna 'VentDir' a float
        df['VentDir'] = df['VentDir'].astype(float)
        logging.debug("S'ha convertit la columna 'VentDir' a float correctament.")
    except Exception as e:
        logging.error(f"Error en convertir 'VentDir' a float: {e}")
        raise

    if df['VentDir'].dtype == object:
        mapping = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
            'Calma': 0
        }
        df['VentDir'] = df['VentDir'].map(mapping)
        logging.debug("S'ha aplicat el mapping de valors de 'VentDir'.")

    try:
        df['VentDir_sin'] = np.sin(np.deg2rad(df['VentDir']))
        df['VentDir_cos'] = np.cos(np.deg2rad(df['VentDir']))
        logging.debug("S'han creat les columnes 'VentDir_sin' i 'VentDir_cos'.")
    except Exception as e:
        logging.error(f"Error en calcular 'VentDir_sin' o 'VentDir_cos': {e}")
        raise

    if add_components:
        try:
            df['Vent_u'] = - df['VentFor'] * np.sin(np.deg2rad(df['VentDir']))
            df['Vent_v'] = - df['VentFor'] * np.cos(np.deg2rad(df['VentDir']))
            logging.debug("S'han creat correctament les components 'Vent_u' i 'Vent_v'.")
        except Exception as e:
            logging.error(f"Error en calcular 'Vent_u' i 'Vent_v': {e}")
            raise

    try:
        df.drop(columns=['VentDir'], inplace=True)
        logging.debug("S'ha eliminat la columna original 'VentDir'.")
    except Exception as e:
        logging.error(f"Error en eliminar la columna 'VentDir': {e}")
        raise

    return df


def process_df_for_norm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa el DataFrame per generar les columnes derivades necessàries
    per al vector final de features.
    """
    # Assegura't que 'Timestamp' és datetime
    if not np.issubdtype(df['Timestamp'].dtype, np.datetime64):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    df = add_cyclical_time_features(df)
    df = add_solar_features(df)
    df = encode_wind_direction(df, add_components=ADD_WIND_COMPONENTS)
    return df

##########################################
# Processament dels fitxers i càlcul global #
##########################################

# Llista de fitxers CSV (tots els que compleixen la condició)  
all_files = glob.glob(os.path.join(input_root, "**", "*dadesPC_utc.csv"), recursive=True)

data_list = []

for file in tqdm(all_files, desc="Processant fitxers"):
    # Filtra només els anys del conjunt d'entrenament
    base = os.path.basename(file)
    any_fitxer = int(base[:4])
    if any_fitxer not in TRAIN_YEARS:
        continue
    try:
        df = pd.read_csv(file)

        df = pd.read_csv(file)

        # Converteix la columna 'VentFor' a numèrica (si encara no ho és) i aplica la conversió a m/s
        df['VentFor'] = pd.to_numeric(df['VentFor'], errors='coerce').fillna(0) / 3.6

        # Comprova que el fitxer tingui les columnes originals (sense 'Timestamp')
        required_cols = ['id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt', 'VentDir', 'VentFor', 'Patm', 'lat', 'lon']
        if not set(required_cols).issubset(df.columns):
            print(f"El fitxer {file} no té totes les columnes requerides. S'omet.")
            continue
        
        # Si no hi ha la columna 'Timestamp', la afegeix extreient-la del nom del fitxer
        if 'Timestamp' not in df.columns:
            ts_str = extract_timestamp_from_filename(file)
            df.insert(0, 'Timestamp', ts_str)
        
        # Processar el DataFrame per obtenir les columnes derivades
        df = process_df_for_norm(df)

        if 'Alt_norm' not in df.columns:
            alt_mean = 454.3
            alt_std = 175.61
            df['Alt_norm'] = (df['Alt'] - alt_mean) / alt_std

        # Convertir la temperatura a Kelvin
        df['Temp'] = df['Temp'] + 273.15

        # Aplicar transformació logarítmica a 'Pluja'
        # Converteix la columna 'Pluja' a numèrica i substitueix qualsevol error o NaN per 0
        df['Pluja'] = pd.to_numeric(df['Pluja'], errors='coerce').fillna(0)
        # Assegura't que tots els valors siguin ≥ 0 (si algun és menor, el forcem a 0)
        df['Pluja'] = np.maximum(df['Pluja'], 0)

        df['Pluja'] = np.log1p(df['Pluja'])

        pressure_ref = 1013.0
        df = add_dew_point(df)
        df = add_potential_temperature(df, pressure_ref)

        # Reescalar la humitat de 0-100 a 0-1
        df['Humitat'] = df['Humitat'] / 100.0

        df['Patm'] = df['Patm'] - pressure_ref

        # Seleccionar les columnes definides per FEATURE_COLUMNS
        df = df[FEATURE_COLUMNS]

        data_list.append(df)
    except Exception as e:
        print(f"Error processant {file}: {e}")

if not data_list:
    raise ValueError("No s'han trobat fitxers vàlids per al càlcul de paràmetres.")

# Concatenar totes les dades
data_all = pd.concat(data_list, ignore_index=True)

# Calcular paràmetres globals per cada feature
PC_norm_params = {}
for col in FEATURE_COLUMNS:
    PC_norm_params[col] = {
        "mean": float(data_all[col].mean()),
        "std": float(data_all[col].std())
    }

# Guardar els paràmetres globals en format JSON
with open(output_norm_params, "w") as f:
    json.dump(PC_norm_params, f, indent=4)

print(f"Paràmetres dels Països Catalans calculats i guardats a {output_norm_params}")

# Defineix la carpeta on es desaran els histogrames
output_folder = "histogrames"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Generant i desant histogrammes per a cada feature:")

for col in FEATURE_COLUMNS:
    plt.figure(figsize=(8, 4))
    plt.hist(data_all[col].dropna(), bins=50, color='blue', alpha=0.7)
    plt.title(f"Distribució de {col}")
    plt.xlabel(col)
    plt.ylabel("Freqüència")
    plt.grid(True)
    
    # Desa l'imatge amb el nom de la feature
    save_path = os.path.join(output_folder, f"{col}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Histograma de {col} desat a {save_path}")
