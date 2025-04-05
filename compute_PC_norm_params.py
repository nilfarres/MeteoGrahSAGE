#!/usr/bin/env python3
"""
compute_PC_norm_params.py

Script per calcular els paràmetres de normalització dels Països Catalans (mitjana i desviació estàndard)
per als features dels nodes a partir dels fitxers CSV preprocessats (sortida de prep_GPU_parallel.py).

Els fitxers d'entrada han de tenir les columnes:
  'id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt', 'VentDir', 'VentFor', 'Patm', 'lat', 'lon', 'Timestamp'

Abans de calcular les estadístiques, s'apliquen les mateixes funcions que s'utilitzen a toData_GPU_parallel.py
per generar les columnes derivades: VentDir_sin, VentDir_cos, hora_sin, hora_cos, dia_sin, dia_cos, cos_sza, DewPoint i PotentialTemp.

Els paràmetres dels Països Catalans es guardaran en un fitxer JSON (per exemple, PC_norm_params.json).
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import calendar

# Definició de FEATURE_COLUMNS que s'utilitzaran a toData_GPU_parallel.py
FEATURE_COLUMNS = [
    'Temp', 'Humitat', 'Pluja', 'VentFor', 'Patm', 'Alt_norm',
    'VentDir_sin', 'VentDir_cos', 'hora_sin', 'hora_cos', 'dia_sin', 
    'dia_cos', 'cos_sza', 'DewPoint', 'PotentialTemp'
]

# Ruta d'entrada:
input_root = "D:/DADES_METEO_PC_PREPROCESSADES_GPU_PARALLEL"
# Ruta de sortida per als paràmetres dels Països Catalans
output_norm_params = "PC_norm_params.json"

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
    """
    Afegeix una nova columna 'cos_sza' que conté el cosinus de l'angle solar zenital,
    calculat a partir del Timestamp i la latitud de l'estació.
    """
    if not np.issubdtype(df['Timestamp'].dtype, np.datetime64):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    # Convertir latitud a radians
    lat_rad = np.deg2rad(df['lat'])
    day_of_year = df['Timestamp'].dt.dayofyear
    # Declinació solar aproximada (en graus) i convertida a radians
    dec_deg = 23.44 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
    dec_rad = np.deg2rad(dec_deg)
    # Hora local (suposem que Timestamp és hora local)
    hour_local = df['Timestamp'].dt.hour + df['Timestamp'].dt.minute / 60.0
    # Calcular HRA: (hora_local - 12)*15 graus, convertir a radians
    HRA_rad = np.deg2rad((hour_local - 12) * 15)
    
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
    # Si es volen afegir components addicionals (no necessari per aquest càlcul)
    if add_components:
        df['Vent_u'] = df['VentFor'] * df['VentDir_cos']
        df['Vent_v'] = df['VentFor'] * df['VentDir_sin']
    # Eliminar la columna original
    df.drop(columns=['VentDir'], inplace=True)
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
    df = encode_wind_direction(df, add_components=False)
    return df

##########################################
# Processament dels fitxers i càlcul global #
##########################################

# Llista de fitxers CSV (tots els que compleixen la condició)  
all_files = glob.glob(os.path.join(input_root, "**", "*dadesPC_utc.csv"), recursive=True)

data_list = []

for file in all_files:
    try:
        df = pd.read_csv(file)
        # Comprova que el fitxer tingui les columnes originals
        required_cols = ['id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt', 'VentDir', 'VentFor', 'Patm', 'lat', 'lon', 'Timestamp']
        if not set(required_cols).issubset(df.columns):
            print(f"El fitxer {file} no té totes les columnes requerides. S'omet.")
            continue
        # Processar el DataFrame per obtenir les columnes derivades
        df = process_df_for_norm(df)

        if 'Alt_norm' not in df.columns:
            alt_mean = 454.3
            alt_std = 175.61
            df['Alt_norm'] = (df['Alt'] - alt_mean) / alt_std

        df['Temp'] = df['Temp'] + 273.15

        # Activar o descativar la línia de sota en funció de si es vol fer o no a toData_GPU_parallel.py
        df['Pluja'] = np.log1p(df['Pluja'])

        pressure_ref = 1013.0
        df = add_dew_point(df)
        df = add_potential_temperature(df, pressure_ref)

        df['Patm'] = df['Patm'] - pressure_ref

        # Seleccionar les columnes definit per FEATURE_COLUMNS
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
