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

Les dades originals romanen intactes.
"""

import os                                  # Per a la gestió de fitxers i directoris
import re                                  # Per a les expressions regulars
import pandas as pd                        # Per a la manipulació de dades
import numpy as np                         # Per a càlculs numèrics
import logging                             # Per a la gestió de logs
from datetime import datetime, timedelta   # Per a la manipulació de dates i hores
from tqdm import tqdm                      # Per a la barra de progrés

# Configurar el logger perquè escrigui només en el fitxer de logs
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"processament_dades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename)]
)

# Llista de fonts fiables d'estacions meteorològiques
# (les dades d'altres fonts poden ser menys fiables o no estar completament actualitzades)
FONTS_OFICIALS = ["Aemet", "METEOCAT", "METEOCAT_WEB", "Meteoclimatic", "Vallsdaneu",
                   "SAIH", "avamet", "Meteoprades", "MeteoPirineus", "WLINK_DAVIS"]

# Definim els anys disponibles per a interpolació (inclou el 2015) i els anys que es pre-processaran (2016 a 2023)
YEARS_FOR_INTERPOLATION = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
PROCESSED_YEARS = [year for year in YEARS_FOR_INTERPOLATION if year >= 2016]

# Diccionari per emmagatzemar en caché els fitxers llegits (per evitar recàrregues innecessàries)
file_cache = {}

def get_file_path_for_timestamp(root_directory: str, timestamp: datetime) -> str:
    """
    Construeix el camí del fitxer a partir del timestamp i el directori arrel,
    tenint en compte que l'estructura és: any/mes/dia/hora/fitxer.
    Exemple: F:/DADES_METEO_PC/2023/08/25/00/2023082500dadesPC_utc.csv
    """
    any_str = f"{timestamp.year}"
    mes_str = f"{timestamp.month:02d}"
    dia_str = f"{timestamp.day:02d}"
    hora_str = f"{timestamp.hour:02d}"
    filename = f"{timestamp.year}{timestamp.month:02d}{timestamp.day:02d}{timestamp.hour:02d}dadesPC_utc.csv"
    return os.path.join(root_directory, any_str, mes_str, dia_str, hora_str, filename)

def load_file(file_path: str) -> pd.DataFrame:
    """
    Carrega el fitxer CSV utilitzant la codificació adequada i emmagatzema el resultat a la caché.
    """
    #Comprovem si el fitxer ja està a la caché per evitar recàrregues innecessàries
    if file_path in file_cache:
        return file_cache[file_path]
    if not os.path.exists(file_path):
        return None
    # Intentem llegir el fitxer amb 'utf-8' i, si falla, amb 'latin-1'. Si falla de nou, retornem None.
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
    """
    Obté el valor de la variable per a una estació concreta d'un fitxer corresponent al timestamp.
    Es busca per l'id d'estació i es filtra també per les fonts oficials.
    Retorna el valor (numèric) si és vàlid o None en cas contrari.
    """
    file_path = get_file_path_for_timestamp(root_directory, timestamp)
    df = load_file(file_path)
    if df is None or df.empty:
        return None
    # Filtrar per fonts fiables i per l'estació concreta
    if 'Font' in df.columns:
        df = df[df['Font'].isin(FONTS_OFICIALS)]
        if df.empty:
            return None
    else:
        return None
    #Busquem la fila que coincideixi amb l'id de l'estació i retornem el valor de la variable convertit a numèric
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
        logging.error(f"Error convertint la variable {variable} per l'estació {station_id} al fitxer {file_path}: {e}")
        return None
    # Si la variable és 'Pluja' i el valor és NaN, considerem que no plou i ho reemplacem per 0
    if variable == 'Pluja' and pd.isna(value):
        value = 0.0
    return value

def get_neighbor_value(root_directory: str, current_timestamp: datetime, station_id, variable: str, direction: str):
    """
    Cerca el valor vàlid més proper per a la variable d'una estació en la direcció indicada 
    (entenent "proper" com a dades dels fitxers d'hores adjacents).
    direction: 'backward' per hores anteriors, 'forward' per hores posteriors.
    
    La cerca es fa de la següent manera:
      1. Per cada hora dins d'un marge de 8 hores (offset 1 a 8) en la direcció especificada,
         es calcula el timestamp candidat i es comprova si l'estació té una dada vàlida.
      2. Si no es troba cap dada vàlida en aquest interval, es comprova la mateixa hora del dia anterior (o posterior).
    
    Retorna una tupla (valor, timestamp) si es troba, o None en cas contrari.
    """
    delta = -1 if direction == 'backward' else 1
    for offset in range(1, 9):
        candidate_time = current_timestamp + timedelta(hours=delta * offset)
        value = get_station_value(root_directory, candidate_time, station_id, variable)
        if value is not None and not pd.isna(value):
            return value, candidate_time
    # Cerca ampliada: mateixa hora del dia anterior (o posterior)
    candidate_time = current_timestamp + timedelta(days=delta)
    value = get_station_value(root_directory, candidate_time, station_id, variable)
    # Si s'ha trobat un valor vàlid, es retorna una tupla (valor, timestamp). Si no, es retorna None.
    if value is not None and not pd.isna(value):
        return value, candidate_time
    return None

def interpolate_value(root_directory: str, current_timestamp: datetime, station_id, variable: str):
    """
    Interpola el valor per a una variable d'una estació concreta en el fitxer actual.
    Es busca un valor vàlid a l'hora anterior i un a l'hora posterior utilitzant get_neighbor_value.
    Si es troben ambdós, es realitza una interpolació lineal; si no, es registra el cas i es retorna NaN.
    """
    backward = get_neighbor_value(root_directory, current_timestamp, station_id, variable, 'backward')
    forward = get_neighbor_value(root_directory, current_timestamp, station_id, variable, 'forward')

    # Si es troben valors vàlids per a les hores anterior i posterior, es realitza la interpolació
    if backward is not None and forward is not None:
        value_back, time_back = backward
        value_forward, time_forward = forward
        total_seconds = (time_forward - time_back).total_seconds()
        if total_seconds == 0:
            logging.warning(f"Temps zero per interpolació en {variable} per l'estació {station_id} al timestamp {current_timestamp}.")
            return np.nan
        fraction = (current_timestamp - time_back).total_seconds() / total_seconds
        interpolated = value_back + (value_forward - value_back) * fraction
        return interpolated
    # Si no es troben valors vàlids, es registra el cas i es retorna NaN
    else:
        #logging.info(f"No s'ha pogut interpolar {variable} per l'estació {station_id} al timestamp {current_timestamp}.")
        return np.nan

def preprocess_csv(file_path: str, root_directory: str) -> pd.DataFrame:
    """
    Preprocessa un fitxer CSV amb dades meteorològiques:
      - Llegeix el fitxer sense modificar-lo, provant amb 'utf-8' i, si falla, amb 'latin-1'.
      - Filtra les dades per mantenir només les fonts oficials (columna 'Font').
      - Extreu la data i l'hora del nom del fitxer per generar la columna 'Timestamp'.
      - Selecciona les columnes d'interès:
           * Per als anys 2016-2023: ['id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt',
             'VentDir', 'VentFor', 'Patm', 'lat', 'lon']
           * Per als fitxers de l'any 2015 (que poden no tenir 'Patm'): ['id', 'Font', 'Temp',
             'Humitat', 'Pluja', 'Alt', 'VentDir', 'VentFor', 'lat', 'lon']
      - Converteix les columnes numèriques.
      - Processa la variable "Pluja": calcula la pluja real per hora (diferència respecte l'hora anterior),
        considerant que si és la primera hora del dia, el valor previ és 0.
      - Imputa els valors nuls de les variables 'Temp', 'Humitat', 'VentDir', 'VentFor' i 'Patm' (si existeix)
        fent interpolació amb dades vàlides dels fitxers adjacents (dins d'un marge de 8 hores o, si no,
        a la mateixa hora del dia anterior/posterior). Si per alguna variable encara hi ha NaN, es descarta la fila.
      - Ajusta els intervals de 'Humitat' (0-100) i 'VentFor' (0-70).
    Retorna:
        pd.DataFrame: DataFrame processat o None en cas d'error crític.
    """

    # Llegir el fitxer i comprovar si existeix
    if file_path is None:
        return None
    if not os.path.exists(file_path):
        logging.error(f"El fitxer {file_path} no existeix.")
        return None
    # Intentem llegir el fitxer amb 'utf-8' i, si falla, amb 'latin-1'. Si falla de nou, retornem None.
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

    # Comprovar si el fitxer està buit o sense columnes. Si és així, es retorna un DataFrame buit.
    if data.empty or data.columns.size == 0:
        logging.warning(f"Fitxer {file_path} llegit sense columnes. Es retorna DataFrame buit.")
        return pd.DataFrame()

    # Filtrar per fonts fiables (columna 'Font')
    if 'Font' in data.columns:
        data = data[data['Font'].isin(FONTS_OFICIALS)]
        if data.empty:
            logging.warning(f"Fitxer {file_path} no conté fonts oficials després del filtratge.")
            return pd.DataFrame()
    # Si no hi ha la columna 'Font', es descarta el fitxer
    else:
        logging.warning(f"La columna 'Font' no està present a {file_path}.")
        return None

    # Extreure el Timestamp a partir del nom del fitxer
    base_filename = os.path.basename(file_path)
    match = re.match(r'(\d{4})(\d{2})(\d{2})(\d{2})dadesPC_utc\.csv', base_filename)
    # Primer comprovem si el nom del fitxer és vàlid
    if match:
        year_str, month, day, hour = match.groups()
        hour_int = int(hour)
        # Comprovem que l'hora sigui vàlida (0-23h)
        if hour_int >= 24:
            logging.info(f"Fitxer {file_path} té hora invàlida ({hour_int}). S'omet el processament.")
            return None
        # Creem el timestamp a partir de les dades extretes
        timestamp_str = f"{year_str}-{month}-{day} {hour}:00:00"
        # Convertim el timestamp a datetime
        try:
            ts = pd.to_datetime(timestamp_str, format='%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logging.error(f"Error convertint el Timestamp de {file_path}: {e}")
            return None
        # Afegim el timestamp al DataFrame
        data['Timestamp'] = ts.strftime('%Y-%m-%d %H:%M:%S')
    else:
        logging.error(f"No s'ha pogut extreure el Timestamp de {file_path}.")
        return None
    
    # Comprovem si l'any del fitxer és vàlid per al preprocessament
    year = ts.year

    # Comprovem si la columna 'Patm' està present si l'any no és 2015
    if year != 2015 and 'Patm' not in data.columns:
        logging.error(f"El fitxer {file_path} per a l'any {year} no conté la columna 'Patm' requerida.")
        return None

    # Definim l'ordre de columnes segons si tenim 'Patm' o no
    if 'Patm' in data.columns:
        column_order = ['id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt', 'VentDir', 'VentFor', 'Patm', 'lat', 'lon']
    else:
        column_order = ['id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt', 'VentDir', 'VentFor', 'lat', 'lon']

    # Seleccionem les columnes d'interès i reordenem
    data = data[[col for col in column_order if col in data.columns] + ['Timestamp']]

    # Converteix les columnes numèriques
    numeric_columns = ['Temp', 'Humitat', 'Alt', 'VentFor', 'lat', 'lon', 'Pluja']
    if 'Patm' in data.columns:
        numeric_columns.append('Patm')
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # La variable 'Pluja', si hi ha NaN, es considera 0
    data['Pluja'] = data['Pluja'].fillna(0)

    # Processar la variable "VentDir": mapar de direccions a graus
    if 'VentDir' in data.columns:
        mapping_direccions = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
            'Calma': np.nan, 'Variable': np.nan, 'Null': np.nan, '': np.nan
        }
        data['VentDir'] = data['VentDir'].map(mapping_direccions)

    # Eliminar files completament buides en les columnes clau (excepte 'Patm' si no existeix)
    imputer_columns = [col for col in numeric_columns if col in data.columns]
    files_antes = data.shape[0]
    data = data.dropna(subset=imputer_columns, how='all')
    if data.shape[0] < files_antes:
        logging.info(f"S'han eliminat {files_antes - data.shape[0]} files buides de {file_path}.")
    if data.empty:
        logging.warning(f"Després del preprocessament, {file_path} queda buit.")
        return data

    current_timestamp = ts

    # Processar la variable "Pluja": calcular la diferència respecte l'hora anterior per obtenir la pluja acumulada per hora
    pluja_values = []
    for idx, row in data.iterrows():
        station_id = row['id']
        current_pluja = row['Pluja'] if not pd.isna(row['Pluja']) else 0.0
        # Si és la primera hora del dia, no hi ha acumulat previ del mateix dia
        if current_timestamp.hour == 0:
            previous_pluja = 0.0
        else:
            previous_timestamp = current_timestamp - timedelta(hours=1)
            previous_pluja = get_station_value(root_directory, previous_timestamp, station_id, 'Pluja')
            if previous_pluja is None:
                previous_pluja = 0.0
        pluja_real = current_pluja - previous_pluja
        if pluja_real < 0:
            pluja_real = 0.0
        pluja_values.append(pluja_real)
    data['Pluja'] = pluja_values

    # Imputar les altres variables amb interpolació si hi ha NaN (excepte 'Pluja' i 'Alt')
    variables_interp = ['Temp', 'Humitat', 'VentDir', 'VentFor']
    if 'Patm' in data.columns:
        variables_interp.append('Patm')
    for idx, row in data.iterrows():
        station_id = row['id']
        for var in variables_interp:
            if pd.isna(row[var]):
                interpolated = interpolate_value(root_directory, current_timestamp, station_id, var)
                data.at[idx, var] = interpolated

    # Abans de retornar, eliminem les files on hi hagi algun NaN en les variables requerides
    n_antes = data.shape[0]
    data = data.dropna(subset=variables_interp, how='any')
    if data.shape[0] < n_antes:
        logging.info(f"S'han eliminat {n_antes - data.shape[0]} estacions per falta de dades interpolades.")

    # Ajustar els intervals de les variables segons especificacions: 'Humitat' (0-100 %), 'VentFor' (0-200 km/h) i 'VentDir' (0-360°)
    if 'Humitat' in data.columns:
        data['Humitat'] = data['Humitat'].clip(0, 100)
    if 'VentFor' in data.columns:
        data['VentFor'] = data['VentFor'].clip(0, 200)
    if 'VentDir' in data.columns:
        data['VentDir'] = data['VentDir'].clip(0, 360)

    return data

def process_all_csvs(root_directory: str, processed_directory: str):
    """
    Processa tots els fitxers CSV amb dades meteorològiques dels anys especificats a PROCESSED_YEARS
    (o tots si la llista està buida) del directori arrel:
      - Llegeix i preprocessa cada fitxer sense modificar les dades originals.
      - Desa els fitxers processats a processed_directory mantenint l'estructura original.
      - Mostra una barra de progrés a la terminal.
    """
    excluded_substrings = [
        "tauladades", "vextrems", "Admin_Estacions",
        "Clima", "Clima METEOCAT", "error_VAR", "html", "png", "var_vextrems", 
        "2013", "2014", "2015"
    ]

    # Funció per filtrar els directoris que contenen les subcadenes excloses
    def dir_valid(d):
        return not any(sub.lower() in d.lower() for sub in excluded_substrings) and not re.search(r'\d+_old', d)

    # Recórrer tots els fitxers del directori arrel amb os.walk i processar els que compleixin els criteris
    files_to_process = []
    for root_dir, dirs, files in os.walk(root_directory):
        print("Processant directori:", root_dir)
        dirs[:] = [d for d in dirs if dir_valid(d)]
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
    if len(files_to_process) == 0:
        print("No s'ha trobat cap fitxer que compleixi els criteris.")
        return

    # Processar els fitxers i desar-los al directori de sortida (barra de progrés amb tqdm)
    print("Iniciant processament...")
    for file_path in tqdm(files_to_process, desc="Processant fitxers", unit="fitxer"):
        processed_df = preprocess_csv(file_path, root_directory)
        if processed_df is not None and not processed_df.empty:
            relative_path = os.path.relpath(os.path.dirname(file_path), root_directory)
            output_dir = os.path.join(processed_directory, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, os.path.basename(file_path))
            try:
                processed_df.to_csv(output_file, index=False)
                logging.info(f"Fitxer processat desat a: {output_file}")
            except Exception as e:
                logging.error(f"Error desant el fitxer {output_file}: {e}")

if __name__ == "__main__":
    # Directoris d'entrada i sortida
    root_directory = 'F:/DADES_METEO_PC'
    processed_directory = 'F:/DADES_METEO_PC_PROCESSATS_IMPUTATS'
    
    # Processar els fitxers
    process_all_csvs(root_directory, processed_directory)
    logging.info("Processament finalitzat.")
    print("Processament finalitzat.")
