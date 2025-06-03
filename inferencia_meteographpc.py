import numpy as np
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime, timedelta

# ------------------- CONFIGURACIÓ -------------------

# 1. Fitxers d'entrada (adapta noms si cal) :contentReference[oaicite:0]{index=0}
PRED_PATH   = 'C:/Users/nfarres/Documents/TFG/models/exec_prova4/y_pred_test.npy'
TRUE_PATH   = 'C:/Users/nfarres/Documents/TFG/models/exec_prova4/y_true_test.npy'
NODES_PATH  = 'nodes_metadata.csv'

# 2. Paràmetres de la generació de seqüències:
DATA_INICI_TEST = datetime(2024, 1, 1, 0, 0)   # PRIMER TIMESTAMP de la PRIMERA seqüència de test
STRIDE          = 12                          # hores (tal com vas entrenar)
WINDOW_SIZE     = 48                          # hores (tal com vas entrenar)
HORIZON         = 6                           # nombre d'horitzons
HORIZON_DELTA   = 1                           # hores: normalment 1 (canvia-ho si cada horitzó són 6h)

# 3. Variables meteorològiques (cal que coincideixin amb les que vas entrenar)
variables = [
    "Temp", "Humitat", "Pluja", "VentFor", "Patm", "Vent_u", "Vent_v"
]

# ------------------- CARREGA DE DADES -------------------

# Carrega les prediccions i valors reals (en “unitats internes”)  
y_pred = np.load(PRED_PATH)   # shape: [S, H, N, F]
y_true = np.load(TRUE_PATH)   # shape: [S, H, N, F]

# → NOMÉS LA PRIMERA SEQÜÈNCIA
y_pred = y_pred[:1]  # shape: [1, H, N, F]
y_true = y_true[:1]
S, H, N, F = y_pred.shape

# Talla la llista de variables al nombre real de variables
variables = variables[:F]

# Carrega la metadata de nodes  
nodes_df = pd.read_csv(NODES_PATH)
assert len(nodes_df) == N, "El nombre de nodes del fitxer i el nombre de nodes de la predicció no coincideixen!"

# ------------------- GENERACIÓ DE TIMESTAMPS -------------------

datetimes = []
for s in range(S):
    t_seq_start = DATA_INICI_TEST + timedelta(hours=s * STRIDE)
    for h in range(H):
        t_pred = t_seq_start + timedelta(hours=WINDOW_SIZE + h * HORIZON_DELTA)
        datetimes.append(t_pred)
datetimes = np.array(datetimes)
assert len(datetimes) == S * H, "Nombre de timestamps no concorda amb nombre de prediccions!"

# ------------------- REORGANITZACIÓ I MÀSCARES -------------------

# Reorganitza les dades a [T, N, F], amb T = S * H
y_pred_flat = y_pred.reshape(S * H, N, F)
y_true_flat = y_true.reshape(S * H, N, F)

# Màscara: si no tens màscara, tot vàlid (1)
mask = np.ones((S * H, N), dtype=np.int8)

# ------------------- CONVERSIÓ D'UNITATS -------------------

# 1) Identifica el column‐index de cada variable dins dels últims eixos
idx_temp  = variables.index("Temp")    # temperatura
idx_hum   = variables.index("Humitat") # humitat
idx_plu   = variables.index("Pluja")   # pluja
idx_wind  = variables.index("VentFor") # velocitat del vent
idx_patm  = variables.index("Patm")    # pressió atmosfèrica
idx_vu    = variables.index("Vent_u")  # component zonal del vent
idx_vv    = variables.index("Vent_v")  # component meridional del vent

# 2) Crea nous arrays per a les variables convertides a unitats físiques
y_pred_phys = np.empty_like(y_pred_flat)
y_true_phys = np.empty_like(y_true_flat)

# — Temperatura: de Kelvin a °C
#   Temp_internal = T [K]  →  T_°C = T[K] - 273.15  
y_pred_phys[..., idx_temp] = y_pred_flat[..., idx_temp] - 273.15
y_true_phys[..., idx_temp] = y_true_flat[..., idx_temp] - 273.15

# — Humitat: de fracció [0–1] a percentatge [%]
#   Hum_internal = H_frac  →  H_% = H_frac * 100
y_pred_phys[..., idx_hum] = y_pred_flat[..., idx_hum] * 100.0
y_true_phys[..., idx_hum] = y_true_flat[..., idx_hum] * 100.0

# — Pluja: de log1p(mm) a mm
#   Plu_internal = log(1 + mm)  →  mm = exp(Plu_internal) - 1  
y_pred_phys[..., idx_plu] = np.expm1(y_pred_flat[..., idx_plu])
y_true_phys[..., idx_plu] = np.expm1(y_true_flat[..., idx_plu])

# — VentFor: de m/s a km/h
#   Vent_internal [m/s]  → Vent_km/h = Vent_internal * 3.6  
y_pred_phys[..., idx_wind] = y_pred_flat[..., idx_wind] * 3.6
y_true_phys[..., idx_wind] = y_true_flat[..., idx_wind] * 3.6

# — Patm: de “anomalia en hPa” a hPa absolut  
#   Patm_internal = P - 1013 [hPa]  →  P_hPa = Patm_internal + 1013  
y_pred_phys[..., idx_patm] = y_pred_flat[..., idx_patm] + 1013.0
y_true_phys[..., idx_patm] = y_true_flat[..., idx_patm] + 1013.0

# — Vent_u i Vent_v: de m/s a km/h (components)
y_pred_phys[..., idx_vu] = y_pred_flat[..., idx_vu] * 3.6
y_true_phys[..., idx_vu] = y_true_flat[..., idx_vu] * 3.6
y_pred_phys[..., idx_vv] = y_pred_flat[..., idx_vv] * 3.6
y_true_phys[..., idx_vv] = y_true_flat[..., idx_vv] * 3.6

# 3) (Opcional) Si vols omplir amb NaN on la màscara sigui 0:
#    per exemple:
#    y_pred_phys[mask == 0] = np.nan
#    y_true_phys[mask == 0] = np.nan

# ------------------- CREACIÓ DEL NetCDF -------------------

OUTPUT_NC = 'C:/Users/nfarres/Documents/TFG/models/exec_prova4/predictions_meteographpc_test.nc'
ncfile = Dataset(OUTPUT_NC, mode='w', format='NETCDF4')

# Dimensions
ncfile.createDimension('time', S * H)
ncfile.createDimension('node', N)
ncfile.createDimension('variable', F)

# Variables de dimensió
times_nc = ncfile.createVariable('time', np.int32, ('time',))
nodes_nc = ncfile.createVariable('node', str, ('node',))
lats_nc  = ncfile.createVariable('lat', np.float32, ('node',))
lons_nc  = ncfile.createVariable('lon', np.float32, ('node',))
alts_nc  = ncfile.createVariable('alt', np.float32, ('node',))
vars_nc  = ncfile.createVariable('variable', str, ('variable',))

# Variables amb valors predits i reals (en unitats físiques)
pred_nc = ncfile.createVariable(
    'prediction', np.float32,
    ('time', 'node', 'variable'),
    fill_value=np.nan
)
true_nc = ncfile.createVariable(
    'groundtruth', np.float32,
    ('time', 'node', 'variable'),
    fill_value=np.nan
)
mask_nc = ncfile.createVariable('mask', np.int8, ('time', 'node'))  # 1: vàlid, 0: no vàlid

# Atributs “units” per a cada variable meteorològica
# (en l’ordre de la llista `variables`)
#   Temp: °C
#   Humitat: %
#   Pluja: mm
#   VentFor: km/h
#   Patm: hPa
#   Vent_u, Vent_v: km/h
unit_list = [
    "°C",   # Temp
    "%",    # Humitat
    "mm",   # Pluja
    "km/h", # VentFor
    "hPa",  # Patm
    "km/h", # Vent_u
    "km/h"  # Vent_v
]
for i, varname in enumerate(variables):
    pred_nc.setncattr('units', '')    # (no té sentit per tota la matriu; l’exportarem per variable)
    true_nc.setncattr('units', '')
# Un cop creades totes, afegim un array d’atributs “units”:
# En NetCDF4 no es pot posar directament un vector d’atributs; en comptes d’això, crearem un diccionari
# d’atributs dins de l’arxiu. Exemple:
for i, u in enumerate(unit_list):
    ncfile.setncattr(f'unit_{variables[i]}', u)

# Assignació de valors als arrays de dades
times_nc[:]        = np.arange(S * H)
nodes_nc[:]        = nodes_df['node_id'].astype(str).values
lats_nc[:]         = nodes_df['lat'].values
lons_nc[:]         = nodes_df['lon'].values
alts_nc[:]         = nodes_df['alt'].values
vars_nc[:]         = np.array(variables, dtype='str')

pred_nc[:, :, :]   = y_pred_phys
true_nc[:, :, :]   = y_true_phys
mask_nc[:, :]      = mask

# Variable extra amb timestamps ISO (string)
timestr_nc = ncfile.createVariable('time_str', str, ('time',))
time_strs = np.array([dt.strftime('%Y-%m-%dT%H:%MZ').encode() for dt in datetimes])
timestr_nc[:] = time_strs

# Atributs globals
ncfile.description = 'Prediccions meteorològiques test MeteoGraphPC (unitats físiques)'
ncfile.history     = 'Creat ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
ncfile.source      = 'Inferència model MeteoGraphPC amb unitats converides a °C, mm, km/h, %, etc.'

ncfile.close()
print(f'Fitxer NetCDF generat correctament amb unitats físiques: {OUTPUT_NC}')