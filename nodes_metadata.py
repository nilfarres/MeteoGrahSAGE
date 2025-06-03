import torch
import pandas as pd
import glob
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

def cartesian_to_geodetic(pos):
    """
    Converteix un array Nx3 de posicions cartesianes (x, y, z) a (lat, lon, alt).
    Les coordenades d'entrada han d'estar en km.
    La sortida és lat/lon en graus i altitud en metres.
    """
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    R = 6371.0  # km

    lon = np.arctan2(y, x)
    hyp = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, hyp)
    r = np.sqrt(x**2 + y**2 + z**2)
    alt = (r - R) * 1000  # passa a metres

    lat = np.degrees(lat)
    lon = np.degrees(lon)
    return lat, lon, alt

# Ruta als fitxers Data horaris
files = sorted(glob.glob('D:/DADES_METEO_PC_TO_DATA/*.pt'))

# Diccionari per anar acumulant la info: id -> (x, y, z)
node_dict = OrderedDict()

for f in tqdm(files, desc='Carregant fitxers'):
    try:
        data = torch.load(f, map_location='cpu')
        ids = data.ids
        pos = data.pos.numpy()
        for i, nid in enumerate(ids):
            if nid not in node_dict:
                node_dict[nid] = (pos[i, 0], pos[i, 1], pos[i, 2])
    except Exception as e:
        print(f'Error a {f}: {e}')

global_id_union = list(node_dict.keys())
xyz = np.array([node_dict[nid] for nid in global_id_union])  # shape [N, 3]

# Converteix de cartesianes a geodèsiques
lats, lons, alts = cartesian_to_geodetic(xyz)

df = pd.DataFrame({
    'node_id': global_id_union,
    'lat': lats,
    'lon': lons,
    'alt': alts
})

df.to_csv('nodes_metadata.csv', index=False)
print('Fitxer nodes_metadata.csv creat amb coordenades geogràfiques correctes!')
