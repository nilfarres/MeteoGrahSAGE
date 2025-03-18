import os, glob                           # Per gestionar directoris i cercar fitxers dins d'una estructura de carpetes
import pandas as pd                       # Per llegir i manipular els fitxers CSV
import torch                              # Per treballar amb tensors i desar els objectes Data convertits a fitxers .pt
from torch_geometric.data import Data     # Per encapsular la informació del graf (nodes, arestes, característiques)
from torch_geometric.nn import knn_graph  # Per generar les connexions entre nodes mitjançant l'algorisme kNN (proximitat espacial)

# Directoris d'entrada (fitxers preprocessats) i de sortida (objectes Data)
input_root = "D:/DADES_METEO_PC_PREPROCESSADES_GPU_PARALLEL"
output_root = "D:/DADES_METEO_PC_TO_DATA"

# Columnes necessàries (inclou Alt per a la posició, però no per les característiques)
required_columns = ['id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt', 'VentDir', 'VentFor', 'Patm', 'lat', 'lon']
official_sources = ["Aemet", "METEOCAT", "METEOCAT_WEB", "Meteoclimatic", "Vallsdaneu",
                   "SAIH", "avamet", "Meteoprades", "MeteoPirineus", "WLINK_DAVIS"]

# Cerca tots els fitxers que acaben amb "dadesPC_utc.csv" replicant l'estructura any/mes/dia/hora
pattern = os.path.join(input_root, "*", "*", "*", "*", "*dadesPC_utc.csv")
file_list = glob.glob(pattern)
print(f"Nombre de fitxers trobats: {len(file_list)}")

# Processa cada fitxer
for file_path in file_list:
    try:
        # Llegir el fitxer CSV
        df = pd.read_csv(file_path)
        
        # Seleccionar només les columnes necessàries
        df = df[required_columns]
        
        # Filtrar per mantenir només les fonts oficials
        df = df[df['Font'].isin(official_sources)]
        
        # Si el dataframe queda buit, s'omet aquest fitxer
        if df.empty:
            print(f"El fitxer {file_path} no conté dades oficials. S'omet.")
            continue
        
        # Cada fila representa una estació meteorològica.
        # Característiques del node: Temp, Humitat, Pluja, VentDir, VentFor, Patm (sense Alt)
        feature_cols = ['Temp', 'Humitat', 'Pluja', 'VentDir', 'VentFor', 'Patm']
        x = torch.tensor(df[feature_cols].values, dtype=torch.float)
        
        # La posició es defineix amb [lat, lon, Alt]
        pos = torch.tensor(df[['lat', 'lon', 'Alt']].values, dtype=torch.float)
        
        num_nodes = x.size(0)
        
        # Comprovar que hi hagi suficients nodes per connectar-ne almenys 1
        if num_nodes < 2:
            print(f"El fitxer {file_path} té menys d'una connexió possible (num_nodes={num_nodes}). S'omet.")
            continue
        
        # Adaptar el valor de k segons el nombre d'estacions: 
        # Es defineix com el mínim entre 5 i (num_nodes - 1) perquè un node no es pot connectar a si mateix.
        k_actual = min(5, num_nodes - 1)
        
        # Generar les connexions (arestes) amb kNN basant-se en la posició en 3D
        edge_index = knn_graph(pos, k=k_actual, loop=False)
        
        # Crear l'objecte Data de torch_geometric amb x, pos i edge_index
        data = Data(x=x, pos=pos, edge_index=edge_index)
        
        # Afegir informació addicional (opcional) per conservar els identificadors i fonts originals
        data.ids = list(df['id'])
        data.fonts = list(df['Font'])
        
        # Replicar l'estructura original per desar el fitxer convertit: any/mes/dia/hora
        rel_path = os.path.relpath(file_path, input_root)
        # Canviar l'extensió a .pt (format de PyTorch)
        rel_path_pt = rel_path.replace("dadesPC_utc.csv", "pt")
        output_file = os.path.join(output_root, rel_path_pt)
        
        # Crear els directoris necessaris si no existeixen
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Desa l'objecte Data
        torch.save(data, output_file)
        
        print(f"Conversió correcta: {file_path} -> {output_file}")
        
    except Exception as e:
        print(f"Error en processar el fitxer {file_path}: {e}")
