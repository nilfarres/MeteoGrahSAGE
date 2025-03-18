#!/usr/bin/env python3
"""
Script per a transformar els fitxers CSV preprocessats en objectes Data de torch_geometric,
aprofitant la GPU per als càlculs i paral·lelitzant el processament per accelerar l'execució.

Aquest codi fa el següent:
  - Llegeix els fitxers CSV preprocessats.
  - Selecciona les columnes necessàries i filtra per fonts fiables.
  - Converteix les dades a tensors: les característiques (x) i la posició (pos).
  - Genera les connexions entre nodes amb l'algorisme kNN.
  - Crea l'objecte Data de torch_geometric i afegeix informació addicional.
  - Desa els objectes Data mantenint l'estructura original.
  
Els càlculs pesats (conversió a tensors i knn_graph) s'executen a la GPU.
El processament es paral·lelitza amb ProcessPoolExecutor.
"""

import os, glob
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Directoris d'entrada (fitxers preprocessats) i de sortida (objectes Data)
input_root = "D:/DADES_METEO_PC_PREPROCESSADES_GPU_PARALLEL"
output_root = "D:/DADES_METEO_PC_TO_DATA"

# Columnes necessàries i fonts fiables
required_columns = ['id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt', 'VentDir', 'VentFor', 'Patm', 'lat', 'lon']
official_sources = ["Aemet", "METEOCAT", "METEOCAT_WEB", "Meteoclimatic", "Vallsdaneu",
                    "SAIH", "avamet", "Meteoprades", "MeteoPirineus", "WLINK_DAVIS"]

# Configurar el logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join("logs", f"toData_{torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'cpu'}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"))]
)

def process_file(file_path: str):
    try:
        # Llegir el fitxer CSV preprocessat
        df = pd.read_csv(file_path)
        
        # Seleccionar només les columnes necessàries
        df = df[required_columns]
        
        # Filtrar per mantenir només les fonts fiables
        df = df[df['Font'].isin(official_sources)]
        if df.empty:
            logging.info(f"El fitxer {file_path} no conté dades oficials. S'omet.")
            return
        
        # Convertir les característiques dels nodes a tensor (es treballa a GPU per accelerar)
        feature_cols = ['Temp', 'Humitat', 'Pluja', 'VentDir', 'VentFor', 'Patm']
        # Convertir a tensor en dispositiu CUDA
        x = torch.tensor(df[feature_cols].values, dtype=torch.float, device='cuda')
        
        # La posició es defineix amb [lat, lon, Alt]
        pos = torch.tensor(df[['lat', 'lon', 'Alt']].values, dtype=torch.float, device='cuda')
        
        num_nodes = x.size(0)
        if num_nodes < 2:
            logging.info(f"El fitxer {file_path} té menys d'una connexió possible (num_nodes={num_nodes}). S'omet.")
            return
        
        # Definir k com a mínim entre 5 i num_nodes - 1
        k_actual = min(5, num_nodes - 1)
        
        # Generar les connexions amb knn_graph utilitzant posicions en GPU
        edge_index = knn_graph(pos, k=k_actual, loop=False)
        
        # Convertir els tensors a CPU per a la desada (per compatibilitat)
        x = x.cpu()
        pos = pos.cpu()
        edge_index = edge_index.cpu()
        
        # Crear l'objecte Data de torch_geometric
        data = Data(x=x, pos=pos, edge_index=edge_index)
        data.ids = list(df['id'])
        data.fonts = list(df['Font'])
        
        # Replicar l'estructura original per desar el fitxer convertit
        rel_path = os.path.relpath(file_path, input_root)
        rel_path_pt = rel_path.replace("dadesPC_utc.csv", "pt")
        output_file = os.path.join(output_root, rel_path_pt)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Desa l'objecte Data
        torch.save(data, output_file)
        logging.info(f"Conversió correcta: {file_path} -> {output_file}")
    except Exception as e:
        logging.error(f"Error en processar {file_path}: {e}")

def main():
    # Cercar tots els fitxers amb estructura any/mes/dia/hora que acaben en "dadesPC_utc.csv"
    pattern = os.path.join(input_root, "*", "*", "*", "*", "*dadesPC_utc.csv")
    file_list = glob.glob(pattern)
    logging.info(f"Nombre de fitxers trobats: {len(file_list)}")
    if not file_list:
        print("No s'ha trobat cap fitxer.")
        return

    # Processament paral·lel amb ProcessPoolExecutor
    max_workers = 8  # Aquest valor es pot ajustar segons els nuclis disponibles
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in file_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processant fitxers", unit="fitxer"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processant {futures[future]}: {e}")

    logging.info("Processament finalitzat.")
    print("Processament finalitzat.")

if __name__ == "__main__":
    main()
