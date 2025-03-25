#!/usr/bin/env python3
"""
Script per transformar fitxers CSV preprocessats en objectes Data de torch_geometric,
aprofitant la GPU per als càlculs i paral·lelitzant el processament per reduir el temps d'execució.

Millores implementades:
  - Normalització opcional de les característiques dels nodes.
  - Elecció dinàmica del mètode per generar el graf:
      * Si num_nodes < 50, s'utilitza knn_graph amb k = min(5, num_nodes - 1).
      * Si num_nodes >= 50, s'utilitza radius_graph amb radi basat en la quantil 75.
  - Afegiment d'"edge_attr" amb:
      • La distància euclidiana entre nodes.
      • La diferència absoluta en Temp.
      • La diferència absoluta en Humitat.
      • La diferència absoluta en VentFor.
  - Inclusió de l'altitud ("Alt") com a característica addicional dels nodes.
  - Processament paral·lel amb ProcessPoolExecutor.
"""

import os, glob
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Directoris d'entrada i de sortida
input_root = "D:/DADES_METEO_PC_PREPROCESSADES_GPU_PARALLEL"
output_root = "D:/DADES_METEO_PC_TO_DATA"

# Columnes necessàries i fonts oficials
required_columns = ['id', 'Font', 'Temp', 'Humitat', 'Pluja', 'Alt', 'VentDir', 'VentFor', 'Patm', 'lat', 'lon']
official_sources = ["Aemet", "METEOCAT", "METEOCAT_WEB", "Meteoclimatic", "Vallsdaneu",
                    "SAIH", "avamet", "Meteoprades", "MeteoPirineus", "WLINK_DAVIS"]

# Configurar logging
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"toData_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename)]
)

def normalize_features(x: torch.Tensor) -> torch.Tensor:
    """
    Normalitza les característiques per a que tinguin mitjana 0 i desviació estàndard 1.
    """
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    std[std == 0] = 1  # Evita divisió per zero
    return (x - mean) / std

def process_file(file_path: str):
    try:
        # Llegir el fitxer CSV preprocessat
        df = pd.read_csv(file_path)
        
        # Seleccionar només les columnes necessàries
        df = df[required_columns]
        
        # Filtrar per fonts oficials
        df = df[df['Font'].isin(official_sources)]
        if df.empty:
            logging.info(f"El fitxer {file_path} no conté dades oficials. S'omet.")
            return
        
        # Opcional: Normalització de les característiques
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[['Temp', 'Humitat', 'Pluja', 'VentDir', 'VentFor', 'Patm', 'Alt']] = scaler.fit_transform(df[['Temp', 'Humitat', 'Pluja', 'VentDir', 'VentFor', 'Patm', 'Alt']])
        
        # Convertir les característiques dels nodes a tensor a la GPU.
        # Ara s'inclou "Alt" com a característica addicional.
        feature_cols = ['Temp', 'Humitat', 'Pluja', 'VentDir', 'VentFor', 'Patm', 'Alt']
        x = torch.tensor(df[feature_cols].values, dtype=torch.float, device='cuda')
        # Normalitzar (usant la funció definida)
        x = normalize_features(x)
        
        # La posició es defineix amb [lat, lon, Alt]
        pos = torch.tensor(df[['lat', 'lon', 'Alt']].values, dtype=torch.float, device='cuda')
        
        num_nodes = x.size(0)
        if num_nodes < 2:
            logging.info(f"El fitxer {file_path} té menys d'una connexió possible (num_nodes={num_nodes}). S'omet.")
            return
        
        # Elecció dinàmica del mètode per generar el graf
        if num_nodes < 50:
            k_actual = min(5, num_nodes - 1)
            edge_index = knn_graph(pos, k=k_actual, loop=False)
        else:
            sample_size = min(50, num_nodes)
            indices = torch.randperm(num_nodes)[:sample_size]
            sample_pos = pos[indices]
            dists = torch.cdist(sample_pos, sample_pos)
            dists_flat = dists[dists > 0]
            radius = torch.quantile(dists_flat, 0.75).item()
            edge_index = radius_graph(pos, r=radius, loop=False)
        
        # Calcular edge attributes:
        # 1. Distància euclidiana entre nodes (de posicions: [lat, lon, Alt])
        # 2. Diferència absoluta en Temp (index 0 de x)
        # 3. Diferència absoluta en Humitat (index 1 de x)
        # 4. Diferència absoluta en VentFor (index 4 de x)
        src, dst = edge_index
        edge_attr_dist = torch.norm(pos[src] - pos[dst], dim=1, keepdim=True)
        diff_temp = torch.abs(x[src, 0] - x[dst, 0]).unsqueeze(1)
        diff_humitat = torch.abs(x[src, 1] - x[dst, 1]).unsqueeze(1)
        diff_ventFor = torch.abs(x[src, 4] - x[dst, 4]).unsqueeze(1)
        edge_attr = torch.cat([edge_attr_dist, diff_temp, diff_humitat, diff_ventFor], dim=1)
        
        # Traslladar tensors a CPU per desar l'objecte Data
        x = x.cpu()
        pos = pos.cpu()
        edge_index = edge_index.cpu()
        edge_attr = edge_attr.cpu()
        
        # Crear l'objecte Data de torch_geometric
        data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
        data.ids = list(df['id'])
        data.fonts = list(df['Font'])
        if 'Timestamp' in df.columns:
            data.timestamp = df['Timestamp'].iloc[0]
        
        # Replicar l'estructura original per desar el fitxer convertit
        rel_path = os.path.relpath(file_path, input_root)
        rel_path_pt = rel_path.replace("dadesPC_utc.csv", "pt")
        output_file = os.path.join(output_root, rel_path_pt)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        torch.save(data, output_file)
        logging.info(f"Conversió correcta: {file_path} -> {output_file}")
    except Exception as e:
        logging.error(f"Error en processar {file_path}: {e}")

def main():
    pattern = os.path.join(input_root, "*", "*", "*", "*", "*dadesPC_utc.csv")
    file_list = glob.glob(pattern)
    logging.info(f"Nombre de fitxers a processar: {len(file_list)}")
    if not file_list:
        print("No s'ha trobat cap fitxer.")
        return

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
