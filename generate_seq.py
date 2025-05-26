#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
generate_seq.py

Script per generar seqüències temporals de grafs meteorològics dinàmics.

Aquest script processa una carpeta de snapshots horaris en format ".pt" 
(generats per "toData.py") i crea seqüències temporals preparades per entrenar 
models de Graph Neural Networks amb PyTorch Geometric Temporal.

FUNCIONALITATS PRINCIPALS:
  - Llegeix automàticament tots els fitxers acabats en ".pt" del directori DADES_METEO_PC_TO_DATA i ordena cronològicament.
  - Genera seqüències temporals de finestra lliscant (sliding window) amb una longitud i un stride configurables.
  - Genera les etiquetes futures (targets) amb un horitzó de predicció configurable.
  - Remapeja globalment tots els nodes a la mateixa unió d'IDs per facilitar el processament per batch.
  - Desa cada seqüència com a fitxer ".pt" a la carpeta de sortida, amb totes les dades i màscares necessàries.
  - Permet processament paral·lel per accelerar la generació de seqüències.

INSTRUCCIONS D'ÚS:
  1. Important: cal haver executa prèviament "prep.py", "compute_PC_norm_params" i "toData.py" en aquest ordre.
  2. Configura els arguments de la línia de comandes:
      --input_dir    (directori amb els grafs horaris ".pt" d'entrada, per defecte DADES_METEO_PC_TO_DATA).
      --output_dir   (directori de sortida de les seqüències temporals generades).
      --window_size  (llargada de la seqüència, en hores).
      --stride       (interval entre seqüències, en hores).
      --num_workers  (processos en paral·lel, opcional).
  3. Modifica el valor de la variable HORIZON_HOURS a l'inici de l'script per definir l'horitzó de predicció de cada seqüència (per defecte 6 hores).
  3. Executa l'script i trobaràs les seqüències generades a la carpeta de sortida.

REQUISITS:
  - Python 3.x
  - Llibreries: torch, tqdm, argparse, glob

AUTOR: Nil Farrés Soler
==============================================================================
"""

import os
import glob
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

HORIZON_HOURS = 6  # Horitzó de predicció en hores (6h = 6, 12h = 12, 24h = 24, etc.)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera seqüències temporals diàries de graf dinàmics."
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Directori amb snapshots horaris (*.pt) generats per toData.py'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directori on desar les seqüències generades'
    )
    parser.add_argument(
        '--window_size', type=int, default=48,
        help="Nombre d'hores per seqüència (per defecte: 48)"
    )
    parser.add_argument(
        '--stride', type=int, default=6,
        help='Salt en hores entre inici de seqüències (per defecte: 6)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Nombre de processos per a processament paral·lel'
    )
    return parser.parse_args()

def _init_globals(id_union, id2idx, Nu):
    global global_id_union, id2idx_global, N_u_global
    global_id_union = id_union
    id2idx_global   = id2idx
    N_u_global      = Nu

def get_node_ids(data):
    """
    Retorna la llista d'identificadors de nodes per a cada snapshot Data.
    S'espera atribut 'ids' o 'node_id'.
    """
    for attr in ('ids', 'node_id', 'node_ids'):
        if hasattr(data, attr):
            return list(getattr(data, attr))
    raise AttributeError("Cada Data ha de tenir 'ids' o 'node_id'.")


def process_window(start_idx, files, window_size, output_dir):
    seq_files = files[start_idx:start_idx + window_size]
    y_files = files[start_idx + window_size : start_idx + window_size + HORIZON_HOURS]

    if len(seq_files) < window_size or len(y_files) < HORIZON_HOURS:
        return

    # Nom de l'arxiu de sortida
    ts_start = os.path.splitext(os.path.basename(seq_files[0]))[0]
    ts_end   = os.path.splitext(os.path.basename(seq_files[-1]))[0]
    out_name = f"{ts_start}_{ts_end}.pt"
    out_path = os.path.join(output_dir, out_name)
    if os.path.exists(out_path):
        return

    # Carrega tots els Data objects
    data_seq = [torch.load(fp, map_location='cpu') for fp in seq_files]

    # Seqüències a omplir
    x_seq = []
    mask_seq = []
    edge_index_seq = []
    edge_attr_seq = []
    id_seq = []  # ara recollim només IDs presents per timestamp
    y_seq = []
    y_mask_seq = []
    timestamps = []
    fonts_seq = []
    norm_params_seq = []
    meta_seq = []
    pos_seq = []
    year_seq = []

    # B) Construir seqüència d'entrada remapejada globalment
    for idx, data in enumerate(data_seq):
        # Reindexar features i màscara
        F = data.x.size(1)
        x_t    = torch.zeros((N_u_global, F), dtype=data.x.dtype)
        mask_t = torch.zeros(N_u_global, dtype=torch.bool)
        local_ids = get_node_ids(data)
        for local_i, nid in enumerate(local_ids):
            g_i = id2idx_global[nid]
            x_t[g_i]    = data.x[local_i]
            mask_t[g_i] = True
        x_seq.append(x_t)
        mask_seq.append(mask_t)

        # Guardem només els IDs presents en aquest pas
        id_seq.append(local_ids)

        # Reindexar topologia
        src, dst = data.edge_index
        src, dst = src.tolist(), dst.tolist()
        src_g = [id2idx_global[local_ids[i]] for i in src]
        dst_g = [id2idx_global[local_ids[i]] for i in dst]
        edge_index_seq.append(torch.tensor([src_g, dst_g], dtype=torch.long))
        edge_attr_seq.append(data.edge_attr)

        # Capturar timestamps i metadades
        ts_str = os.path.splitext(os.path.basename(seq_files[idx]))[0]
        ts_dt  = datetime.strptime(ts_str, "%Y%m%d%H")
        timestamps.append(ts_dt)
        fonts_seq.append(getattr(data, 'fonts',      None))
        norm_params_seq.append(getattr(data, 'norm_params', None))
        meta_seq.append(getattr(data, 'meta',       None))
        pos_seq.append(getattr(data, 'pos',        None))
        year_seq.append(ts_dt.year)

    # C) Construir seqüència d'etiquetes futures
    for fp_h in y_files:
        g_h = torch.load(fp_h, map_location='cpu')
        Fh = g_h.x.size(1)
        assert Fh == x_seq[0].size(1), "Les dimensions de feature no coincideixen"
        y_t    = torch.zeros((N_u_global, Fh), dtype=g_h.x.dtype)
        y_mask = torch.zeros(N_u_global, dtype=torch.bool)
        local_ids = get_node_ids(g_h)
        for local_i, nid in enumerate(local_ids):
            g_i = id2idx_global[nid]
            y_t[g_i]    = g_h.x[local_i]
            y_mask[g_i] = True
        y_seq.append(y_t)
        y_mask_seq.append(y_mask)

    # Desa tot
    torch.save({
        'x_seq': x_seq,
        'mask_seq': mask_seq,
        'edge_index_seq': edge_index_seq,
        'edge_attr_seq': edge_attr_seq,
        'y_seq': y_seq,
        'y_mask_seq': y_mask_seq,
        'timestamps': timestamps,
        'fonts_seq': fonts_seq,
        'norm_params_seq': norm_params_seq,
        'meta_seq': meta_seq,
        'pos_seq': pos_seq,
        'year_seq': year_seq,
        'id_seq': id_seq,
    }, out_path)
    logging.info(f"Desat seqüència {out_name}")


def main():
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        logging.error(f"input_dir «{args.input_dir}» no existeix o no és un directori.")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    # Llista fitxers ordenats
    files = sorted(glob.glob(os.path.join(args.input_dir, '*.pt')),
                   key=lambda p: os.path.basename(p))
    total = len(files)
    if total < args.window_size:
        logging.error(f"Pocs snapshots ({total}) per a window_size={args.window_size}")
        return

    # ID union global
    all_ids = set()
    for fp in tqdm(files, desc='Calculant ID union global'):
        g = torch.load(fp, map_location='cpu')
        all_ids.update(get_node_ids(g))
    global_id_union = sorted(all_ids)
    id2idx_global = {nid: i for i, nid in enumerate(global_id_union)}
    N_u_global = len(global_id_union)
    logging.info(f"ID union global: {N_u_global} nodes")

    # Seqüències
    starts = list(range(0, total - args.window_size - HORIZON_HOURS + 1, args.stride))
    logging.info(f"Generant {len(starts)} seqüències (window={args.window_size}, stride={args.stride}), horizon={HORIZON_HOURS}h")
    with ProcessPoolExecutor(max_workers=args.num_workers,
                             initializer=_init_globals,
                             initargs=(global_id_union, id2idx_global, N_u_global)) as executor:
        futures = [executor.submit(process_window, s, files, args.window_size, args.output_dir)
                   for s in starts]
        for future in tqdm(as_completed(futures), total=len(futures), desc='Seqüències'):
            future.result()

    logging.info("Totes les seqüències s'han generat correctament.")

if __name__ == '__main__':
    main()
