#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_seq.py

Versió corregida perquè s'adapti perfectament als Data objects generats per toData.py.
Genera seqüències temporals diàries de graf dinàmics, preservant tots els atributs i metadades.
"""

import os
import glob
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera seqüències temporals diàries de grafs dinàmics."
    )
    parser.add_argument(
        '--input_dir', type=str, default='D:\DADES_METEO_PC_TO_DATA',
        help='Directori amb snapshots horaris (*.pt) generats per toData.py'
    )
    parser.add_argument(
        '--output_dir', type=str, default='D:\DADES_METEO_PC_generated_seqs_ws48_str24',
        help='Directori on desar les seqüències generades'
    )
    parser.add_argument(
        '--window_size', type=int, default=24,
        help="Nombre d'hores per seqüència (per defecte: 24)"
    )
    parser.add_argument(
        '--stride', type=int, default=24,
        help='Salt en hores entre inici de seqüències (per defecte: 24)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Nombre de processos per a processament paral·lel'
    )
    return parser.parse_args()


def get_node_ids(data):
    """
    Retorna la llista d'identificadors de nodes per a cada snapshot Data.
    S'espera atribut 'ids' o 'node_id'.
    """
    if hasattr(data, 'ids'):
        return list(data.ids)
    if hasattr(data, 'node_id'):
        return list(data.node_id)
    raise AttributeError("Cada Data ha de tenir 'ids' o 'node_id'.")


def reindex_edges(edge_index, id_list, id2idx):
    """
    Reindexa un tensor edge_index ([2, E]) de índexs locals a globals segons id2idx.
    """
    src, dst = edge_index
    src_global = torch.tensor([id2idx[id_list[i]] for i in src.tolist()], dtype=torch.long)
    dst_global = torch.tensor([id2idx[id_list[i]] for i in dst.tolist()], dtype=torch.long)
    return torch.stack([src_global, dst_global], dim=0)


def process_window(start_idx, files, window_size, output_dir):
    seq_files = files[start_idx:start_idx + window_size]
    if len(seq_files) < window_size:
        return

    # Ruta i nom de sortida
    ts_start = os.path.splitext(os.path.basename(seq_files[0]))[0]
    ts_end   = os.path.splitext(os.path.basename(seq_files[-1]))[0]
    out_name = f"{ts_start}_{ts_end}.pt"
    out_path = os.path.join(output_dir, out_name)
    if os.path.exists(out_path):
        logging.debug(f"Ja existeix {out_name}, saltant.")
        return

    # Carrega tots els Data objects
    data_seq = [torch.load(fp) for fp in seq_files]

    # Conjunt global de node IDs i mapping a índex global
    all_ids = sorted({nid for g in data_seq for nid in get_node_ids(g)})
    id2idx = {nid: idx for idx, nid in enumerate(all_ids)}
    N = len(all_ids)
    F = data_seq[0].x.size(1)

    # Seqüències a omplir
    x_seq = []
    mask_seq = []
    edge_index_seq = []
    edge_attr_seq = []
    y_seq = []
    timestamps = []
    fonts_seq = []
    norm_params_seq = []
    meta_seq = []
    pos_seq = []
    year_seq = []

    for g in data_seq:
        node_ids = get_node_ids(g)
        # Carrega 
        x_t = torch.zeros((N, F), dtype=g.x.dtype)
        mask_t = torch.zeros((N,), dtype=torch.bool)
        for local_idx, nid in enumerate(node_ids):
            idx = id2idx[nid]
            x_t[idx] = g.x[local_idx]
            mask_t[idx] = True

        ei = reindex_edges(g.edge_index, node_ids, id2idx)
        ea = g.edge_attr if hasattr(g, 'edge_attr') else None
        y  = g.y          if hasattr(g, 'y')         else None

        x_seq.append(x_t)
        mask_seq.append(mask_t)
        edge_index_seq.append(ei)
        edge_attr_seq.append(ea)
        y_seq.append(y)
        timestamps.append(g.timestamp)
        fonts_seq.append(g.fonts          if hasattr(g,'fonts')      else None)
        norm_params_seq.append(g.norm_params if hasattr(g,'norm_params') else None)
        meta_seq.append(g.meta            if hasattr(g,'meta')       else None)
        pos_seq.append(g.pos              if hasattr(g,'pos')        else None)
        year_seq.append(g.year            if hasattr(g,'year')       else None)

    # Desa tot en un sol dict per seqüència
    torch.save({
        'all_node_ids': all_ids,
        'x_seq': x_seq,
        'mask_seq': mask_seq,
        'edge_index_seq': edge_index_seq,
        'edge_attr_seq': edge_attr_seq,
        'y_seq': y_seq,
        'timestamps': timestamps,
        'fonts_seq': fonts_seq,
        'norm_params_seq': norm_params_seq,
        'meta_seq': meta_seq,
        'pos_seq': pos_seq,
        'year_seq': year_seq
    }, out_path)
    logging.info(f"Desat seqüència {out_name}")


def main():
    args = parse_args()
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Llista i ordena fitxers .pt per nom de fitxer
    files = sorted(
        glob.glob(os.path.join(args.input_dir, '*.pt')),
        key=lambda p: os.path.basename(p)
    )
    total = len(files)
    if total < args.window_size:
        logging.error(f"Pocs snapshots ({total}) per a window_size={args.window_size}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    starts = list(range(0, total - args.window_size + 1, args.stride))
    logging.info(f"Generant {len(starts)} seqüències (window={args.window_size}, stride={args.stride})...")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_window, s, files, args.window_size, args.output_dir)
                   for s in starts]
        for _ in tqdm(as_completed(futures), total=len(futures), desc='Seqüències'):
            pass

    logging.info("Totes les seqüències s'han generat correctament.")


if __name__ == '__main__':
    main()
