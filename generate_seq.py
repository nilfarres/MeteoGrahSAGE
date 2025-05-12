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
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

HORIZON_HOURS = 168  # Horitzó de predicció en hores (6h = 6, 12h = 12, 24h = 24, etc.)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera seqüències temporals diàries de grafs dinàmics."
    )
    parser.add_argument(
        '--input_dir', type=str, default='D:\DADES_METEO_PC_TO_DATA_v7_correcte',
        help='Directori amb snapshots horaris (*.pt) generats per toData_v7.py'
    )
    parser.add_argument(
        '--output_dir', type=str, default='D:\DADES_METEO_PC_generated_seqs_v6_ws168_str6_hh168',
        help='Directori on desar les seqüències generades'
    )
    parser.add_argument(
        '--window_size', type=int, default=168,
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

    # Processament per snapshot amb reindexat local
    for i, g in enumerate(data_seq):
        # Identificadors i característiques de nodes d'aquest gràfic
        node_ids = get_node_ids(g)
        N_t = len(node_ids)
        F = g.x.size(1)

        # Mapatge local id -> índex
        id2idx_t = {nid: idx for idx, nid in enumerate(node_ids)}

        # Matriu de característiques x_t: [N_t x F]
        x_t = torch.zeros((N_t, F), dtype=g.x.dtype)
        for local_idx, nid in enumerate(node_ids):
            x_t[local_idx] = g.x[local_idx]

        # Màscara de nodes presents
        mask_t = torch.ones((N_t,), dtype=torch.bool)

        # Reindexat d'arestes amb filtratge
        edges = g.edge_index.T.tolist()  # llista de [u_local, v_local]
        uvs = []
        for u, v in edges:
            # Assegura que els índexs locals existeixen
            if u < N_t and v < N_t and u >= 0 and v >= 0:
                uvs.append([u, v])
        ei_t = torch.tensor(uvs, dtype=torch.long).T  # [2 x E_t]

        # Validació estricta per rang
        if ei_t.numel() > 0:
            mn, mx = int(ei_t.min()), int(ei_t.max())
            assert mn >= 0 and mx < N_t, (
                f"Aresta fora de rang: mins={mn}, maxs={mx} vs N_t={N_t}"
            )

        # Atributs d'arestes (si existeixen)
        ea = None
        if hasattr(g, 'edge_attr') and g.edge_attr is not None:
            # Suposem que edge_attr s'alinea amb les arestes vàlides
            ea = g.edge_attr[:len(uvs), :] if g.edge_attr.dim() == 2 else None


        # === DEBUG DIMENSIONS ===
        # Prenem el timestamp d’aquest snapshot
        ts = os.path.splitext(os.path.basename(seq_files[i]))[0]
        # Comprovem nombre d’arestes vs. rows d’edge_attr
        n_edges = ei_t.size(1)
        n_attrs = ea.size(0) if ea is not None else None
        logging.info(f"DEBUG {ts}: edges={n_edges}, edge_attr_rows={n_attrs}")
        # Assert perquè pari si hi ha discrepància
        assert ea is None or n_edges == n_attrs, (
            f"MISMATCH {ts}: {n_edges} edges != {n_attrs} edge_attr rows"
        )
        # Comprovem també que els índexs siguin locals (< N_t)
        if n_edges > 0:
            max_idx = ei_t.max().item()
            assert max_idx < N_t, (
                f"OUT_OF_RANGE {ts}: max edge index {max_idx} >= num nodes {N_t}"
            )
        # ===========================


        # Afegim a les seqüències
        x_seq.append(x_t)
        mask_seq.append(mask_t)
        edge_index_seq.append(ei_t)
        edge_attr_seq.append(ea)

        # Etiqueta y_seq per a cada pas de la finestra
        ts_name = os.path.splitext(os.path.basename(seq_files[i]))[0]
        ts_h = datetime.strptime(ts_name, "%Y%m%d%H") + timedelta(hours=HORIZON_HOURS)
        fp_h = os.path.join(os.path.dirname(seq_files[0]),
                            ts_h.strftime("%Y%m%d%H") + ".pt")
        if not os.path.exists(fp_h):
            logging.warning(f"Seqüència {ts_name}: falta {fp_h}")
            y_seq.append(None)   # o bé fer `return` per descartar tota la seqüència
        else:
            g_h = torch.load(fp_h)
            y_seq.append(g_h.x)


        # Meta dades auxiliars
        timestamps.append(g.timestamp)
        fonts_seq.append(getattr(g, 'fonts', None))
        norm_params_seq.append(getattr(g, 'norm_params', None))
        meta_seq.append(getattr(g, 'meta', None))
        pos_seq.append(getattr(g, 'pos', None))
        year_seq.append(getattr(g, 'year', None))

    # 1. Creem la màscara de y_seq
    y_mask_seq = [y is not None for y in y_seq]
    # 2. (Opcional) descartem seqüències incompletes
    if not all(y_mask_seq):
        logging.warning(f"Seqüència {ts_start}_{ts_end}: etiqueta +6h faltant en algun pas, descartada.")
        return
    # 3. Substituïm els None per zeros
    y_seq = [y if y is not None else torch.zeros_like(x_seq[i])
            for i, y in enumerate(y_seq)]


    # Desa tot en un sol dict per seqüència
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
    logging.info(f"Generant {len(starts)} seqüències (window={args.window_size}, stride={args.stride}), horizon={HORIZON_HOURS}h")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_window, s, files, args.window_size, args.output_dir)
                   for s in starts]
        for _ in tqdm(as_completed(futures), total=len(futures), desc='Seqüències'):
            pass

    logging.info("Totes les seqüències s'han generat correctament.")


if __name__ == '__main__':
    main()
