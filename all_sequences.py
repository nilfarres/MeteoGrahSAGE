#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
all_sequences.py

Script per agrupar seqüències temporals de grafs meteorològics en chunks.

Aquest script llegeix tots els fitxers de seqüència ".pt" generats prèviament 
(amb "generate_seq.py"), els agrupa en lots (chunks) de mida configurable 
(i opcionalment en guarda les metadades) per optimitzar el processament en 
models de Machine Learning o per a transferència massiva de dades.

FUNCIONALITATS PRINCIPALS:
  - Busca i ordena automàticament tots els fitxers de seqüència ".pt" del directori especificat.
  - Agrupa les seqüències en fitxers més grans, cadascun amb un nombre determinat de seqüències (chunk).
  - Desa, per a cada chunk, un fitxer amb totes les seqüències i un altre amb només les metadades (noms de fitxer).
  - Informa per pantalla del progrés i de qualsevol error de lectura de fitxers.

INSTRUCCIONS D'ÚS:
  1. Modifica les rutes "SEQ_DIR" i "OUTPUT_BASE" per indicar els directoris d'entrada i de sortida.
  2. Configura la mida del chunk amb la variable "CHUNK_SIZE" a l'inici de l'script si cal (per defecte genera grups de 50 seqüències).
  3. Executa l'script. Es generaran fitxers anomenats "chunk_XXX.pt" i "chunk_XXX_meta.pt" al directori de sortida.
  4. Revisa els missatges d'error per si hi ha fitxers que no s'han pogut carregar correctament.

REQUISITS:
  - Python 3.x
  - Llibreries: torch, tqdm, glob, os

AUTOR: Nil Farrés Soler
==============================================================================
"""

import torch
import glob
import os
from tqdm import tqdm

SEQ_DIR = r'F:\ws48_str12_hh6\DADES_METEO_PC_generated_seqs_ws48_str12_hh6'
OUTPUT_BASE = r'F:\ws48_str12_hh6\All_Sequences_ws48_str12_hh6_chunksde50'

CHUNK_SIZE = 50  # Nombre de seqüències per fitxer agrupat -> CAL MODIFICAR AQUEST PARAMETRE A AQUI!

# 1. Troba i ordena tots els fitxers de seqüència
fitxers = sorted(glob.glob(os.path.join(SEQ_DIR, "*.pt")))

assert len(fitxers) > 0, f"No s'han trobat fitxers de seqüències a {SEQ_DIR}"

print(f"S'han trobat {len(fitxers)} fitxers de seqüències. Iniciant el procés d'agrupació per chunks de {CHUNK_SIZE}...")

# 2. Divideix en chunks i desa cada un
for i in range(0, len(fitxers), CHUNK_SIZE):
    chunk_files = fitxers[i:i + CHUNK_SIZE]
    all_sequences = []
    errors = []
    for f in tqdm(chunk_files, desc=f"Carregant chunk {i // CHUNK_SIZE + 1}"):
        try:
            seq = torch.load(f, map_location="cpu")
            all_sequences.append(seq)
        except Exception as e:
            print(f"Error carregant {f}: {e}")
            errors.append(f)

    # Fitxer gran: només les seqüències!
    seq_output = os.path.join(OUTPUT_BASE, f"chunk_{i // CHUNK_SIZE + 1:03d}.pt")
    torch.save({"sequences": all_sequences}, seq_output)

    # Fitxer petit: només les metadades (basenames)
    meta_output = os.path.join(OUTPUT_BASE, f"chunk_{i // CHUNK_SIZE + 1:03d}_meta.pt")
    basenames = [os.path.basename(f) for f in chunk_files]
    torch.save({"filenames": basenames}, meta_output)

    print(f"Guardat {seq_output} amb {len(all_sequences)} seqüències.")
    print(f"Guardat {meta_output} amb {len(basenames)} noms de fitxer.")
    if errors:
        print(f"{len(errors)} fitxers no s'han pogut carregar en aquest chunk.")

print("Procés complet!")
