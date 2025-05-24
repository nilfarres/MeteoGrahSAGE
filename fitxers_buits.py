#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fitxers_buits.py
==============================================================================
Script per detectar fitxers CSV buits i no llegibles dins l'estructura de carpetes de DADES_METEO_PC.

Aquest script recorre recursivament el directori arrel i localitza tots els fitxers
que acaben amb "dadesPC_utc.csv", excepte dins dels directoris exclosos.
Per a cada fitxer, comprova si està totalment buit o si no es pot llegir correctament.
Els resultats (fitxers buits i no llegibles) es guarden en un fitxer de text 
dins el directori de sortida especificat.

Ús:
  1. Edita les rutes "root_directory" (directori d'origen) i "output_directory" (on es guardarà el resultat) al final del codi.
  2. Executa l'script. El procés pot trigar depenent de la quantitat de fitxers.
  3. Consulta el fitxer de resultats generat per veure la llista de fitxers buits i no llegibles.

Requisits:
  - Python 3.x
  - Llibreries: pandas, tqdm

Autor: Nil Farrés Soler
==============================================================================
"""

import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_file(file_path):
    #print(f"Processant: {file_path}")
    if os.path.getsize(file_path) == 0:
        return ('empty', file_path)
    try:
        pd.read_csv(
            file_path,
            encoding='utf-8',
            na_values=['', ' '],
            quotechar='"',
            sep=',',
            engine='python',
            on_bad_lines='skip'
        )
        return ('ok', file_path)
    except UnicodeDecodeError:
        try:
            pd.read_csv(
                file_path,
                encoding='latin-1',
                na_values=['', ' '],
                quotechar='"',
                sep=',',
                engine='python',
                on_bad_lines='skip'
            )
            return ('ok', file_path)
        except Exception:
            return ('unreadable', file_path)
    except Exception:
        return ('unreadable', file_path)

def find_empty_csv_files_parallel(root_directory, output_directory):
    print("Iniciant el procés per trobar fitxers buits i no llegibles...")

    # Llista de directoris a ignorar
    excluded_directories = [
        "tauladades", "vextrems", "Admin_Estacions",
        "Clima", "Clima METEOCAT", "error_VAR", "html", "png", "var_vextrems", 
        "2013", "2014", "2015"
    ]

    files_to_process = []
    for root, dirs, files in os.walk(root_directory):
        # Ignorem els directoris que coincideixin amb la llista d'exclusions
        if any(excluded_dir in root for excluded_dir in excluded_directories):
            #print(f"S'ha ignorat el directori: {root}")
            continue

        #print(f"Processant directori: {root} amb {len(files)} fitxers.")
        for file in files:
            if file.endswith('dadesPC_utc.csv'):
                files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        print("No s'han trobat fitxers que acabin amb 'dadesPC_utc.csv' en aquest directori.")
        return

    print(f"S'han trobat {len(files_to_process)} fitxers per processar.")
    results = []
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_file, files_to_process),
            total=len(files_to_process),
            desc="Processant fitxers"
        ))

    empty_files = [res[1] for res in results if res[0] == 'empty']
    unreadable_files = [res[1] for res in results if res[0] == 'unreadable']

    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, "fitxers_buits_PC_python.txt")
    with open(output_file, "w") as f:
        f.write(f"S'han trobat {len(empty_files)} fitxers totalment buits.\n")
        for idx, file in enumerate(empty_files, 1):
            f.write(f"{idx}: {file}\n")
        f.write("\n")
        f.write(f"S'han trobat {len(unreadable_files)} fitxers no llegibles.\n")
        for idx, file in enumerate(unreadable_files, 1):
            f.write(f"{idx}: {file}\n")

    print(f"S'han trobat {len(empty_files)} fitxers totalment buits.")
    print(f"S'han trobat {len(unreadable_files)} fitxers no llegibles.")
    print(f"Els resultats s'han guardat a: {output_file}")

try:
    root_directory = 'F:/DADES_METEO_PC'
    output_directory = 'D:/Documentos/TREBALL_FINAL_DE_GRAU/fitxers_buits'
    find_empty_csv_files_parallel(root_directory, output_directory)
except Exception as e:
    print(f"S'ha produït un error inesperat: {e}")






