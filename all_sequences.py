import torch
import glob
import os
from tqdm import tqdm

SEQ_DIR = r'F:\ws120_str12_hh120\DADES_METEO_PC_generated_seqs_v8_ws120_str12_hh120' #Substituir pels directoris pertinents
OUTPUT_BASE = r'F:\ws120_str12_hh120\All_Sequences_v8_ws120_str12_hh120_chunksde50'

CHUNK_SIZE = 50  # Nombre de seqüències per fitxer agrupat

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
