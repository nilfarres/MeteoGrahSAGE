import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_file(file_path):
    """
    Processa un fitxer CSV per verificar si la primera línia coincideix amb la capçalera coneguda.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                first_line = f.readline().strip()
        except Exception:
            return ('unreadable', file_path)
    
    # Capçalera coneguda
    known_header = '"nu","id","Font","Data","Poblacio","lat","lon","Temp","Temp.Max","Temp.Min","Amplitud.Termica","VentDir","VentFor","VentMax","Simbolvent","Windchill","Humitat","Humidex","Pluja","Alt","Patm","WEBCAMS","NomOK","Comarca"'
    
    if first_line == known_header:
        return ('has_header', file_path)
    else:
        return ('missing_header', file_path)

def find_csv_files_missing_header(root_directory, output_directory):
    print("Iniciant el procés per trobar fitxers CSV sense capçalera a la primera línia...")
    
    # Llista de directoris a ignorar
    excluded_directories = [
        "tauladades", "vextrems", "Admin_Estacions",
        "Clima", "Clima METEOCAT", "error_VAR", "html", "png",
        "var_vextrems", "2013", "2014", "2015"
    ]
    
    files_to_process = []
    for root, dirs, files in os.walk(root_directory):
        if any(excluded_dir in root for excluded_dir in excluded_directories):
            continue
        for file in files:
            if file.endswith("dadesPC_utc.csv"):
                files_to_process.append(os.path.join(root, file))
                
    if not files_to_process:
        print("No s'han trobat fitxers CSV que acabin amb 'dadesPC_utc.csv' en aquest directori.")
        return

    print(f"S'han trobat {len(files_to_process)} fitxers per processar.")
    
    results = []
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_file, files_to_process),
            total=len(files_to_process),
            desc="Processant fitxers"
        ))
        
    # Fitxers sense capçalera a la primera línia
    missing_header_files = [res[1] for res in results if res[0] == 'missing_header']
    
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, "fitxers_csv_sense_capçalera_PC.txt")
    
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(f"S'han trobat {len(missing_header_files)} fitxers CSV sense capçalera a la primera línia.\n")
        for idx, file in enumerate(missing_header_files, 1):
            f.write(f"{idx}: {file}\n")
    
    print(f"S'han trobat {len(missing_header_files)} fitxers CSV sense capçalera a la primera línia.")
    print(f"Els resultats s'han guardat a: {output_file}")

if __name__ == '__main__':
    try:
        root_directory = 'F:/DADES_METEO_PC'
        output_directory = 'D:/Documentos/TREBALL_FINAL_DE_GRAU/fitxers_sense_capçalera'
        find_csv_files_missing_header(root_directory, output_directory)
    except Exception as e:
        print(f"S'ha produït un error inesperat: {e}")

