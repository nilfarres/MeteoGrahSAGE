# Aplicació de Graph Neural Networks per a la millora de les prediccions meteorològiques als Països Catalans.

**Nil Farrés Soler , juny de 2025**

Aquest repositori recull el pipeline complet per al processament massiu, la normalització, la transformació en grafs i l'entrenament del model MeteoGraphPC basat en Graph Neural Networks (GNNs) per a la predicció de variables meteorològiques als Països Catalans. Es compta amb dades meteorològiques horàries des de 2016 i fins a 2024 cedides per la secció de meteorologia de 3Cat. Tot el procés està pensat per ser robust, escalable i reutilitzable.

A més dels scripts, s'hi inclouen dades reals d'exemple: **un dia complet (1 de gener de 2016)** de dades meteorològiques oficials, a raó d'un fitxer CSV per hora, dins el fitxer `DADES_METEO_PC.zip`. Això permet provar el pipeline sense necessitat de descarregar grans volums de dades.

---

## Taula de continguts

- [Descripció general del pipeline](#descripció-general-del-pipeline)
- [Estructura del repositori](#estructura-del-repositori)
- [Requisits](#requisits)
- [Ordre d'execució i explicació de cada script](#ordre-dexecució-i-explicació-de-cada-script)
- [Com utilitzar les dades d'exemple](#com-utilitzar-les-dades-dexemple)
- [Notes addicionals i recomanacions](#notes-adicionals-i-recomanacions)
- [Autor i contacte](#autor-i-contacte)
- [Condicions d’ús i citació](#condicions-dús-i-citació)

---

## Descripció general del pipeline

Aquest projecte cobreix totes les fases necessàries per a la modelització meteorològica basada en grafs dinàmics:
1. **Validació i neteja de dades massives** (detecció d'errors, buits de dades, capçaleres incorrectes).
2. **Preprocessament i filtratge avançat** (càlculs, imputació de valors, transformacions).
3. **Visualització ràpida de les dades preprocessades**.
4. **Normalització global** de variables meteorològiques.
5. **Conversió a grafs dinàmics** amb PyTorch Geometric.
6. **Generació de seqüències temporals** de grafs per a models seqüencials.
7. **Agrupació en chunks per un entrenament més eficient**.
8. **Entrenament, validació i test del model MeteoGraphPC utilitzant GNN**.
9. **Anàlisi detallat de les prediccions de MeteoGraphPC**.

---

## Estructura del repositori

```text
.
├── DADES_METEO_PC.zip             # Un dia complet de dades meteorològiques horàries (fitxers CSV, per a proves).
├── fitxers_buits.py               # Detecció de fitxers buits o no llegibles.
├── sense_nomcol.py                # Detecció de fitxers sense la capçalera correcta.
├── prep.py                        # Preprocessament massiu i neteja de dades meteorològiques.
├── visualitzador_dades_prep.py    # Visualització ràpida de fitxers preprocessats.
├── compute_PC_norm_params.py      # Càlcul de paràmetres de normalització dels Països Catalans.
├── PC_norm_params.json            # Fitxer generat amb les mitjanes i desviacions estàndard dels Països Catalans.
├── toData.py                      # Conversió de fitxers preprocessats a grafs dinàmics (PyTorch Geometric). Execució amb execute_toData.bat.
├── generate_seq.py                # Generació de seqüències temporals de grafs dinàmics.
├── all_sequences.py               # Agrupació de seqüències temporals en chunks per entrenament.
├── MeteoGraphPC.py                # Entrenament, validació i test del model MeteoGraphPC basat en GNN. Execució amb run_MeteoGraphPC.bat.
├── visualitzacio_metriques.py     # Visualització de les mètriques del model durant l'entrenament, la validació i el test a partir d'un fitxer csv.
├── nodes_metadata.py              # Crea un fitxer csv amb tots els nodes del dataset juntament amb la seva localització.
├── matriu_corr.py                 # Genera una matriu de correlació de les prediccions al test i una per les dades reals.
├── inferencia_meteographpc.py     # Crea un fitxer en format NetCDF a partir de les prediccions generades per MeteoGraphPC.
├── mapa_preds.py                  # Genera mapes per visualitzar les prediccions de MeteoGraphPC a partir d'un fitxer NetCDF. Execució amb mapa_preds.bat
```

**Important: cal executar els codis en l'ordre anterior.**

---

## Requisits

- **Python 3.8+**  
- **Llibreries Python** (es recomana utilitzar un entorn virtual):
    - `pandas`
    - `numpy`
    - `cupy` (opcional, per acceleració numèrica, recomanada GPU)
    - `tqdm`
    - `logging`
    - `matplotlib`
    - `scikit-learn`
    - `torch` i `torch_geometric`
    - `torch_geometric_temporal`
    - `networkx`
    - `scipy`
    - `argparse`
    - `netCDF4`
    - Altres dependències bàsiques de la llibreria estàndard

**Instal·lació ràpida de dependències principals:**
```bash
conda create -n meteographpc python==3.8
conda activate meteographpc
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch_geometric
pip install torch-geometric-temporal==0.56.0
```

---

## Ordre d'execució i explicació de cada script principal

 1. **`fitxers_buits.py`**
    - **Funció:** detectar fitxers CSV buits o no llegibles dins l'estructura de carpetes.
    - **Quan usar-lo:** com a primer pas, per assegurar que les dades d'origen no tinguin buits greus.
    - **Ús:** edita les variables de ruta i executa l'script. Es genera un fitxer amb el llistat de fitxers problemàtics.
    - **Requisits:** `pandas`, `tqdm`

2. **`sense_nomcol.py`**
    - **Funció:** detectar fitxers CSV que no tenen la capçalera estàndard a la primera línia.
    - **Quan usar-lo:** abans de processar, per detectar errors estructurals als fitxers.
    - **Ús:** edita les rutes al final de l'script i executa'l. Llista els fitxers sense capçalera correcta.
    - **Requisits:** `tqdm`

3. **`prep.py`**
    - **Funció:** preprocessar i netejar els fitxers CSV meteorològics originals.
    - **Tasques:** conversió d'unitats, càlcul de variables, filtratge de dades, interpolació de buits, etc.
    - **Quan usar-lo:** obligatòriament primer per generar les dades netes i homogènies.
    - **Ús:** edita les rutes al final del codi i executa'l. Les dades preprocessades es guarden en un nou directori.
    - **Requisits:** `pandas`, `numpy`, `cupy`, `tqdm`, `logging`
    
4. **`visualitzador_dades_prep.py`**
    - **Funció:** visualitzar ràpidament mapes i distribucions bàsiques de les dades meteorològiques preprocessades.
    - **Ús:** edita la variable `file_path` pel fitxer que vols visualitzar. Executa'l i trobaràs les imatges generades en una carpeta.
    - **Requisits:** `pandas`, `matplotlib`, `numpy`, `os`

5. **`compute_PC_norm_params.py`**
    - **Funció:** calcular les mitjanes i desviacions estàndard globals de totes les variables meteorològiques.
    - **Sortida:** fitxer `PC_norm_params.json` i gràfics histogrames.
    - **Quan usar-lo:** després de `prep.py` i abans de normalitzar/transformar les dades.
    - **Ús:** edita la variable d'entrada i executa'l. El fitxer JSON es farà servir als següents passos.
    - **Requisits:** `pandas`, `numpy`, `matplotlib`, `tqdm`

6. **`toData.py`**
    - **Funció:** convertir els fitxers CSV preprocessats en objectes de graf dinàmic compatibles amb PyTorch Geometric.
    - **Quan usar-lo:** un cop es tenen les dades preprocessades i els paràmetres de normalització.
    - **Ús:** executa amb el fitxer `execute_toData.bat`. S'han de passar les rutes d'entrada, sortida i el fitxer `PC_norm_params.json`.
    - **Requisits:** `pandas`, `numpy`, `torch`, `torch_geometric`, `tqdm`, `networkx`, `scipy`

7. **`generate_seq.py`**
    - **Funció:** generar seqüències temporals de grafs (sliding window) per entrenar models seqüencials.
    - **Quan usar-lo:** un cop generats els snapshots horaris amb `toData.py`.
    - **Ús:** executa'l havent definit els arguments següents: directori d'entrada (snapshots `.pt`), directori de sortida, mida de finestra, stride i horitzó de predicció.
    - **Requisits:** `torch`, `tqdm`, `argparse`, `glob`

8. **`all_sequences.py`**
    - **Funció:** agrupar les seqüències temporals en chunks grans (fitxers agrupats) per a processament massiu o per transferència/entrenament eficient.
    - **Ús:** modifica les variables de ruta i mida de chunk dins el codi. Executa'l i trobaràs els fitxers agrupats i les metadades.
    - **Requisits:** `torch`, `tqdm`, `glob`, `os`

9. **`MeteoGraphPC.py`**
    - **Funció:** entrenar i avaluar el model MeteoGraphPC basat en GNN sobre les seqüències de grafs dinàmics generades.
    - **Característiques:** entrenament, validació, test, càlcul de mètriques i comparativa amb baselines.
    - **Ús:** executa'l amb el fitxer `run_MeteoGraphPC.sh` passant la ruta a les seqüències (`--seq_dir`) i la resta de paràmetres desitjats (features, targets, optimitzadors, etc.).
    - **Requisits:** `torch`, `torch_geometric`, `torch_geometric_temporal`, `numpy`, `pandas`, `tqdm`, `scikit-learn`

---

## Com utilitzar les dades d’exemple

Aquest repositori inclou el fitxer **DADES_METEO_PC.zip**, que conté un dia complet (1 de gener de 2016) de dades meteorològiques horàries dels Països Catalans (un fitxer CSV per hora).

### Passos per fer servir aquestes dades de mostra:

1. Descomprimeix el fitxer `DADES_METEO_PC.zip` al teu ordinador, a la ruta que prefereixis.
2. A dins hi trobaràs fitxers anomenats `YYYYMMDDHHdadesPC_utc.csv` (per exemple, `2016010100dadesPC_utc.csv`, `2016010101dadesPC_utc.csv`, ..., fins a `2016010123dadesPC_utc.csv`).
3. Fes servir la carpeta resultant com a font d'entrada quan executis els scripts del pipeline, especialment:
    - `prep.py`
    - `compute_PC_norm_params.py`
    - `toData.py`
    - La resta d'scripts que utilitzen les dades meteorològiques d'origen.

### Notes:

- Aquest conjunt de dades d'exemple et permet provar tot el pipeline sense necessitat de descarregar dades massives. Ara bé, caldrà modificar la funció `split` del codi MeteoGraphPC.py ja que ara es troba creada amb l'objectiu de dividir el dataset complet de dades (entre 2016 i 2024).  
- es poden modificar en qualsevol moment les rutes d'entrada als scripts segons on s'hagi desat la carpeta descomprimida.
- Per treballar amb més dades, cal posar-se en contacte amb l'autor d'aquest projecte o amb la secció de meteorologia de 3Cat i es valorarà la sol·licitud.

---

## Notes addicionals i recomanacions

- **Logs i sortides:**  
  la majoria d'scripts generen automàticament carpetes amb els resultats, com ara `logs/`, `histogrames/`, `visualitzacio_.../`, etc. Cal revisa-les per comprovar que tot el procés s'hagi executat correctament i per identificar possibles errors o avisos.

- **Escalabilitat:**  
  el pipeline està pensat per funcionar tant amb conjunts de dades petits (com l'exemple del repositori) com amb milers de fitxers. Molts scripts aprofiten la paral·lelització (CPU o GPU) per optimitzar el temps de processament.

- **Compatibilitat:**  
  els scripts són compatibles amb Windows o Linux, sempre que es tinguin instal·lades correctament les dependències Python necessàries. Per a ús intensiu, es recomana executar-los en un entorn de càlcul robust (servidor o clúster amb GPU).

- **Estructura flexible:**  
  es poden modificar fàcilment les rutes d'entrada i sortida, així com els paràmetres de preprocessament, normalització, finestres temporals, etc., editant les variables dels scripts o passant arguments per línia de comandes.

- **Es recomana seguir sempre l'ordre d'execució** proposat per evitar inconsistències i garantir la coherència entre els diferents passos.

- **Documentació interna:**  
  cada script inclou una capçalera explicativa amb instruccions. Cal consulta-la sempre que es tinguin dubtes sobre el funcionament concret d'algun pas i, si cal, contactar amb l'autor del projecte.

- **Recomanació general:**  
  cal treballa primer amb les dades d'exemple per assegurar-se que s'entén el pipeline, i després escalar-ho als propis conjunts de dades pertinents.

---

## Autor i contacte

Aquest projecte ha estat desenvolupat per:

**Nil Farrés Soler**  
- Correu electrònic: nil.farres@autonoma.cat  
- GitHub: [github.com/nilfarres](https://github.com/nilfarres)

Amb la col·laboració de la [secció de meteorologia de 3Cat](https://www.3cat.cat/el-temps/) i el [Centre de Visió per Computador](https://www.cvc.uab.es/).

Si es tenen dubtes o suggeriments, no dubtar a posar-se en contacte amb l'autor d'aquest projecte.  
S'agrairà qualsevol feedback o menció de resultats obtinguts gràcies a aquest projecte.

---

## Condicions d’ús i citació

Aquest projecte es distribueix **sense llicència explícita**.  
Tots els drets reservats. **No està permès fer-ne ús, còpia, modificació ni distribució sense autorització prèvia i per escrit de l'autor.**

Si es vol utilitzar el codi, les dades o qualsevol recurs d'aquest repositori per a:
- ús personal o educatiu
- ús en treballs acadèmics o de recerca
- publicacions, presentacions o projectes derivats

**cal que es demani permís a l'autor i, en cas que la proposta sigui acceptada, es citi explícitament l'autor i el repositori**:

> Nil Farrés Soler. *Aplicació de Graph Neural Networks per a la millora de les prediccions meteorològiques als Països Catalans*, 2025. [github.com/nilfarres/MeteoGraphPC](https://github.com/nilfarres/MeteoGraphPC)

Qualsevol ús comercial o publicació sense permís explícit es considera una vulneració dels drets de l'autor.

Aquest repositori només conté codi font i dades meteorològiques públiques d'exemple.

---

*Per a qualsevol ús, dubte o consulta, contacta directament amb l'autor del projecte.*
