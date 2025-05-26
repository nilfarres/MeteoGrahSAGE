# MeteoGraphPC: aplicació de Graph Neural Networks (GNN) per a la millora de les prediccions meteorològiques als Països Catalans.

**Nil Farrés Soler , juny de 2025**

Aquest repositori recull el pipeline complet per al processament massiu, la normalització, la transformació en grafs i l'entrenament de diverses versions del model MeteoGraphPC basades en Graph Neural Networks (GNNs) per a la predicció de variables meteorològiques als Països Catalans. Es compta amb dades meteorològiques horàries des de 2016 i fins a 2024 cedides per la secció de meteorologia de 3Cat. Tot el procés està pensat per ser robust, escalable i reutilitzable.

A més dels scripts, s'hi inclouen dades reals d'exemple: **un dia complet (1 de gener de 2016)** de dades meteorològiques oficials, a raó d'un fitxer CSV per hora, dins el fitxer `DADES_METEO_PC.zip`. Això permet provar el pipeline sense necessitat de descarregar grans volums de dades.

---

## Taula de continguts

- [Descripció general del pipeline](#descripció-general-del-pipeline)
- [Estructura del repositori](#estructura-del-repositori)
- [Requisits](#requisits)
- [Ordre d'execució i explicació de cada script](#ordre-dexecució-i-explicació-de-cada-script)
    - [1. fitxers_buits.py](#1-fitxers_buitspy)
    - [2. sense_nomcol.py](#2-sense_nomcolpy)
    - [3. prep.py](#3-preppy)
    - [4. visualitzador_dades_prep.py](#4-visualitzador_dades_preppy)
    - [5. compute_PC_norm_params.py](#5-compute_pc_norm_paramspy)
    - [6. toData.py](#6-todatapy)
    - [7. generate_seq.py](#7-generate_seqpy)
    - [8. all_sequences.py](#8-all_sequencespy)
    - [9. MeteoGraphPC.py](#9-meteographpcpy)
- [Com utilitzar les dades d'exemple](#com-utilitzar-les-dades-dexemple)
- [Notes addicionals i recomanacions](#notes-adicionals-i-recomanacions)
- [Autor i contacte](#autor-i-contacte)
- [Llicència](#llicència)

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
8. **Entrenament i test de diverses versions del model MeteoGraphPC utilitzant GNN**.

---

## Estructura del repositori

.
├── DADES_METEO_PC.zip             # Un dia complet de dades meteorològiques horàries (fitxers CSV, per a proves).
├── fitxers_buits.py               # Detecció de fitxers buits o no llegibles.
├── sense_nomcol.py                # Detecció de fitxers sense la capçalera correcta.
├── prep.py                        # Preprocessament massiu i neteja de dades meteorològiques.
├── visualitzador_dades_prep.py    # Visualització ràpida de fitxers preprocessats.
├── compute_PC_norm_params.py      # Càlcul de paràmetres de normalització dels Països Catalans.
├── PC_norm_params.json            # Fitxer generat amb les mitjanes i desviacions estàndard dels Països Catalans.
├── toData.py                      # Conversió de fitxers preprocessats a grafs dinàmics (PyTorch Geometric).
├── execute_toData.bat             # Fitxer d'execució de `toData.py`.
├── generate_seq.py                # Generació de seqüències temporals de grafs dinàmics.
├── all_sequences.py               # Agrupació de seqüències temporals en chunks per entrenament.
├── MeteoGraphPC.py                # Entrenament i test de les diverses versions del model MeteoGraphPC basades en GNN.

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
    - Altres dependències bàsiques de la llibreria estàndard

**Instal·lació ràpida de dependències principals:**
```bash
pip install pandas numpy tqdm matplotlib scikit-learn torch torch_geometric torch_geometric_temporal networkx scipy argparse cupy
```

---

## Ordre d’execució i explicació dels scripts essencials

### 1. **`fitxers_buits.py`**
    - **Funció:** detecta fitxers CSV buits o no llegibles dins l'estructura de carpetes.
    - **Quan usar-lo:** com a primer pas, per assegurar que les dades d'origen no tinguin buits greus.
    - **Ús:** edita les variables de ruta i executa l'script. Es genera un fitxer amb el llistat de fitxers problemàtics.
    - **Requisits:** `pandas`, `tqdm`

### 2. **`sense_nomcol.py`**
    - **Funció:** detecta fitxers CSV que no tenen la capçalera estàndard a la primera línia.
    - **Quan usar-lo:** abans de processar, per detectar errors estructurals als fitxers.
    - **Ús:** edita les rutes al final de l'script i executa'l. Llista els fitxers sense capçalera correcta.
    - **Requisits:** `tqdm`

### 3. **`prep.py`**
    - **Funció:** preprocessament i neteja massiva de fitxers CSV meteorològics.
    - **Tasques:** conversió d’unitats, càlcul de variables, filtratge de dades, interpolació de buits, etc.
    - **Quan usar-lo:** obligatòriament primer per generar les dades netes i homogènies.
    - **Ús:** edita les rutes al final del codi i executa'l. Les dades preprocessades es guarden en un nou directori.
    - **Requisits:** `pandas`, `numpy`, `cupy`, `tqdm`, `logging`
    
### 4. **`visualitzador_dades_prep.py`**
    - **Funció:** visualitza ràpidament mapes i distribucions bàsiques de les dades meteorològiques preprocessades.
    - **Ús:** edita la variable `file_path` pel fitxer que vols visualitzar. Executa'l i trobaràs les imatges generades en una carpeta.
    - **Requisits:** `pandas`, `matplotlib`, `numpy`, `os`

### 5. **`compute_PC_norm_params.py`**
    - **Funció:** calcula les mitjanes i desviacions estàndard globals de totes les variables meteorològiques.
    - **Sortida:** fitxer `PC_norm_params.json` i gràfics histogrames.
    - **Quan usar-lo:** després de `prep.py` i abans de normalitzar/transformar les dades.
    - **Ús:** edita la variable d'entrada i executa'l. El fitxer JSON es farà servir als següents passos.
    - **Requisits:** `pandas`, `numpy`, `matplotlib`, `tqdm`

### 6. **`toData.py`**
    - **Funció:** converteix els fitxers CSV preprocessats en objectes de graf dinàmic compatibles amb PyTorch Geometric.
    - **Quan usar-lo:** Un cop tens les dades preprocessades i els paràmetres de normalització.
    - **Ús:** executa amb arguments de línia de comandes o fitxer `.bat`/`.sh`. S'han de passar les rutes d'entrada, sortida i el fitxer `PC_norm_params.json`.
    - **Requisits:** `pandas`, `numpy`, `torch`, `torch_geometric`, `tqdm`, `networkx`, `scipy`

### 7. **`generate_seq.py`**
    - **Funció:** genera seqüències temporals de grafs (sliding window) per entrenar models seqüencials.
    - **Quan usar-lo:** un cop generats els snapshots horaris amb `toData.py`.
    - **Ús:** executa'l amb arguments: directori d'entrada (snapshots `.pt`), directori de sortida, mida de finestra, stride, etc.
    - **Requisits:** `torch`, `tqdm`, `argparse`, `glob`

### 8. **`all_sequences.py`**
    - **Funció:** agrupa les seqüències temporals en chunks grans (fitxers agrupats) per a processament massiu o per transferència/entrenament eficient.
    - **Ús:** modifica les variables de ruta i mida de chunk dins el codi. Executa'l i trobaràs els fitxers agrupats i les metadades.
    - **Requisits:** `torch`, `tqdm`, `glob`, `os`

### 9. **`MeteoGraphPC.py`**
    - **Funció:** entrena i avalua models GNN seqüencials sobre les seqüències de grafs dinàmics generades.
    - **Característiques:** diverses arquitectures, validació, test, càlcul de mètriques i comparativa amb baselines.
    - **Ús:** executa'l passant la ruta a les seqüències (`--seq_dir`) i la resta de paràmetres desitjats (features, targets, optimitzadors, etc.).
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

> Nil Farrés Soler. *MeteoGraphPC: aplicació de Graph Neural Networks (GNN) per a la millora de les prediccions meteorològiques als Països Catalans*, 2025. [github.com/nilfarres/MeteoGraphPC](https://github.com/nilfarres/MeteoGraphPC)

Qualsevol ús comercial o publicació sense permís explícit es considera una vulneració dels drets de l'autor.

---

## Política de privacitat

Aquest repositori només conté codi font i dades meteorològiques públiques d'exemple.  
No es recull, emmagatzema ni processa cap dada personal ni identificativa dels usuaris.

---

*Per a qualsevol ús, dubte o consulta, contacta directament amb l'autor del projecte.*



