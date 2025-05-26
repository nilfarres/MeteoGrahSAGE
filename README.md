# MeteoGraphPC: aplicació de Graph Neural Networks (GNN) per a la millora de les prediccions meteorològiques als Països Catalans.

**Nil Farrés Soler , juny de 2025**

Aquest repositori recull el pipeline complet per al processament massiu, la normalització, la transformació en grafs i l'entrenament de les diverses versions del model MeteoGraphPC basades en Graph Neural Networks (GNNs) per a la predicció de variables meteorològiques als Països Catalans. Tot el procés està pensat per ser robust, escalable i reutilitzable.

A més dels scripts, s'hi inclouen dades reals d'exemple: **un dia complet (1 de gener de 2016)** de dades meteorològiques oficials, a raó d'un fitxer CSV per hora, dins el fitxer `DADES_METEO_PC.zip`. Això permet provar el pipeline sense necessitat de descarregar grans volums de dades.

---

## Taula de continguts

- [Descripció general del pipeline](#descripció-general-del-pipeline)
- [Estructura del repositori](#estructura-del-repositori)
- [Requisits](#requisits)
- [Ordre d’execució i explicació de cada script](#ordre-dexecució-i-explicació-de-cada-script)
    - [1. fitxers_buits.py](#1-fitxers_buitspy)
    - [2. sense_nomcol.py](#2-sense_nomcolpy)
    - [3. prep.py](#3-preppy)
    - [4. visualitzador_dades_prep.py](#4-visualitzador_dades_preppy)
    - [5. compute_PC_norm_params.py](#5-compute_pc_norm_paramspy)
    - [6. toData.py](#6-todatapy)
    - [7. generate_seq.py](#7-generate_seqpy)
    - [8. all_sequences.py](#8-all_sequencespy)
    - [9. MeteoGraphPC.py](#9-meteographpcpy)
- [Com utilitzar les dades d’exemple](#com-utilitzar-les-dades-dexemple)
- [Notes addicionals i recomanacions](#notes-adicionals-i-recomanacions)
- [Autor i contacte](#autor-i-contacte)
- [Llicència](#llicència)

---

## Descripció general del pipeline

El projecte cobreix totes les fases necessàries per a la modelització meteorològica basada en grafs:
1. **Validació i neteja de dades massives** (detecció d’errors, buits, capçaleres incorrectes).
2. **Preprocessament i filtratge avançat** (càlculs, imputació de valors, transformacions).
3. **Normalització global** de variables meteorològiques.
4. **Conversió a grafs dinàmics** amb PyTorch Geometric.
5. **Generació de seqüències temporals** de grafs per a models seqüencials.
6. **Agrupació en chunks per entrenament eficient**.
7. **Entrenament i test de diversos models GNN seqüencials (MeteoGraphPC)**.
8. **Visualització ràpida de dades preprocessades**.

---

## Estructura del repositori
