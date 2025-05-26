#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
visualitzador_dades_prep.py

Script per visualitzar ràpidament dades meteorològiques de fitxers CSV preprocessats.

Aquest script llegeix un fitxer CSV amb dades meteorològiques preprocessades (sortida de prep.py) 
i genera automàticament diferents mapes i visualitzacions bàsiques per a totes les 
variables més rellevants: ubicació de les estacions, temperatura, humitat, precipitació (s'hi n'hi ha hagut), 
direcció i força del vent, pressió atmosfèrica i altitud.

FUNCIONALITATS:
  - Llegeix un fitxer acabat en "dadesPC_utc.csv" i en crea una carpeta amb els mapes corresponents.
  - Genera mapes amb la distribució de cada variable, utilitzant matplotlib.
  - Visualitza la ubicació de les estacions, la temperatura, la humitat, la precipitació (si n'hi ha hagut), 
    la direcció i força del vent, la pressió atmosfèrica i l'altitud.
  - Desa cada gràfic com a imatge PNG dins una carpeta amb el nom del fitxer.

INSTRUCCIONS D'ÚS:
  1. Modifica la variable "file_path" per posar-hi el nom del fitxer que vols visualitzar.
  2. Executa l'script. Es crearà una carpeta amb les imatges generades.
  3. Trobaràs les visualitzacions (en format PNG) dins la carpeta "visualitzacio_[nom_fitxer]".

REQUISITS:
  - Python 3.x
  - Llibreries: pandas, matplotlib, numpy, os

AUTOR: Nil Farrés Soler
==============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------- PARÀMETRES ----------- #
file_path = "2016010100dadesPC_utc.csv"  # Substitueix pel fitxer que vulguis visualitzar
df = pd.read_csv(file_path)

# Crear el nom de la carpeta a partir del nom del fitxer
nom_base = os.path.splitext(os.path.basename(file_path))[0]
carpeta_sortida = f"visualitzacio_{nom_base}"
os.makedirs(carpeta_sortida, exist_ok=True)

# ----------- MAPA DE LES ESTACIONS ----------- #
plt.figure(figsize=(8, 6))
plt.scatter(df["lon"], df["lat"], c="red", marker="o", s=50, label="Estacions")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Ubicació de les estacions meteorològiques")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{carpeta_sortida}/ubicacio_estacions.png")
plt.close()

# ----------- MAPA DE TEMPERATURA ----------- #
plt.figure(figsize=(8, 6))
sc = plt.scatter(df["lon"], df["lat"], c=df["Temp"], cmap="YlOrRd", marker="o", edgecolor="black", s=50)
plt.colorbar(sc, label="Temperatura (°C)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Temperatura (°C)")
plt.clim(df["Temp"].min(), df["Temp"].max())
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{carpeta_sortida}/temperatura.png")
plt.close()

# ----------- MAPA D’HUMITAT ----------- #
plt.figure(figsize=(8, 6))
sc = plt.scatter(df["lon"], df["lat"], c=df["Humitat"], cmap="Greens", marker="o", edgecolor="black", s=50)
plt.colorbar(sc, label="Humitat relativa (%)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Humitat (%)")
plt.clim(df["Humitat"].min(), df["Humitat"].max())
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{carpeta_sortida}/humitat.png")
plt.close()

# ----------- COMPROVAR SI HI HA PRECIPITACIÓ ----------- #
if df["Pluja"].max() > 0:
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df["lon"], df["lat"], c=df["Pluja"], cmap="Blues", marker="o", edgecolor="black", s=50)
    plt.colorbar(sc, label="Pluja (mm)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title("Precipitació (mm)")
    plt.clim(df["Pluja"].min(), df["Pluja"].max())
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{carpeta_sortida}/precipitacio.png")
    plt.close()
else:
    print("Totes les estacions tenen 0.0 mm de precipitació, no es genera mapa.")

# ----------- MAPA DE DIRECCIÓ DEL VENT ----------- #
plt.figure(figsize=(8, 6))
scale_factor = 0.05
u = np.cos(np.radians(df["VentDir"])) * df["VentFor"] * scale_factor
v = np.sin(np.radians(df["VentDir"])) * df["VentFor"] * scale_factor
plt.quiver(df["lon"], df["lat"], u, v, angles="xy", scale_units="xy", scale=1, color="blue", alpha=0.7)
plt.scatter(df["lon"], df["lat"], color="red", marker="o", s=50, label="Estacions")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Direcció i intensitat del vent")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{carpeta_sortida}/direccio_vent.png")
plt.close()

# ----------- MAPA DE FORÇA DEL VENT ----------- #
plt.figure(figsize=(8, 6))
sc = plt.scatter(df["lon"], df["lat"], c=df["VentFor"], cmap="Purples", marker="o", edgecolor="black", s=50)
plt.colorbar(sc, label="Velocitat del vent (km/h)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Velocitat del vent (km/h)")
plt.clim(df["VentFor"].min(), df["VentFor"].max())
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{carpeta_sortida}/velocitat_vent.png")
plt.close()

# ----------- MAPA DE PRESSIÓ ----------- #
plt.figure(figsize=(8, 6))
sc = plt.scatter(df["lon"], df["lat"], c=df["Patm"], cmap="coolwarm", marker="o", edgecolor="black", s=50)
plt.colorbar(sc, label="Pressió (hPa)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Pressió atmosfèrica (hPa)")
plt.clim(df["Patm"].min(), df["Patm"].max())
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{carpeta_sortida}/pressio_atm.png")
plt.close()

# ----------- MAPA DE D'ALTITUD ----------- #
plt.figure(figsize=(8, 6))
sc = plt.scatter(df["lon"], df["lat"], c=df["Alt"], cmap="viridis", marker="o", edgecolor="black", s=50)
plt.colorbar(sc, label="Altitud (m)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Altitud (m)")
plt.clim(df["Alt"].min(), df["Alt"].max())
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{carpeta_sortida}/altitud.png")
plt.close()

print (f"Visualitzacions generades correctament a la carpeta: {carpeta_sortida}")
