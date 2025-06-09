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
  1. Important: cal haver executat primer l'script "prep.py".
  2. Modifica la variable "file_path" per posar-hi el nom del fitxer que vols visualitzar.
  3. Executa l'script. Es crearà una carpeta amb les imatges generades.
  4. Trobaràs les visualitzacions (en format PNG) dins la carpeta "visualitzacio_[nom_fitxer]".

REQUISITS:
  - Python 3.x
  - Llibreries: pandas, matplotlib, numpy, os

AUTOR: Nil Farrés Soler
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

# ----------- PARÀMETRES ----------- #
file_path = "2024010300dadesPC_utc.csv"  # Substituir pel fitxer que es vulgui visualitzar
df = pd.read_csv(file_path)
nom_base = os.path.splitext(os.path.basename(file_path))[0]
carpeta_sortida = f"visualitzacio_dadesreals_{nom_base}"
os.makedirs(carpeta_sortida, exist_ok=True)

# ----------- COLORMAPS PERSONALITZATS ----------- #
cmap_temp = mcolors.LinearSegmentedColormap.from_list(
    "MeteoTemp",
    [
        (0.00, "#0033A0"), # blau fosc (molt fred)
        (0.22, "#339FFF"), # blau clar (fred)
        (0.40, "#A7F797"), # verd clar (temperatura suau, ~15-18°C)
        (0.55, "#F7F797"), # groc pàl·lid (suau/càlid, ~22-25°C)
        (0.70, "#FF8C00"), # taronja (càlid, ~30°C)
        (0.88, "#FF1E00"), # vermell (molt càlid)
        (1.00, "#A70000"), # granate (extrem)
    ]
)
cmap_pluja = mcolors.LinearSegmentedColormap.from_list(
    "MeteoPluja",
    [
        (0.00, "#FFFFFF"),
        (0.01, "#B2D2F1"),
        (0.15, "#88BFFD"),
        (0.20, "#5C9BE7"),
        (0.30, "#2A94EB"),
        (0.40, "#007CC4"),
        (0.60, "#004A8B"),
        (0.85, "#002050"),
        (1.00, "#4B0469"),
    ]
)

cmap_humitat = mcolors.LinearSegmentedColormap.from_list(
    "MeteoHumitat",
    [
        (0.00, "#F7FCF5"),
        (0.20, "#D9F0A3"),
        (0.40, "#A1D99B"),
        (0.65, "#41AB5D"),
        (0.85, "#238B45"),
        (1.00, "#00441B"),
    ]
)
cmap_patm = mcolors.LinearSegmentedColormap.from_list(
    "MeteoPatm",
    [
        (0.00, "#4A90E2"),
        (0.40, "#A7F797"),
        (1.00, "#BD10E0"),
    ]
)
cmap_ventfor = mcolors.LinearSegmentedColormap.from_list(
    "MeteoVentFor",
    [
        (0.00, "#C7E6FF"),  # blau cel
        (0.08, "#A1D0FF"),  # blau molt clar
        (0.16, "#75C0FF"),  # blau clar
        (0.22, "#5BA3F7"),  # blau mitjà
        (0.28, "#5180CF"),  # blau intens
        (0.34, "#6E6FC9"),  # blau-lila suau
        (0.40, "#8B77D1"),  # blau-lila
        (0.48, "#A086E2"),  # lila-blau clar
        (0.56, "#B699F4"),  # lila clar
        (0.62, "#A984E8"),  # lila clàssic
        (0.70, "#9C6BC8"),  # lila mitjà
        (0.78, "#8B49C6"),  # lila/morat
        (0.86, "#713BAF"),  # morat intens
        (0.93, "#613DC1"),  # morat fosc
        (1.00, "#2B184E"),  # morat-negre
    ]
)
cmap_alt = "viridis"

# ----------- FUNCIONS D'ESTÈTICA CARTOGRÀFICA ----------- #
def mapa_base(ax, lons, lats, margin=0.1):
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="white", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.8, edgecolor="black")
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.6, edgecolor="gray")
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), linewidth=0.5, edgecolor="blue", alpha=0.5)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="lightblue", alpha=0.5)
    ax.add_feature(cfeature.STATES.with_scale("10m"), linewidth=0.5, edgecolor="gray", linestyle=":")
    
    ax.set_extent([
        np.nanmin(lons) - margin, np.nanmax(lons) + margin,
        np.nanmin(lats) - margin, np.nanmax(lats) + margin
    ], crs=ccrs.PlateCarree())

def etiqueta_minmax(ax, vals, unitat):
    min_val = np.nanmin(vals)
    max_val = np.nanmax(vals)
    ax.text(
        0.99, 0.01,
        f"Min: {min_val:.2f} · Max: {max_val:.2f}\nDades reals · TFG Nil Farrés",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9, color="gray",
        bbox=dict(facecolor="white", alpha=0.3, edgecolor="none", pad=1)
    )

def plot_scatter_cartopy(lons, lats, vals, cmap, titol, unitat, outpath, vmin, vmax):
    fig = plt.figure(figsize=(10, 8), dpi=300, facecolor="white")
    ax = plt.axes(projection=ccrs.PlateCarree(), facecolor="white")
    mapa_base(ax, lons, lats)

    mask_valid = ~np.isnan(vals)
    mask_zero = (vals == 0) & mask_valid
    mask_nozero = (vals != 0) & mask_valid

    # Primer pinta els zeros de blanc
    ax.scatter(
        lons[mask_zero], lats[mask_zero], c="#FFFFFF",
        edgecolor="k", linewidth=0.4, s=50, alpha=0.92,
        transform=ccrs.PlateCarree(), zorder=3, label="0 mm"
    )
    # Ara pinta la resta de punts (no zeros) amb el colormap normal
    sc = ax.scatter(
        lons[mask_nozero], lats[mask_nozero], c=vals[mask_nozero],
        cmap=cmap, edgecolor="k", linewidth=0.4, s=50, alpha=0.92,
        transform=ccrs.PlateCarree(), zorder=4, vmin=vmin, vmax=vmax
    )

    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
    cbar.set_label(unitat, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title(titol, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Longitude (°)", fontsize=12)
    ax.set_ylabel("Latitude (°)", fontsize=12)
    ax.tick_params(labelsize=10)
    etiqueta_minmax(ax, vals, unitat)
    plt.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# ----------- MAPA DE LES ESTACIONS ----------- #
fig = plt.figure(figsize=(10, 8), dpi=300, facecolor="white")
ax = plt.axes(projection=ccrs.PlateCarree(), facecolor="white")
mapa_base(ax, df["lon"], df["lat"])
ax.scatter(df["lon"], df["lat"], color="red", marker="o", s=60, edgecolor="black", label="Estacions", transform=ccrs.PlateCarree())
ax.set_title("Ubicació de les estacions meteorològiques", fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Longitude (°)", fontsize=12)
ax.set_ylabel("Latitude (°)", fontsize=12)
ax.tick_params(labelsize=10)
plt.legend()
plt.tight_layout()
fig.savefig(f"{carpeta_sortida}/ubicacio_estacions.png", dpi=300)
plt.close(fig)

# ----------- MAPA DE TEMPERATURA ----------- #
plot_scatter_cartopy(
    df["lon"], df["lat"], df["Temp"],
    cmap_temp, "Temperatura a 2m", "Temperatura (°C)",
    f"{carpeta_sortida}/temperatura.png", vmin=-20, vmax=45
)

# ----------- MAPA D'HUMITAT ----------- #
plot_scatter_cartopy(
    df["lon"], df["lat"], df["Humitat"],
    cmap_humitat, "Humitat relativa a 2m", "Humitat relativa (%)",
    f"{carpeta_sortida}/humitat.png", vmin=0, vmax=100
)

# ----------- COMPROVAR SI HI HA PRECIPITACIÓ ----------- #
if df["Pluja"].max() > 0:
    plot_scatter_cartopy(
        df["lon"], df["lat"], df["Pluja"],
        cmap_pluja, "Pluja acumulada", "Pluja (mm)",
        f"{carpeta_sortida}/precipitacio.png", vmin=0, vmax=150
    )
else:
    print("Totes les estacions tenen 0.0 mm de precipitació, no es genera mapa.")

# ----------- MAPA DE DIRECCIÓ DEL VENT ----------- #
fig = plt.figure(figsize=(10, 8), dpi=300, facecolor="white")
ax = plt.axes(projection=ccrs.PlateCarree(), facecolor="white")
mapa_base(ax, df["lon"], df["lat"])
scale_factor = 0.05
u = np.cos(np.radians(df["VentDir"])) * df["VentFor"] * scale_factor
v = np.sin(np.radians(df["VentDir"])) * df["VentFor"] * scale_factor
ax.quiver(df["lon"], df["lat"], u, v, angles="xy", scale_units="xy", scale=1, color="blue", alpha=0.7, transform=ccrs.PlateCarree())
ax.scatter(df["lon"], df["lat"], color="red", marker="o", s=60, edgecolor="black", label="Estacions", transform=ccrs.PlateCarree())
ax.set_title("Direcció i intensitat del vent", fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Longitude (°)", fontsize=12)
ax.set_ylabel("Latitude (°)", fontsize=12)
ax.tick_params(labelsize=10)
plt.legend()
plt.tight_layout()
fig.savefig(f"{carpeta_sortida}/direccio_vent.png", dpi=300)
plt.close(fig)

# ----------- MAPA DE VELOCITAT DEL VENT ----------- #
plot_scatter_cartopy(
    df["lon"], df["lat"], df["VentFor"],
    cmap_ventfor, "Velocitat del vent a 10m", "Velocitat del vent (km/h)",
    f"{carpeta_sortida}/velocitat_vent.png", vmin=0, vmax=200
)

# ----------- MAPA DE PRESSIÓ ----------- #
plot_scatter_cartopy(
    df["lon"], df["lat"], df["Patm"],
    cmap_patm, "Pressió atmosfèrica (hPa)", "Pressió (hPa)",
    f"{carpeta_sortida}/pressio_atm.png", vmin=980, vmax=1040
)

# ----------- MAPA DE D'ALTITUD ----------- #
plot_scatter_cartopy(
    df["lon"], df["lat"], df["Alt"],
    cmap_alt, "Altitud (m)", "Altitud (m)",
    f"{carpeta_sortida}/altitud.png", vmin=np.nanmin(df["Alt"]), vmax=np.nanmax(df["Alt"])
)

print(f"Visualitzacions generades correctament a la carpeta: {carpeta_sortida}")
