import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar el fitxer CSV
file_path = "NOM_DEL_FITXER.csv"  # Substitueix pel nom del fitxer correcte
df = pd.read_csv(file_path)

# ----------------- MAPA DE LES ESTACIONS -----------------
plt.figure(figsize=(8, 6))
plt.scatter(df["lon"], df["lat"], c="red", marker="o", label="Estacions")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Ubicació de les Estacions Meteorològiques")
plt.legend()
plt.grid(True)
plt.show()

# ----------------- MAPA DE TEMPERATURA -----------------
plt.figure(figsize=(8, 6))
sc = plt.scatter(df["lon"], df["lat"], c=df["Temp"], cmap="coolwarm", marker="o", edgecolor="black")
cbar = plt.colorbar(sc)
cbar.set_label("Temperatura (°C)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Temperatura a les Estacions Meteorològiques")
plt.grid(True)
plt.show()

# ----------------- MAPA DE HUMITAT -----------------
plt.figure(figsize=(8, 6))
sc = plt.scatter(df["lon"], df["lat"], c=df["Humitat"], cmap="Blues", marker="o", edgecolor="black")
cbar = plt.colorbar(sc)
cbar.set_label("Humitat (%)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Humitat a les Estacions Meteorològiques")
plt.grid(True)
plt.show()

# ----------------- COMPROVAR SI HI HA PRECIPITACIÓ -----------------
if df["Pluja"].max() > 0:
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df["lon"], df["lat"], c=df["Pluja"], cmap="Greens", marker="o", edgecolor="black")
    cbar = plt.colorbar(sc)
    cbar.set_label("Pluja (mm)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title("Precipitació a les Estacions Meteorològiques")
    plt.grid(True)
    plt.show()
else:
    print("Totes les estacions tenen 0.0 mm de precipitació, no es genera mapa.")

# ----------------- MAPA DE DIRECCIÓ DEL VENT -----------------
plt.figure(figsize=(8, 6))
scale_factor = 0.05  # Factor d'ajust per fer els vectors més visibles
u = np.cos(np.radians(df["VentDir"])) * df["VentFor"] * scale_factor
v = np.sin(np.radians(df["VentDir"])) * df["VentFor"] * scale_factor
plt.quiver(df["lon"], df["lat"], u, v, angles="xy", scale_units="xy", scale=1, color="blue", alpha=0.7)
plt.scatter(df["lon"], df["lat"], color="red", marker="o", label="Estacions")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Direcció i Intensitat del Vent a les Estacions Meteorològiques")
plt.legend()
plt.grid(True)
plt.show()

# ----------------- MAPA DE VELOCITAT DEL VENT -----------------
plt.figure(figsize=(8, 6))
sc = plt.scatter(df["lon"], df["lat"], c=df["VentFor"], cmap="Purples", marker="o", edgecolor="black")
cbar = plt.colorbar(sc)
cbar.set_label("Velocitat del Vent")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Velocitat del Vent a les Estacions Meteorològiques")
plt.grid(True)
plt.show()
