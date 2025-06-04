import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓ
# ─────────────────────────────────────────────────────────────────────────────
# Ruta al fitxer CSV
csv_path = "C:/Users/nfarres/Documents/TFG/models/model_curt_termini_prova/train_MeteoGraphPC_v1_ws48_str12_hh6_20250603_093600_complet.csv"

# Ruta de sortida dels gràfics i taules
output_dir = "C:/Users/nfarres/Documents/TFG/models/model_curt_termini_prova/plots_metriques"
os.makedirs(output_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CARREGA I PREPARA EL DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────
# Saltem les 3 primeres línies de metadades i usem la 4a com a header
df = pd.read_csv(csv_path, skiprows=3)

# Neteja mínima dels noms de columnes
df.rename(columns=lambda x: x.strip().lstrip("#").strip(), inplace=True)

# Ens quedem només amb les etapes que ens interessen
df = df[df["stage"].isin(["train", "val", "test"])].copy()

# Omplim els NaN d'epoch i el convertim a enter
df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").ffill().astype(int)

# Convertem les mètriques a numèric
metrics = ["loss", "RMSE", "MAE", "R2", "SMAPE"]
for col in metrics:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Separem train / val / test
df_train = df[df["stage"] == "train"].copy()
df_val   = df[df["stage"] == "val"].copy()
df_test  = df[df["stage"] == "test"].copy().drop(columns="epoch", errors='ignore')

# Matriu per calcular diferències
df_diff  = df_train.merge(df_val, on="epoch", suffixes=("_train", "_val"))

# ─────────────────────────────────────────────────────────────────────────────
# ESTÈTICA
# ─────────────────────────────────────────────────────────────────────────────
sns.set(style="whitegrid")
palette = {"train": "#1f77b4", "val": "#ff7f0e"}
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ─────────────────────────────────────────────────────────────────────────────
# 1. MATRIU DE CORRELACIÓ ENTRE MÈTRIQUES
# ─────────────────────────────────────────────────────────────────────────────
def matriu_correlacio(df_subset, stage):
    corr = df_subset[metrics].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"Matriu de correlació de mètriques ({stage})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"correlacio_metrics_{stage}.png"), dpi=150)
    plt.close()

matriu_correlacio(df_train, "train")
matriu_correlacio(df_val,   "val")

# ─────────────────────────────────────────────────────────────────────────────
# 2. GRÀFICS PER A CADA MÈTRICA + ANOTACIÓ MILLOR VAL LOSS
# ─────────────────────────────────────────────────────────────────────────────
best_val_loss_epoch = int(df_val.loc[df_val["loss"].idxmin(), "epoch"])
for metric in metrics:
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=df_train, x="epoch", y=metric, label="train",
                 marker="o", color=palette["train"])
    sns.lineplot(data=df_val,   x="epoch", y=metric, label="val",
                 marker="o", color=palette["val"])
    if not df_test.empty and pd.notnull(df_test[metric].values[0]):
        test_val = df_test[metric].values[0]
        plt.axhline(test_val, linestyle="--", color="gray",
                    label=f"test = {test_val:.4f}")

    if metric == "loss":
        min_loss = df_val["loss"].min()
        plt.axvline(best_val_loss_epoch, linestyle=":", color="red",
                    label=f"Millor val (època {best_val_loss_epoch})")
        plt.text(best_val_loss_epoch, min_loss, f"{min_loss:.4f}",
                 color="red", ha="left", va="bottom")

    plt.title(f"Evolució de la mètrica: {metric}")
    plt.xlabel("Època")
    plt.ylabel(metric)
    plt.legend(title="Conjunt")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_entrenament.png"), dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 3. GRÀFIC DE DIFERÈNCIA (VAL - TRAIN) PER MÈTRICA
# ─────────────────────────────────────────────────────────────────────────────
for metric in metrics:
    diff = df_diff[f"{metric}_val"] - df_diff[f"{metric}_train"]
    plt.figure(figsize=(8, 4))
    sns.lineplot(x=df_diff["epoch"], y=diff, marker="o", color="purple")
    plt.axhline(0, linestyle="--", color="black", linewidth=1)
    plt.title(f"Diferència (val - train) de {metric}")
    plt.xlabel("Època")
    plt.ylabel(f"Diferència de {metric}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_diferencia_val_train.png"), dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4. BOXPLOT DE MÈTRIQUES PER TRAMS D'ÈPOQUES
# ─────────────────────────────────────────────────────────────────────────────
def boxplot_metrics(df_subset, stage):
    n_epochs = df_subset["epoch"].nunique()
    if n_epochs < 6:
        return
    bins = [
        df_subset["epoch"].min(),
        df_subset["epoch"].min() + (n_epochs // 3),
        df_subset["epoch"].min() + 2*(n_epochs // 3),
        df_subset["epoch"].max() + 1
    ]
    labels = ['Inici', 'Mig', 'Final']
    df_subset["Tram"] = pd.cut(df_subset["epoch"], bins=bins,
                               labels=labels, right=False, include_lowest=True)
    for metric in metrics:
        plt.figure(figsize=(7, 4))
        sns.boxplot(
            data=df_subset,
            x="Tram",
            y=metric,
            hue="Tram",        # ara palette s’associa a hue
            palette="Set2",
            legend=False       # evitem la llegenda duplicada
        )

        plt.title(f"Distribució de {metric} ({stage}) per tram d'èpoques")
        plt.xlabel("Tram d'èpoques")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"boxplot_{metric}_{stage}.png"), dpi=150)
        plt.close()

boxplot_metrics(df_train, "train")
boxplot_metrics(df_val,   "val")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TAULA RESUM
# ─────────────────────────────────────────────────────────────────────────────
def crea_taula_resum():
    resum = {}
    for stg, dfx in zip(['train', 'val', 'test'], [df_train, df_val, df_test]):
        if dfx.empty:
            continue
        row = {}
        for metric in metrics:
            if stg == "test":
                row[metric] = dfx[metric].values[0] if metric in dfx.columns else np.nan
            else:
                row[metric] = dfx[metric].values[-1] if metric in dfx.columns else np.nan
        resum[stg] = row

    df_resum = pd.DataFrame(resum).T
    df_resum.index.name = "Conjunt"
    # Sempre guardem el CSV
    df_resum.to_csv(os.path.join(output_dir, "resum_metrics.csv"))

    # I només intentem markdown/latex si hi ha tabulate
    try:
        import tabulate  # comprova si està instal·lat
        df_resum.round(4).to_markdown(
            index=True,
            tablefmt="github",
            buf=open(os.path.join(output_dir, "resum_metrics.md"), "w")
        )
        df_resum.round(4).to_latex(
            index=True,
            buf=open(os.path.join(output_dir, "resum_metrics.tex"), "w")
        )
    except ImportError:
        print("tabulate no disponible: només s'ha creat resum_metrics.csv")

    return df_resum

print(f"Gràfics i taules generats a: {os.path.abspath(output_dir)}")