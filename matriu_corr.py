import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Paths
root_dir = r'C:/Users/nfarres/Documents/TFG/models/model_curt_termini'
out_dir = os.path.join(root_dir, 'plots_metriques')
os.makedirs(out_dir, exist_ok=True)

# Variables (adapta segons el teu model!)
variables = [
    "Temp", "Humitat", "Pluja", "VentFor", "Patm", "Vent_u", "Vent_v"
]

def save_correlation_matrix(npfile, label):
    # Carrega les dades
    data = np.load(npfile)
    S, H, N, F = data.shape
    variables_used = variables[:F]
    
    data_flat = data.reshape(-1, F)
    data_flat = data_flat[~np.isnan(data_flat).any(axis=1)]
    
    corr_matrix = np.corrcoef(data_flat, rowvar=False)
    
    # Desa com a CSV
    df_corr = pd.DataFrame(corr_matrix, index=variables_used, columns=variables_used)
    csv_path = os.path.join(out_dir, f'matriu_correlacio_{label}.csv')
    df_corr.to_csv(csv_path)
    
    # Dibuixa i desa la figura
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                xticklabels=variables_used, yticklabels=variables_used, vmin=-1, vmax=1)
    plt.title(f'Matriu de correlació ({label})')
    plt.tight_layout()
    png_path = os.path.join(out_dir, f'matriu_correlacio_{label}.png')
    plt.savefig(png_path)
    plt.close()
    print(f"Matriu de correlació desada a: {csv_path} i {png_path}")

# Calcula per a les dades reals
save_correlation_matrix(os.path.join(root_dir, 'C:/Users/nfarres/Documents/TFG/models/model_curt_termini/y_true_test.npy'), 'real')

# Calcula per a les prediccions
save_correlation_matrix(os.path.join(root_dir, 'C:/Users/nfarres/Documents/TFG/models/model_curt_termini/y_pred_test.npy'), 'prediccio')