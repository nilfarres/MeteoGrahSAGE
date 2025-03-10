import os, glob                                            # Per gestionar directoris i cercar fitxers dins d'una estructura de carpetes
import torch                                               # Per treballar amb tensors i desar els objectes Data convertits a fitxers .pt
import torch.nn as nn                                      # Per definir models de xarxes neuronals
import torch.nn.functional as F                            # Per funcions comunes de xarxes neuronals
import torch.optim as optim                                # Per optimitzadors com Adam
from torch_geometric.data import DataLoader                # Per carregar dades en forma de mini-batches
from torch_geometric.nn import SAGEConv, global_mean_pool  # Per capes SAGEConv i global_mean_pool
from tqdm import tqdm                                      # Per mostrar una barra de progrés durant l'entrenament
import argparse                                            # Per llegir arguments de la línia de comandes

# =============================================================================
# Classe per carregar els grafos convertits (cada graf és un fitxer .pt)
# =============================================================================
class MeteoGraphDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split="train"):
        """
        Args:
            root_dir (str): Directori arrel on es troben els fitxers .pt (resultat de toData.py)
            split (str): "train" per a entrenament (anys 2016-2022) o "test" per a validació (anys 2023-2024)
        """
        # Cerquem recursivament tots els fitxers .pt dins del directori
        all_paths = glob.glob(os.path.join(root_dir, '**', '*.pt'), recursive=True)
        filtered_paths = []
        for fp in all_paths:
            # Extraiem la ruta relativa respecte al directori arrel
            rel = os.path.relpath(fp, root_dir)
            parts = rel.split(os.sep)
            if len(parts) == 0:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                continue
            # Divisió temporal: entrenament amb anys 2016-2022, validació amb anys 2023-2024
            if split == "train" and (2016 <= year <= 2022):
                filtered_paths.append(fp)
            elif split == "test" and (2023 <= year <= 2024):
                filtered_paths.append(fp)
        self.file_paths = filtered_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Carreguem l'objecte Data a partir del fitxer
        file_path = self.file_paths[idx]
        data = torch.load(file_path)
        # Definim com a target global la mitjana de cada variable meteorològica del graf.
        # Les variables a predir són: Temp, Humitat, Pluja, VentDir, VentFor i Patm.
        global_target = data.x.mean(dim=0)  # Tensor de forma [6]
        data.y = global_target  # Afegim aquest target global
        # També es pot utilitzar data.x com a target a nivell de node.
        return data

# =============================================================================
# Definició del model MeteoGraphSAGE
# =============================================================================
class MeteoGraphSAGE(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=32, out_channels=6, num_layers=2, dropout=0.5):
        """
        Args:
            in_channels (int): Nombre de característiques d'entrada per node (6 variables).
            hidden_channels (int): Nombre de canals en les capes ocultes.
            out_channels (int): Dimensió de la sortida (6, per predir totes les variables).
            num_layers (int): Nombre de capes SAGEConv.
            dropout (float): Taxa de dropout.
        """
        super(MeteoGraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        # Primera capa: de l'entrada al primer espai ocult
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        # Capes intermèdies
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        
        # "Node head": capa lineal per a la predicció a nivell de node (per cada estació)
        self.node_head = nn.Linear(hidden_channels, out_channels)
        # "Graph head": capa lineal per a la predicció global (després del pooling)
        self.graph_head = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, data):
        """
        Args:
            data: Un Batch de Data objects amb atributs x, edge_index i batch.
        Retorna:
            node_pred: Prediccions a nivell de node (tensor de forma [N, 6]).
            graph_pred: Predicció global a nivell de graf (tensor de forma [num_graphs, 6]).
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Predicció a nivell de node: cada node té un vector de sortida
        node_pred = self.node_head(x)
        # Agrupació per obtenir la representació global del graf (global mean pooling)
        graph_embedding = global_mean_pool(x, batch)
        graph_pred = self.graph_head(graph_embedding)
        return node_pred, graph_pred

# =============================================================================
# Funcions d'entrenament i validació
# =============================================================================
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Entrenant", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        # Obtenim les prediccions: a nivell de node i a nivell de graf
        node_pred, graph_pred = model(data)
        # Loss per a la predicció a nivell de node: comparem amb les dades originals (data.x)
        node_loss = criterion(node_pred, data.x.to(device))
        # Loss per a la predicció global: comparem amb el target global (data.y)
        graph_loss = criterion(graph_pred, data.y.to(device))
        loss = node_loss + graph_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            node_pred, graph_pred = model(data)
            node_loss = criterion(node_pred, data.x.to(device))
            graph_loss = criterion(graph_pred, data.y.to(device))
            loss = node_loss + graph_loss
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# =============================================================================
# Bloc principal d'entrenament
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Entrenament de MeteoGraphSAGE")
    parser.add_argument('--data_dir', type=str, default="F:/DADES_METEO_PC_GRAPH",
                        help="Directori on es troben els fitxers .pt convertits")
    parser.add_argument('--epochs', type=int, default=50, help="Nombre d'èpoques")
    parser.add_argument('--batch_size', type=int, default=32, help="Mida del batch")
    parser.add_argument('--lr', type=float, default=0.001, help="Taxa d'aprenentatge")
    args = parser.parse_args()

    # Definim el dispositiu (GPU si està disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carreguem els datasets: entrenament amb anys 2016-2022 i validació amb anys 2023-2024
    train_dataset = MeteoGraphDataset(args.data_dir, split="train")
    test_dataset = MeteoGraphDataset(args.data_dir, split="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Inicialitzem el model: l'entrada són 6 característiques per node i la sortida és un vector de 6 per cada tasca.
    model = MeteoGraphSAGE(in_channels=6, hidden_channels=32, out_channels=6, num_layers=2, dropout=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print("Iniciant l'entrenament de MeteoGraphSAGE...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss = test(model, test_loader, criterion, device)
        print(f"Època: {epoch:03d}, Pèrdua Entrenament: {train_loss:.4f}, Pèrdua Test: {test_loss:.4f}")

    # Desa el model entrenat
    torch.save(model.state_dict(), "MeteoGraphSAGE.pth")
    print("Model desat com a MeteoGraphSAGE.pth")

if __name__ == "__main__":
    main()
