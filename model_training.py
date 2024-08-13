import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from data_preparation import prepare_data
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns; sns.set(style='whitegrid')

# Définir le réseau neuronal
class TabularNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TabularNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)
        x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.dropout(x)
        x = self.fc3(x)
        return x # Assurez-vous que la sortie a la forme [batch_size]


# Function to compute the loss value per batch of data
def loss_batch(loss_func, output, target, opt=None):
    target = target.float().unsqueeze(1)  # Assurez-vous que la cible a la forme [batch_size, 1]
    loss = loss_func(output, target)  # Calculer la perte
    pred = (output > 0).float()  # Convertir les sorties en prédictions binaires
    metric_b = pred.eq(target.view_as(pred)).sum().item()  # Calculer la métrique

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b

# Fonction pour calculer la perte et les métriques pour une époque
def loss_epoch(model, loss_func, dataloader, optimizer=None, device=None):
    running_loss = 0.0
    t_metric = 0
    len_data = len(dataloader.dataset)

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        if optimizer:
            optimizer.zero_grad()

        output = model(inputs)
        loss_b, metric_b = loss_batch(loss_func, output, labels, optimizer)  # get loss per batch

        running_loss += loss_b

        if t_metric is not None:
            t_metric += metric_b

    epoch_loss = running_loss / len(dataloader.dataset)
    metric = t_metric / len(dataloader.dataset)

    return epoch_loss, metric

def train_val(model, params, verbose=False, patience=10):
    epochs = params["epochs"]
    loss_func = params["f_loss"]
    opt = params["optimiser"]
    train_dl = params["train"]
    val_dl = params["val"]
    lr_scheduler = params["lr_change"]
    weight_path = params["weight_path"]
    device = next(model.parameters()).device

    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        ''' Get the Learning Rate '''
        current_lr = lr_scheduler.get_last_lr()[0]
        if verbose:
            print(f'Epoch {epoch+1}/{epochs}, current lr={current_lr:.6f}')
        
        '''Train Model Process'''
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt, device)
        
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        '''Evaluate Model Process'''
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, device=device)
        
        # store best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), weight_path)
            epochs_without_improvement = 0
            if verbose:
                print("Copied best model weights!")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping after {epochs_without_improvement} epochs without improvement.")
            break

        # collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        # learning rate schedule
        lr_scheduler.step(val_loss)
        new_lr = lr_scheduler.get_last_lr()[0]
        if current_lr != new_lr:
            if verbose:
                print("Loading best model weights due to LR change!")
            model.load_state_dict(best_model_wts) 

        if verbose:
            print(f"Train loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}, Train Metric: {train_metric:.2f}, Validation Metric: {val_metric:.2f}")
            print("-" * 30)
    
    model.load_state_dict(best_model_wts)

    # Sauvegarder les historiques de perte et de métrique
    with open('loss_hist.json', 'w') as f:
        json.dump(loss_history, f)
    with open('metric_hist.json', 'w') as f:
        json.dump(metric_history, f)

    return model, loss_history, metric_history

import torch
import numpy as np

def inference(model, dataset, device):
    model.eval()  # Mise en mode évaluation
    y_out = torch.zeros(len(dataset))  # Pré-allocation pour les sorties
    y_gt = torch.zeros(len(dataset))  # Pré-allocation pour les cibles
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataset):
            inputs = inputs.to(device)
            outputs = model(inputs).cpu()  # Assurez-vous que les sorties sont sur le CPU
            
            # Convertir les sorties en probabilités si nécessaire
            probs = torch.sigmoid(outputs).numpy()  # Si vous utilisez des logits
            
            y_out[i] = torch.tensor(probs)  # Convertir en tensor avant l'assignation
            y_gt[i] = labels.cpu()  # Assurez-vous que les labels sont sur le CPU
    
    return y_out, y_gt


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_history(loss_hist, metric_hist, epochs):
    # S'assurer que les longueurs correspondent
    train_len = len(loss_hist["train"])
    val_len = len(loss_hist["val"])

    # Tronquer les listes pour qu'elles aient la même longueur
    if train_len > val_len:
        loss_hist["train"] = loss_hist["train"][:val_len]
        metric_hist["train"] = metric_hist["train"][:val_len]
    elif val_len > train_len:
        loss_hist["val"] = loss_hist["val"][:train_len]
        metric_hist["val"] = metric_hist["val"][:train_len]

    # Ploter les courbes de convergence
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.lineplot(x=[*range(1, len(loss_hist["train"]) + 1)], y=loss_hist["train"], ax=ax[0], label='Train Loss')
    sns.lineplot(x=[*range(1, len(loss_hist["val"]) + 1)], y=loss_hist["val"], ax=ax[0], label='Validation Loss')
    sns.lineplot(x=[*range(1, len(metric_hist["train"]) + 1)], y=metric_hist["train"], ax=ax[1], label='Train Metric')
    sns.lineplot(x=[*range(1, len(metric_hist["val"]) + 1)], y=metric_hist["val"], ax=ax[1], label='Validation Metric')
    
    plt.title('Convergence History')
    plt.show()

def main():
    train_loader, val_loader = prepare_data()
    
    input_dim = 30  # Exemple de dimension d'entrée
    hidden_dim = 128
    output_dim = 1  # Sortie binaire

    model = TabularNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    params_train = {
        "train": train_loader, "val": val_loader,
        "epochs": 50,
        "optimiser": optimizer,
        "lr_change": StepLR(optimizer, step_size=30, gamma=0.1),
        "f_loss": criterion,
        "weight_path": "weights.pt",
    }

    cnn_model, loss_hist, metric_hist = train_val(model, params_train, verbose=True, patience=20)

    # Ploter l'historique de l'entraînement
    plot_history(loss_hist, metric_hist, epochs=params_train["epochs"])

    # Charger le meilleur modèle pour l'inférence
    model.load_state_dict(torch.load(params_train["weight_path"]))
    
    # Effectuer l'inférence sur le jeu de validation
    y_out, y_gt = inference(model, val_loader.dataset, device)

    # Convertir les probabilités en prédictions binaires
    y_test_pred = (y_out > 0.5).int()

    print("Predicted classes (binary):", y_test_pred.flatten())
    print("Predicted probabilities:", y_out.flatten())

    # Sauvegarder les prédictions
    np.save("predictions.npy", y_out)
    np.save("ground_truth.npy", y_gt)
    
    return loss_hist, metric_hist

if __name__ == "__main__":
    main()
