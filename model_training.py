import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from data_preparation import prepare_data
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Définir le réseau neuronal
class TabularNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TabularNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.sigmoid(x)

# Fonction pour calculer la perte et les métriques pour une époque
def loss_epoch(model, loss_func, dataloader, optimizer=None, device=None):
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        if optimizer:
            optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        if optimizer:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_labels.extend(labels.cpu().numpy())
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, accuracy, precision, recall, f1
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
        current_lr = lr_scheduler.get_last_lr()[0]
        if verbose:
            print(f'Epoch {epoch+1}/{epochs}, current lr={current_lr:.6f}')
        
        model.train()
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = loss_epoch(model, loss_func, train_dl, opt, device)
        
        loss_history["train"].append(train_loss)
        metric_history["train"].append({
            "accuracy": train_accuracy,
            "precision": train_precision,
            "recall": train_recall,
            "f1": train_f1
        })
        
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy, val_precision, val_recall, val_f1 = loss_epoch(model, loss_func, val_dl, device=device)
        
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
        
        loss_history["val"].append(val_loss)
        metric_history["val"].append({
            "accuracy": val_accuracy,
            "precision": val_precision,
            "recall": val_recall,
            "f1": val_f1
        })
        
        lr_scheduler.step(val_loss)
        new_lr = lr_scheduler.get_last_lr()[0]
        if current_lr != new_lr:
            if verbose:
                print("Loading best model weights due to LR change!")
            model.load_state_dict(best_model_wts) 

        if verbose:
            print(f"Train loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}, Accuracy: {100 * val_accuracy:.2f}%, Precision: {train_precision:.2f}, Recall: {train_recall:.2f}, F1 Score: {train_f1:.2f}")
            print("-" * 30)
    
    model.load_state_dict(best_model_wts)

    # Sauvegarder les historiques de perte et de métrique
    with open('loss_hist.json', 'w') as f:
        json.dump(loss_history, f)
    with open('metric_hist.json', 'w') as f:
        json.dump(metric_history, f)

    return model, loss_history, metric_history

# Fonction d'inférence
def inference(model, dataset, device, num_classes=2):
    len_data = len(dataset)
    y_out = torch.zeros(len_data, num_classes)  # Initialiser le tenseur de sortie
    y_gt = np.zeros(len_data, dtype="uint8")  # Initialiser le tableau de vérité terrain
    
    model = model.to(device)  # Déplacer le modèle vers l'appareil
    
    with torch.no_grad():
        for i in tqdm(range(len_data)):
            x, y = dataset[i]
            y_gt[i] = y
            x = x.unsqueeze(0).to(device)  # Ajouter la dimension de lot et déplacer vers l'appareil
            y_out[i] = model(x)
    
    return y_out.numpy(), y_gt

def plot_learning_curves(loss_history, metric_history):
    epochs = range(1, len(loss_history["train"]) + 1)
    
    plt.figure(figsize=(16, 10))

    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, loss_history["train"], 'b', label='Train Loss')
    if len(loss_history["val"]) == len(epochs):
        plt.plot(epochs, loss_history["val"], 'r', label='Validation Loss')
    else:
        min_len = min(len(loss_history["train"]), len(loss_history["val"]))
        plt.plot(epochs[:min_len], loss_history["val"][:min_len], 'r', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [m["accuracy"] for m in metric_history["train"]], 'b', label='Train Accuracy')
    if len(metric_history["val"]) == len(epochs):
        plt.plot(epochs, [m["accuracy"] for m in metric_history["val"]], 'r', label='Validation Accuracy')
    else:
        min_len = min(len(metric_history["train"]), len(metric_history["val"]))
        plt.plot(epochs[:min_len], [m["accuracy"] for m in metric_history["val"]][:min_len], 'r', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Precision
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [m["precision"] for m in metric_history["train"]], 'b', label='Train Precision')
    if len(metric_history["val"]) == len(epochs):
        plt.plot(epochs, [m["precision"] for m in metric_history["val"]], 'r', label='Validation Precision')
    else:
        min_len = min(len(metric_history["train"]), len(metric_history["val"]))
        plt.plot(epochs[:min_len], [m["precision"] for m in metric_history["val"]][:min_len], 'r', label='Validation Precision')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    # Plot Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [m["recall"] for m in metric_history["train"]], 'b', label='Train Recall')
    if len(metric_history["val"]) == len(epochs):
        plt.plot(epochs, [m["recall"] for m in metric_history["val"]], 'r', label='Validation Recall')
    else:
        min_len = min(len(metric_history["train"]), len(metric_history["val"]))
        plt.plot(epochs[:min_len], [m["recall"] for m in metric_history["val"]][:min_len], 'r', label='Validation Recall')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()


def main():
    train_loader, val_loader = prepare_data()
    
    input_dim = 30  # Dimension d'entrée exemple
    hidden_dim = 128
    output_dim = 2
    
    model = TabularNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    params_train = {
        "train": train_loader, "val": val_loader,
        "epochs": 100,
        "optimiser": optimizer,
        "lr_change": ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20),
        "f_loss": criterion,
        "weight_path": "weights.pt",
    }

    cnn_model, loss_hist, metric_hist = train_val(model, params_train, verbose=True, patience=10)
    
    # Charger le meilleur modèle pour l'inférence
    model.load_state_dict(torch.load(params_train["weight_path"]))
    
    # Effectuer l'inférence sur le jeu de validation
    y_out, y_gt = inference(model, val_loader.dataset, device)

    # Class predictions 0,1
    y_test_pred = np.argmax(y_out, axis=1)
    print(y_test_pred.shape)
    print(y_test_pred[0:5])

    # Probabilités de sélection prédites
    preds = np.exp(y_out[:, 1])
    print(preds.shape)
    print(preds[0:5])

    # Sauvegarder les prédictions
    np.save("predictions.npy", y_out)
    np.save("ground_truth.npy", y_gt)
    
    # Tracer les courbes d'apprentissage
    plot_learning_curves(loss_hist, metric_hist)

    return loss_hist, metric_hist

if __name__ == "__main__":
    main()
