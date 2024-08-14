import numpy as np
import pandas as pd
import os
import chardet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from joblib import dump, load
import matplotlib.pyplot as plt
from tqdm import trange
from model_definition import Dense, ReLU, Sigmoid, AdamOptimizer

# --- Fonctions de perte et gradients ---
def softmax_crossentropy(logits, reference_answers):
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]
    xentropy = -logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))
    return xentropy

def grad_softmax_crossentropy(logits, reference_answers):
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    return (-ones_for_answers + softmax) / logits.shape[0]


# --- Entraînement et prédiction ---
def forward(network, X):
    activations = []
    input = X
    for l in network:
        activations.append(l.forward(input))
        input = activations[-1]
    return activations

def train(network, X, y):
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations
    logits = layer_activations[-1]

    loss = softmax_crossentropy(logits, y)
    loss_grad = grad_softmax_crossentropy(logits, y)

    for layer_index in range(len(network) - 1, -1, -1):
        layer = network[layer_index]
        loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)

    return np.mean(loss)

def predict_(network, X):
    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)

def safe_opener(file):
    """
    Fonction utilisée pour ouvrir en toute sécurité le fichier CSV.
    """
    cwd = os.getcwd()
    try:
        with open(os.path.join(cwd, file), 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']
        
        data = pd.read_csv(os.path.join(cwd, file), encoding=encoding, delimiter=',')
        
        print("Colonnes après lecture:")
        print(data.columns)

        return data

    except Exception as e:
        print("Erreur lors de l'ouverture du fichier CSV :", e)
        raise e

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    if shuffle:
        indices = np.random.permutation(len(inputs))
    else:
        indices = np.arange(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]

def train_epoch(network, X, y, batchsize, optimizer=None):
    total_loss = 0
    for x_batch, y_batch in iterate_minibatches(X, y, batchsize=batchsize, shuffle=True):
        loss = train(network, x_batch, y_batch)
        total_loss += loss
        if optimizer:
            # Ici, vous devez mettre à jour les poids du modèle avec l'optimiseur
            pass
    return total_loss / (len(X) / batchsize)

from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_predictions(preds, y):
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)
    return precision, recall, f1

def train_with_optimizer(optimizer, model, X_train, y_train, X_val, y_val, epochs, batchsize):
    train_loss_log = []
    val_loss_log = []
    train_accuracy_log = []
    val_accuracy_log = []

    for epoch in range(epochs):
        train_loss = train_epoch(model, X_train, y_train, batchsize=batchsize, optimizer=optimizer)
        logits_val = forward(model, X_val)[-1]
        val_loss = np.mean(softmax_crossentropy(logits_val, y_val))
        
        preds_train = predict_(model, X_train)
        preds_val = predict_(model, X_val)
        
        train_accuracy = np.mean(preds_train == y_train)
        val_accuracy = np.mean(preds_val == y_val)

        train_loss_log.append(train_loss)
        val_loss_log.append(val_loss)
        train_accuracy_log.append(train_accuracy)
        val_accuracy_log.append(val_accuracy)
        
        print(f"Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")

    return train_loss_log, val_loss_log, train_accuracy_log, val_accuracy_log

def training():
    np.random.seed(42)

    saved_files = {
        'X_train': './processed_data/X_train.npy',
        'X_val': './processed_data/X_val.npy',
        'y_train': './processed_data/y_train.npy',
        'y_val': './processed_data/y_val.npy'
    }

    X_train = np.load(saved_files['X_train'])
    X_val = np.load(saved_files['X_val'])
    y_train = np.load(saved_files['y_train'])
    y_val = np.load(saved_files['y_val'])

    # Configure different optimizers
    optimizers = {
        'SGD': None,  # Remplacer par un vrai optimiseur SGD si nécessaire
        'Adam': AdamOptimizer(lr=0.001)
    }

    results = {}

    for opt_name, optimizer in optimizers.items():
        # Créez une nouvelle instance du réseau pour chaque optimiseur
        network = [
            Dense(X_train.shape[1], 50),
            ReLU(),
            Dense(50, 100),
            ReLU(),
            Dense(100, 2),
            Sigmoid()
        ]

        train_loss_log = []
        val_loss_log = []
        train_accuracy_log = []
        val_accuracy_log = []
        train_precision_log = []
        train_recall_log = []
        train_f1_log = []
        val_precision_log = []
        val_recall_log = []
        val_f1_log = []

        patience = 5
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(10000):
            train_loss = train_epoch(network, X_train, y_train, batchsize=32, optimizer=optimizer)
            logits_val = forward(network, X_val)[-1]
            val_loss = np.mean(softmax_crossentropy(logits_val, y_val))
            
            preds_train = predict_(network, X_train)
            preds_val = predict_(network, X_val)
            
            train_precision, train_recall, train_f1 = evaluate_predictions(preds_train, y_train)
            val_precision, val_recall, val_f1 = evaluate_predictions(preds_val, y_val)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                dump(network, f'best_model_{opt_name}.joblib')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs using {opt_name}.")
                    break

            train_accuracy = np.mean(preds_train == y_train)
            val_accuracy = np.mean(preds_val == y_val)

            train_loss_log.append(train_loss)
            val_loss_log.append(val_loss)
            train_accuracy_log.append(train_accuracy)
            val_accuracy_log.append(val_accuracy)
            train_precision_log.append(train_precision)
            train_recall_log.append(train_recall)
            train_f1_log.append(train_f1)
            val_precision_log.append(val_precision)
            val_recall_log.append(val_recall)
            val_f1_log.append(val_f1)

        results[opt_name] = {
            'train_loss': train_loss_log,
            'val_loss': val_loss_log,
            'train_accuracy': train_accuracy_log,
            'val_accuracy': val_accuracy_log,
            'train_precision': train_precision_log,
            'train_recall': train_recall_log,
            'train_f1': train_f1_log,
            'val_precision': val_precision_log,
            'val_recall': val_recall_log,
            'val_f1': val_f1_log
        }

    # Afficher les courbes
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    for opt_name in results:
        plt.plot(results[opt_name]['train_accuracy'], label=f'{opt_name} Train Accuracy')
        plt.plot(results[opt_name]['val_accuracy'], label=f'{opt_name} Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    for opt_name in results:
        plt.plot(results[opt_name]['train_loss'], label=f'{opt_name} Train Loss')
        plt.plot(results[opt_name]['val_loss'], label=f'{opt_name} Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.title('Training and Validation Loss')

    plt.subplot(2, 2, 3)
    for opt_name in results:
        plt.plot(results[opt_name]['train_precision'], label=f'{opt_name} Train Precision')
        plt.plot(results[opt_name]['val_precision'], label=f'{opt_name} Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    plt.title('Training and Validation Precision')

    plt.subplot(2, 2, 4)
    for opt_name in results:
        plt.plot(results[opt_name]['train_recall'], label=f'{opt_name} Train Recall')
        plt.plot(results[opt_name]['val_recall'], label=f'{opt_name} Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid()
    plt.title('Training and Validation Recall')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    training()