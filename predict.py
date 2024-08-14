# predict.py
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from joblib import load
from data_preparation import data_manipulation, safe_opener  # Ensure data_preprocessing.py is in the same directory
from utils import forward, predict_
import matplotlib.pyplot as plt

def binary_cross_entropy(probas, y):
    """
    Calcule l'erreur binaire de cross-entropie.

    :param probas: Probabilités prédites pour la classe positive.
    :param y: Étiquettes réelles.
    :return: La perte d'entropie croisée binaire.
    """
    # Assurer que les probabilités ne sont pas exactement 0 ou 1 pour éviter les erreurs log
    probas = np.clip(probas, 1e-15, 1 - 1e-15)
    loss = -np.mean(y * np.log(probas) + (1 - y) * np.log(1 - probas))
    return loss

def predict_probas(network, X):
    logits = forward(network, X)[-1]
    return logits

def cross_entropy_loss(probas, y):
    log_likelihood = -np.log(probas[range(y.shape[0]), y])
    loss = np.sum(log_likelihood) / y.shape[0]
    return loss

def predict():
    model_file = "model.joblib"
    pred_files = {
        'X_pred': './processed_data/X_pred.npy',
        'y_pred': './processed_data/y_pred.npy'
    }

    # Charger les données de prédiction
    X = np.load(pred_files['X_pred'])
    y = np.load(pred_files['y_pred'])

    try:
        # Charger le modèle sauvegardé
        network = load(model_file)
    except Exception as e:
        print(f"Can't load the model file {model_file}: {e}")
        raise

    # Effectuer les prédictions
    probas = predict_probas(network, X)
    preds = predict_(network, X)

    # Calculer la matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    print('\nConfusion Matrix:\n', confusion_matrix(y, preds))
    print(f'Accuracy: {((tn + tp) / y.shape[0]) * 100:.4f}%')

    # Extraire les probabilités pour la classe positive (1) pour ROC AUC
    # On suppose que la classe positive est la classe 1, donc la colonne 1 des probabilités
    probas_positive_class = probas[:, 1]
    
    # Calculer et afficher le ROC AUC Score
    try:
        roc_auc = roc_auc_score(y, probas_positive_class)
        print(f'ROC AUC Score: {roc_auc:.2f}')
    except ValueError as e:
        print(f"Error calculating ROC AUC Score: {e}")

    # Calculer et afficher la perte d'entropie croisée
    print(f'Cross Entropy Loss: {binary_cross_entropy(probas_positive_class, y) * 100:.4f}\n')

    # Optionnel: Afficher la matrice de confusion sous forme de graphique
    plt.figure(figsize=(8, 6))
    plt.title('Confusion Matrix')
    plt.imshow(confusion_matrix(y, preds), interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    classes = np.unique(y)  # Assurez-vous que les classes sont bien définies ici
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == "__main__":
    predict()