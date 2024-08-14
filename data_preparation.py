import pandas as pd
import chardet
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from training import safe_opener

# --- Gestion des données ---
def data_manipulation(data, prediction):
    """
    Manipule les données et prépare les ensembles d'entraînement et de test.
    """
    print("Aperçu des données:")
    print(data.head())
    print("Colonnes disponibles:")
    print(data.columns)

    data = data.replace(',', '.', regex=True)
    data = data.replace('B', '0', regex=True)
    data = data.replace('M', '1', regex=True)

    if not prediction:
        if data.shape[1] < 2:
            raise IndexError("Le DataFrame n'a pas suffisamment de colonnes. Vérifiez le fichier CSV.")

        y = data.iloc[:, 1]
        X = data.drop(data.columns[1], axis=1)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        # Diviser l'ensemble d'entraînement + validation en entraînement et validation
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42)

        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train.values)
        X_val = min_max_scaler.transform(X_val)
        X_test = min_max_scaler.transform(X_test.values)
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        return X_train.astype(float), X_val.astype(float), y_train.astype(int), y_val.astype(int), X_test.astype(float), y_test.astype(int)
    else:
        if data.shape[1] < 2:
            raise IndexError("Le DataFrame n'a pas suffisamment de colonnes. Vérifiez le fichier CSV.")

        y = data.iloc[:, 1]
        X = data.drop(data.columns[1], axis=1)
        y = y.to_numpy().astype(int)

        min_max_scaler = MinMaxScaler()
        X = min_max_scaler.fit_transform(X.values)
        return X.astype(float), y

def save_preprocessed_data(data_path, output_dir='./processed_data', prediction=False):
    """
    Sauvegarde les données prétraitées dans des fichiers distincts et retourne les chemins de ces fichiers.

    :param data_path: Chemin vers le fichier CSV d'origine.
    :param output_dir: Dossier où les données prétraitées seront sauvegardées.
    :param prediction: Booléen indiquant si les données sont utilisées pour la prédiction (True) ou l'entraînement (False).
    :return: Un dictionnaire contenant les chemins des fichiers sauvegardés.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = safe_opener(data_path)
    
    saved_files = {}

    if not prediction:
        X_train, X_val, y_train, y_val, X_test, y_test = data_manipulation(data, False)
        
        X_train_path = os.path.join(output_dir, 'X_train.npy')
        X_val_path = os.path.join(output_dir, 'X_val.npy')
        X_test_path = os.path.join(output_dir, 'X_test.npy')
        y_train_path = os.path.join(output_dir, 'y_train.npy')
        y_val_path = os.path.join(output_dir, 'y_val.npy')
        y_test_path = os.path.join(output_dir, 'y_test.npy')
                
        np.save(X_train_path, X_train)
        np.save(X_val_path, X_val)
        np.save(X_test_path, X_test)
        np.save(y_train_path, y_train)
        np.save(y_val_path, y_val)
        np.save(y_test_path, y_test)

        saved_files['X_train'] = X_train_path
        saved_files['X_val'] = X_val_path
        saved_files['X_test'] = X_test_path
        saved_files['y_train'] = y_train_path
        saved_files['y_val'] = y_val_path
        saved_files['y_test'] = y_test_path
        
        print(f'Data for training and testing saved in {output_dir}')
        
    else:
        X_path = os.path.join(output_dir, 'X_pred.npy')
        y_path = os.path.join(output_dir, 'y_pred.npy')

        X, y = data_manipulation(data, True)
        
        np.save(X_path, X)
        np.save(y_path, y)

        saved_files['X_pred'] = X_path
        saved_files['y_pred'] = y_path
        
        print(f'Data for prediction saved in {output_dir}')
    
    return saved_files

if __name__ == '__main__':
    # Pour sauvegarder les données d'entraînement et de test
    training_files = save_preprocessed_data('./data.csv')

    # Pour sauvegarder les données de prédiction
    prediction_files = save_preprocessed_data('./data.csv', prediction=True)

