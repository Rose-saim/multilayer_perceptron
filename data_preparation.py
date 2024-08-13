# data_preparation.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from PIL import Image

class PytorchData(Dataset):
    def __init__(self, data_dir, transform, data_type="train"):
        cdm_data = os.path.join(data_dir, data_type)  # directory of files
        file_names = os.listdir(cdm_data)
        idx_choose = np.random.choice(np.arange(len(file_names)), 4000, replace=False).tolist()
        file_names_sample = [file_names[x] for x in idx_choose]
        self.full_filenames = [os.path.join(cdm_data, f) for f in file_names_sample]

        labels_data = os.path.join(data_dir, "train_labels.csv")
        labels_df = pd.read_csv(labels_data)
        labels_df.set_index("id", inplace=True)
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in file_names_sample]
        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])
        image = self.transform(image)
        return image, self.labels[idx]

def preprocess_data(df):
    # Ensure that 'id' and 'diagnosis' columns are dropped properly
    features = df.drop(columns=['id', 'diagnosis']).values
    labels = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0).values
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def prepare_data():
    # Load and preprocess data
    labels_df = pd.read_csv('./data.csv', header=None)
    labels_df.columns = ['id', 'diagnosis'] + [f'feature{i}' for i in range(1, 31)]
    
    # Remove duplicate rows
    labels_df.drop_duplicates(inplace=True)

    # Split data into training and validation sets
    train_df, val_df = train_test_split(labels_df, test_size=0.1, random_state=42)

    # Preprocess datasets
    train_features, train_labels = preprocess_data(train_df)
    val_features, val_labels = preprocess_data(val_df)
    
    # Create DataLoader instances
    batch_size = 32
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

if __name__ == "__main__":
    prepare_data()