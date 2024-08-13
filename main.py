import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import copy


import matplotlib.pyplot as plt
torch.manual_seed(0) # fix random seed

class pytorch_data(Dataset):
    
    def __init__(self,data_dir,transform,data_type="train"):      
    
        # Get Image File Names
        cdm_data=os.path.join(data_dir,data_type)  # directory of files
        
        file_names = os.listdir(cdm_data) # get list of images in that directory  
        idx_choose = np.random.choice(np.arange(len(file_names)), 
                                      4000,
                                      replace=False).tolist()
        file_names_sample = [file_names[x] for x in idx_choose]
        self.full_filenames = [os.path.join(cdm_data, f) for f in file_names_sample]   # get the full path to images
        
        # Get Labels
        labels_data=os.path.join(data_dir,"train_labels.csv") 
        labels_df=pd.read_csv(labels_data)
        labels_df.set_index("id", inplace=True) # set data frame index to id
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in file_names_sample]  # obtained labels from df
        self.transform = transform
      
    def __len__(self):
        return len(self.full_filenames) # size of dataset
      
    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])  # Open Image with PIL
        image = self.transform(image) # Apply Specific Transformation to Image
        return image, self.labels[idx]
    
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os

def plot_line(df):
    df['id'].hist()
    plt.title('Histogramme de column_name')
    plt.xlabel('Valeurs')
    plt.ylabel('Fréquence')
    plt.show()

# Chargement du fichier CSV sans en-têtes
labels_df = pd.read_csv('./data.csv', header=None)

# Définir les noms des colonnes manuellement
labels_df.columns = ['id', 'diagnosis', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14', 'feature15', 'feature16', 'feature17', 'feature18', 'feature19', 'feature20', 'feature21', 'feature22', 'feature23', 'feature24', 'feature25', 'feature26', 'feature27', 'feature28', 'feature29', 'feature30']

labels_df.head().to_markdown()
os.listdir('./')
# No duplicate ids found
labels_df[labels_df.duplicated(keep=False)]
# Vérifiez si le DataFrame des doublons n'est pas vide
duplicated_rows = labels_df[labels_df.duplicated(keep=False)]
if not duplicated_rows.empty:
    labels_df.drop_duplicates(inplace=True)

imgpath ="./data.csv" # training data is stored in this folder
malignant = labels_df.loc[labels_df['diagnosis']=='M']['id'].values    # get the ids of malignant cases
normal = labels_df.loc[labels_df['diagnosis']=='B']['id'].values       # get the ids of the normal cases
# plot_fig(malignant,'Malignant Cases')

# Créez un DataFrame à partir des IDs des cas malins
malignant_df = labels_df[labels_df['id'].isin(malignant)]
normal_df = labels_df[labels_df['id'].isin(normal)]
# plot_line(malignant_df)
# plot_line(normal_df)

# Diviser le DataFrame en ensembles d'entraînement et de validation
train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)

# Afficher les tailles des ensembles

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Fonction pour préparer les données en tenseurs PyTorch
def preprocess_data(df):
    features = df.drop(columns=['id', 'diagnosis']).values  # Exclure 'id' et 'diagnosis'
    labels = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0).values  # Convertir 'diagnosis' en 0 et 1
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Préparer les données
features, labels = preprocess_data(labels_df)

# Diviser les données en ensembles d'entraînement et de validation
train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

# Créer les TensorDatasets
train_dataset = TensorDataset(train_features, train_labels)
val_dataset = TensorDataset(val_features, val_labels)

# Définir la taille des lots
batch_size = 32

# Créer les DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

import torch.nn as nn
import torch.nn.functional as F

# Définition du modèle
class TabularNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TabularNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Instanciation du modèle
input_dim = train_features.shape[1]  # Nombre de features
hidden_dim = 128
output_dim = 2  # Deux classes: Malignant (1) et Benign (0)


# Fonction d'entraînement/évaluation
def loss_epoch(model, loss_func, dataloader, optimizer=None, device=None):
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)  # Assurez-vous que les données sont sur le bon device

        # Vérification des types
        if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("Inputs and labels should be tensors.")

        if optimizer:
            optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        if optimizer:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total

    return epoch_loss, accuracy
def train_val(model, params, verbose=False):
    epochs = params["epochs"]
    loss_func = params["f_loss"]
    opt = params["optimiser"]
    train_dl = params["train"]
    val_dl = params["val"]
    lr_scheduler = params["lr_change"]
    weight_path = params["weight_path"]
    device = next(model.parameters()).device  # Get device from model

    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in tqdm(range(epochs), desc='Training Progress', unit='epoch'):
        current_lr = lr_scheduler.get_last_lr()[0]
        if verbose:
            print(f'Epoch {epoch+1}/{epochs}, current lr={current_lr:.6f}')
        
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt, device)
        
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, device=device)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), weight_path)
            if verbose:
                print("Copied best model weights!")
        
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        lr_scheduler.step(val_loss)
        new_lr = lr_scheduler.get_last_lr()[0]
        if current_lr != new_lr:
            if verbose:
                print("Loading best model weights due to LR change!")
            model.load_state_dict(best_model_wts) 

        if verbose:
            print(f"Train loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}, Accuracy: {100 * val_metric:.2f}%")
            print("-" * 30)
    
    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history


def main():
    model = TabularNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    params_train = {
        "train": train_loader, "val": val_loader,
        "epochs": 50,
        "optimiser": optimizer,
        "lr_change": ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20),
        "f_loss": nn.NLLLoss(reduction="sum"),
        "weight_path": "weights.pt",
    }

    cnn_model, loss_hist, metric_hist = train_val(model, params_train, verbose=True)

    import seaborn as sns
    sns.set(style='whitegrid')

    epochs = params_train["epochs"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist["train"],ax=ax[0],label='loss_hist["train"]')
    sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist["val"],ax=ax[0],label='loss_hist["val"]')
    sns.lineplot(x=[*range(1,epochs+1)],y=metric_hist["train"],ax=ax[1],label='metric_hist["train"]')
    sns.lineplot(x=[*range(1,epochs+1)],y=metric_hist["val"],ax=ax[1],label='metric_hist["val"]')
    ax[0].set_title('Loss History')
    ax[1].set_title('Accuracy History')
    plt.title('Convergence History')
    plt.show()

# if __name__ == "__main__":
#     main()


#     params_train = {
#         "train": train_loader, "val": val_loader,
#         "epochs": 50,
#         "optimiser": optimizer,
#         "lr_change": ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20),
#         "f_loss": nn.NLLLoss(reduction="sum"),
#         "weight_path": "weights.pt",
#     }

#     cnn_model, loss_hist, metric_hist = train_val(model, params_train)
#     import seaborn as sns; sns.set(style='whitegrid')

#     epochs=params_train["epochs"]

#     fig,ax = plt.subplots(1,2,figsize=(12,5))


#     plt.title('Convergence History')

if __name__ == "__main__":
    main()