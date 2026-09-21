import random
import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from data_preprocessing import X, y, preprocessing1, preprocessing2, preprocessing3, preprocessing4
from tqdm import tqdm

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers_size = [input_size]+hidden_sizes+[output_size]
        layers = []
        for i in range(len(layers_size)-2):
            layers.append(nn.Linear(layers_size[i], layers_size[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layers_size[-2], layers_size[-1]))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

def train(params, X_train, y_train, X_val, y_val):
    X_train_pre = params["preprocessing"].fit_transform(X_train)
    X_val_pre = params["preprocessing"].transform(X_val)
    X_train_tensor = torch.tensor(X_train_pre, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_pre, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
    train_set = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_set, batch_size=params["batch_size"])
    val_set = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_set, batch_size=params["batch_size"])

    input_s = len(X_train_tensor[0])
    model = MLP(input_s, params["hidden_sizes"], 1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    patience = 50
    epochs_wo_improv = 0

    best_epoch = 0
    best_loss = math.inf
    best_outputs = []
    for epoch in range(params["epochs"]):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).flatten()
            loss = criterion(outputs, y_batch)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_train_loss /= len(train_loader)
        
        model.eval()
        epoch_val_loss = 0
        total_outputs = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch).flatten()
                probas = torch.sigmoid(outputs).cpu().numpy()
                total_outputs.append(probas)
                loss = criterion(outputs, y_batch)
                epoch_val_loss += loss.item()
        epoch_val_loss /= len(val_loader)
        total_outputs = np.concatenate(total_outputs)
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_epoch = epoch+1
            best_outputs = total_outputs
            epochs_wo_improv = 0
        else:
            epochs_wo_improv += 1
        """if (epoch+1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{params["epochs"]}: train loss = {epoch_train_loss} / val loss = {epoch_val_loss}")"""
        if epochs_wo_improv >= patience:
            break
    return best_loss, best_outputs, best_epoch

param_grid = {
    "preprocessing": [preprocessing1, preprocessing2, preprocessing3, preprocessing4],
    "epochs": [2000],
    "learning_rate": [1e-3],
    "hidden_sizes": [[8], [16], [32], [8, 8], [16, 16]],
    "batch_size": [16, 32, 64],
    "weight_decay": [0, 1e-5, 1e-4, 1e-3]
}

def random_params(param_grid):
    return {key: random.choice(val) for key, val in param_grid.items()}

def random_search(nb_iter):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=42)
    best_result = math.inf
    for iter in range(nb_iter):
        params = random_params(param_grid)
        print(f"Params: lr = {params["learning_rate"]} - hidden sizes = {params["hidden_sizes"]} - batch_size = {params["batch_size"]} - weight decay = {params["weight_decay"]}")
        result, outputs, epoch = train(params, X_train, y_train, X_val, y_val)
        print(f"Result: {result} / Best epoch {epoch}")
        if result < best_result:
            best_result = result
    return best_result

if __name__ == "__main__":
    random_search(3)