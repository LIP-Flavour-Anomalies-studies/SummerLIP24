"""
Created on August 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import prepdata as prep
import numpy as np
import uproot3 as uproot
import awkward0 as ak

class ClassificationModel(nn.Module):
    def __init__(self, input_size, n_layers, n_units, activation):
        super(ClassificationModel, self).__init__()

        layers = []
        current_iz = input_size

        for i in range(n_layers):
            layers.append(nn.Linear(current_iz, n_units[i]))
            layers.append(activation())
            current_iz = n_units[i]

        layers.append(nn.Linear(current_iz, 1))  # Final layer for binary classification
        layers.append(nn.Sigmoid())  # Sigmoid activation for binary classification

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class BalancedLoss(nn.Module):
    def __init__(self, alpha=None):
        super(BalancedLoss, self).__init__()
        self.alpha = alpha  

    def forward(self, inputs, targets):
        # Calculate the standard binary cross-entropy loss without reduction
        CE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")

        if self.alpha is not None:
            alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)
            B_loss = alpha_t * CE_loss
        else:
            B_loss = CE_loss

        # Return the mean of the balanced cross-entropy loss
        return torch.mean(B_loss)

class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

def regul(val_loader, model, criterion, epoch, num_epochs, early_stopping):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs).squeeze()
            vl = criterion(val_outputs, val_targets)
            val_loss += vl.item() * val_inputs.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}")

        early_stopping(val_loss, model)

    return val_loss

plt.switch_backend("Agg")

def train_model(model, early_stopping, train_loader, val_loader, criterion, optimizer, num_epochs=1000):
    stop = 0
    tl_vector = []
    vl_vector = []
    idx = num_epochs - 1

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        val_loss = regul(val_loader, model, criterion, epoch, num_epochs, early_stopping)

        tl_vector.append(train_loss)
        vl_vector.append(val_loss)

        if early_stopping.early_stop and stop == 0:
            idx = epoch - early_stopping.patience
            print(f"Early stopping at epoch {idx}\n Lowest loss: {-early_stopping.best_score}")
            stop = 1

    early_stopping.load_best_model(model)

    return tl_vector, vl_vector, idx

def objective(trial):
    # Define hyperparameters to be optimized
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    n_layers = trial.suggest_int("n_layers", 1, 5)
    n_units = [trial.suggest_int(f"n_units_l{i}", 4, 128) for i in range(n_layers)]
    activation = trial.suggest_categorical("activation", [nn.ReLU, nn.Tanh, nn.LeakyReLU])

    dir = "/user/u/u24gmarujo/root_fl/"
    MC_file = "MC.root"
    ED_file = "ED.root"

    x, y = prep.prepdata(dir, MC_file, ED_file)
    dataset = prep.ClassificationDataset(x, y)

    total_length = len(dataset)
    train_length = int(0.5 * total_length)
    test_length = int(0.25 * total_length)
    val_length = total_length - train_length - test_length

    train_set, test_set, val_set = random_split(dataset, [train_length, test_length, val_length])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

    input_size = x.shape[1]
    model = ClassificationModel(input_size, n_layers, n_units, activation)

    # Calculate class weights
    class_wght = torch.tensor([1 / np.sum(y == 0), 1 / np.sum(y == 1)], dtype=torch.float32)
    class_wght = class_wght / class_wght.sum()
        
    criterion = BalancedLoss(alpha=class_wght)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(patience=200, delta=1e-6)

    _, vl_vector, idx = train_model(model, early_stopping, train_loader, val_loader, criterion, optimizer, num_epochs=1000)

    return vl_vector[idx]

def main():
    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)  # Number of trials can be adjusted

        print("Best trial:")
        trial = study.best_trial

        print(f"\nValue: {trial.value}")
        print("Param.: ")
        for key, value in trial.params.items():
            print(f"{key}: {value}")

        # Recreate the best model with the best hyperparameters
        best_lr = trial.params["lr"]
        best_n_layers = trial.params["n_layers"]
        best_n_units = [trial.params[f"n_units_l{i}"] for i in range(best_n_layers)]
        best_activation = trial.params["activation"]

        dir = "/user/u/u24gmarujo/root_fl/"
        MC_file = "MC.root"
        ED_file = "ED.root"

        x, y = prep.prepdata(dir, MC_file, ED_file)
        dataset = prep.ClassificationDataset(x, y)

        total_length = len(dataset)
        train_length = int(0.5 * total_length)
        test_length = int(0.25 * total_length)
        val_length = total_length - train_length - test_length

        train_set, test_set, val_set = random_split(dataset, [train_length, test_length, val_length])

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

        input_size = x.shape[1]
        best_model = ClassificationModel(input_size, best_n_layers, best_n_units, best_activation)

        # Calculate class weights
        class_wght = torch.tensor([1 / np.sum(y == 0), 1 / np.sum(y == 1)], dtype=torch.float32)
        class_wght = class_wght / class_wght.sum()
        
        criterion = BalancedLoss(alpha=class_wght)
        best_optimizer = optim.Adam(best_model.parameters(), lr=best_lr)

        early_stopping = EarlyStopping(patience=200, delta=1e-6)

        # Train the best model on the full training data and capture loss vectors
        tl_vector, vl_vector, idx = train_model(best_model, early_stopping, train_loader, val_loader, criterion, best_optimizer, num_epochs=1000)

        checkpoint_dir = "/user/u/u24gmarujo/SummerLIP24/Machine_Learning/Evaluation/"
        checkpoint_file = "Optim_model_checkpoint.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save the best model, optimizer states, and trial hyperparameters
        torch.save({
            "model_state_dict": best_model.state_dict(),
            "optimizer_state_dict": best_optimizer.state_dict(),
            "dataset": dataset,
            "test_set": test_set,
            "hyperparameters": trial.params}, checkpoint_path)

        print(f"Best model saved to {checkpoint_path}")

        # Plot the best training and validation losses
        indices = range(1, len(tl_vector) + 1)
        plt.figure()
        plt.plot(indices, tl_vector, marker="o", color="navy", label="Training Loss", markersize=1)
        plt.plot(indices, vl_vector, marker="o", color="darkorange", label="Validation Loss", markersize=1)
        plt.scatter(idx + 1, vl_vector[idx], marker="o", color="black", label="Early Stop", s=64)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Best Trial Loss Over Epochs")
        plt.legend()
        plt.savefig("Optim_loss.pdf")
        plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
