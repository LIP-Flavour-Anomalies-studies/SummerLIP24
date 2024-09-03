"""
Created on August 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import uproot3 as uproot
import awkward0 as ak
import numpy as np
import os
import prepdata_v1 as prep
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

class ClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(ClassificationModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # Single output for binary classification
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)  # Apply sigmoid to get output in [0, 1]
        return x
 
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
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Calculate the standard binary cross-entropy loss without reduction
        CE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")
        
        # Calculate the probability of correct classification
        pt = torch.exp(-CE_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)
            F_loss = alpha_t * (1-pt)**self.gamma * CE_loss
        else:
            F_loss = (1-pt)**self.gamma * CE_loss
        
        # Return the mean of the focal loss
        return torch.mean(F_loss)
    
class EarlyStopping:
    def __init__(self, patience, delta, stability):
        self.patience = patience  # Number of epochs to wait for improvement in validation loss, before stopping the training
        self.delta = delta  # Minimum change in validation loss that qualifies as an improvement
        self.stability = stability
        self.best_score = None  # Best validation score encountered during training
        self.early_stop = False  # Boolean flag that indicates if training should be stopped early
        self.counter = 0  # Counts the number of epochs since the last improvement in validation loss
        self.best_model_state = None  # Stores the state of the model when the best validation loss was observed

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score - self.delta and self.best_score - score < self.stability:
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
    
    # Compute validation loss for 1 epoch
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs).squeeze()
            vl = criterion(val_outputs, val_targets) 
            val_loss += vl.item() * val_inputs.size(0)          
        val_loss /= len(val_loader.dataset)              
        print(f"Epoch {epoch+1}/{num_epochs}")  
        
        # Check if validation loss has reached its minimum
        early_stopping(val_loss, model)
    
    return val_loss

# Set the matplotlib backend to 'Agg' for saving plots as files
plt.switch_backend("Agg")

def train_model(model, early_stopping, train_loader, val_loader, criterion, optimizer, num_epochs=1000, flag=0):
    stop = 0
    tl_vector = []
    vl_vector = []
    idx = num_epochs
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()  # Adjust outputs to match the shape of targets
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        val_loss = regul(val_loader, model, criterion, epoch, num_epochs, early_stopping)
       
        # Save training and validation loss in vectors
        tl_vector.append(train_loss)   
        vl_vector.append(val_loss)
        
        # Save best epoch number
        if early_stopping.early_stop and stop == 0:
            idx = epoch - early_stopping.patience
            print(f"Early stopping at epoch {idx}\n Lowest loss: {-early_stopping.best_score}")
            stop = 1
            break
    
    # Load the best model
    early_stopping.load_best_model(model)
        
    indices = range(1, len(tl_vector) + 1) 
        
    # Plot training and validation loss
    plt.figure()
    plt.plot(indices, tl_vector, marker="o", color="navy", label="Training", markersize=1)
    plt.plot(indices, vl_vector, marker="o", color="darkorange", label="Validation", markersize=1)
    plt.scatter(idx + 1, vl_vector[idx], marker="o", color="black", label="Early Stop", s=64)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    
    if flag == 0:
        plt.savefig("B_loss_v1.pdf")
        plt.ylim(0, max(max(tl_vector), max(vl_vector))/1.5) 
    else:
        plt.savefig("F_loss_v1.pdf")
        plt.ylim(0, max(max(tl_vector), max(vl_vector))/6)
    plt.close() 
        
def main():
    try:
        # Input path
        #dir = "..." #Diogo
        dir = "/user/u/u24gmarujo/root_fl/" #Gonçalo
        MC_file = "LargerMC.root"
        ED_file = "LargerED.root"
        
        # Prepare data
        x, y = prep.prepdata(dir, MC_file, ED_file)
        
        print("Shape of x:", x.shape)
        print("Shape of y:", y.shape)
        
        # Create dataset
        dataset = prep.ClassificationDataset(x, y)

        # Calculate lengths based on dataset size
        total_length = len(dataset)
        train_length = int(0.5 * total_length)
        test_length = int(0.25 * total_length)
        val_length = total_length - train_length - test_length

        # Create random splits
        train_set, test_set, val_set = random_split(dataset, [train_length, test_length, val_length])

        # Verify lengths
        print("Training set length:", len(train_set))
        print("Testing set length:", len(test_set))
        print("Validation set length:", len(val_set))
        print()

        # Create DataLoader for training and testing
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

        # Initialize the model
        input_size = x.shape[1]
        
        B_model = ClassificationModel(input_size)
        F_model = ClassificationModel(input_size)

        # Calculate class weights
        class_wght = torch.tensor([1 / np.sum(y == 0), 1 / np.sum(y == 1)], dtype=torch.float32)
        class_wght = class_wght / class_wght.sum()

        # Define loss function and optimizer
        B_criterion = BalancedLoss(alpha=class_wght)
        F_criterion = FocalLoss(alpha=class_wght)
        
        B_optimizer = optim.Adam(B_model.parameters(), lr=0.001)
        F_optimizer = optim.Adam(F_model.parameters(), lr=0.001)

        # Early stopping (delta should be a positive quantity)
        B_early_stopping = EarlyStopping(patience=100, delta=1e-6, stability=1e-2)
        F_early_stopping = EarlyStopping(patience=100, delta=1e-6, stability=1e-2)

        # Train the model
        print("\nTraining model with balanced cross-entropy loss...")
        train_model(B_model, B_early_stopping, train_loader, val_loader, B_criterion, B_optimizer, num_epochs=1000, flag = 0)
        print("\nTraining model with focal loss...")
        train_model(F_model, F_early_stopping, train_loader, val_loader, F_criterion, F_optimizer, num_epochs=1000, flag = 1)

        # Define the directory and filename
        # checkpoint_dir = "/user/u/u24diogobpereira/LocalRep/Machine_Learning/Diogo/Evaluation/" #Diogo
        checkpoint_dir = "/user/u/u24gmarujo/SummerLIP24/Machine_Learning/Evaluation/" # Gonçalo
        
        B_checkpoint_file = "B_model_checkpoint_v1.pth" 
        F_checkpoint_file = "F_model_checkpoint_v1.pth"
        
        B_checkpoint_path = os.path.join(checkpoint_dir, B_checkpoint_file)
        F_checkpoint_path = os.path.join(checkpoint_dir, F_checkpoint_file)
        
        # Create the directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model, optimizer_state_dict, dataset and test_set
        torch.save({"model_state_dict": B_model.state_dict(),
                    "optimizer_state_dict": B_optimizer.state_dict(),
                    "dataset": dataset,
                    "test_set": test_set}, B_checkpoint_path)
        
        torch.save({"model_state_dict": F_model.state_dict(),
                    "optimizer_state_dict": F_optimizer.state_dict(),
                    "dataset": dataset,
                    "test_set": test_set}, F_checkpoint_path)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
# Ensure main runs only when the script is executed directly and not when it is imported    
if __name__ == '__main__':
    main()
