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
import prepdata as prep
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
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Calculate the standard binary cross-entropy loss without reduction
        BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")
        
        # Calculate the probability of correct classification
        pt = torch.exp(-BCE_loss)
        
        # Apply the modulating factor (1-pt)^gamma to the loss
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        # Return the mean of the focal loss
        return torch.mean(F_loss)
    
class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience  # Number of epochs to wait for improvement in validation loss, before stopping the training
        self.delta = delta  # Minimum change in validation loss that qualifies as an improvement
        self.best_score = None  # Best validation score encountered during training
        self.early_stop = False  # Boolean flag that indicates if training should be stopped early
        self.counter = 0  # Counts the number of epochs since the last improvement in validation loss
        self.best_model_state = None  # Stores the state of the model when the best validation loss was observed

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

def train_model(model, early_stopping, train_loader, val_loader, criterion, optimizer, num_epochs=100):
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
    
    # Load the best model
    early_stopping.load_best_model(model)
        
    indices = range(1, num_epochs + 1) 
        
    # Plot training and validation loss
    plt.figure()
    plt.plot(indices, tl_vector, marker="o", color="navy", label="Training", markersize=1)
    plt.plot(indices, vl_vector, marker="o", color="darkorange", label="Validation", markersize=1)
    plt.scatter(idx + 1, vl_vector[idx-100], marker="o", color="black", label="Early Stop", s=64)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.savefig("loss.pdf")  
    plt.close() 
        
def main():
    try:
        # Input path
        #dir = "..." #Diogo
        dir = "/user/u/u24gmarujo/root_fl/" #Gonçalo
        MC_file = "MC.root"
        ED_file = "ED.root"
        
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
        model = ClassificationModel(input_size)

        # Calculate class weights
        total_sampl = len(y)
        class_wght = torch.tensor([total_sampl / (2 * np.sum(y == 0)), total_sampl / (2 * np.sum(y == 1))], dtype=torch.float32)

        # Define loss function and optimizer
        # criterion = nn.BCELoss()
        criterion = FocalLoss(alpha=class_wght[1])
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Early stopping (delta should be a positive quantity)
        early_stopping = EarlyStopping(patience=100, delta=0)

        # Train the model
        train_model(model, early_stopping, train_loader, val_loader, criterion, optimizer, num_epochs=1000)

        # Define the directory and filename
        # checkpoint_dir = "/user/u/u24diogobpereira/LocalRep/Machine_Learning/Diogo/Evaluation/" #Diogo
        checkpoint_dir = "/user/u/u24gmarujo/SummerLIP24/Machine_Learning/Evaluation/" # Gonçalo
        checkpoint_file = "model_checkpoint.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

        # Create the directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model, optimizer_state_dict, dataset and test_set
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dataset": dataset,
                    "test_set": test_set}, checkpoint_path)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
# Ensure main runs only when the script is executed directly and not when it is imported    
if __name__ == '__main__':
    main()
