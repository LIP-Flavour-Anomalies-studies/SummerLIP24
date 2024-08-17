"""
Created on August 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import uproot3 as uproot
import awkward0 as ak
import numpy as np
import prepdata as prep
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

class ClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(ClassificationModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # Single output for binary classification
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)  # Apply sigmoid to get output in [0, 1]
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()  # Adjust outputs to match the shape of targets
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            predicted_labels = (outputs >= 0.5).float()  # Threshold at 0.5 for binary classification
            predictions.extend(predicted_labels.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return torch.tensor(predictions), torch.tensor(targets)

def main():
    try:
        # Input path
        # dir = ... #Diogo
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
        train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False)

        # Initialize the model
        input_size = x.shape[1]
        model = ClassificationModel(input_size)

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        train_model(model, train_dataloader, criterion, optimizer, num_epochs=10)

        # Evaluate on the test set
        predictions, targets = evaluate_model(model, test_dataloader)

        # Calculate accuracy
        accuracy = (predictions == targets).float().mean().item()
        print(f"\nTest set accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
# Ensure main runs only when the script is executed directly and not when it is imported    
if __name__ == '__main__':
    main()
