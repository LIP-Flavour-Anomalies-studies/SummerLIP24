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

# Set the matplotlib backend to 'Agg' for saving plots as files
plt.switch_backend('Agg')

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    
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
        
        train_losses.append(running_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")
    
    # Plot the training loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('training_loss.pdf')  # Save the plot as a PNG file
    plt.close()  # Close the figure to free up memory

def calculate_metrics(predictions, targets):
    tp = ((predictions == 1) & (targets == 1)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1, np.array([[tn, fp], [fn, tp]])

def calculate_roc_auc(targets, probabilities):
    thresholds = sorted(set(probabilities), reverse=True)
    tpr = []  # True positive rate
    fpr = []  # False positive rate
    for thresh in thresholds:
        predicted = (probabilities >= thresh).float()
        tp = ((predicted == 1) & (targets == 1)).sum().item()
        tn = ((predicted == 0) & (targets == 0)).sum().item()
        fp = ((predicted == 1) & (targets == 0)).sum().item()
        fn = ((predicted == 0) & (targets == 1)).sum().item()
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    tpr = np.array(tpr)
    fpr = np.array(fpr)
    auc = np.trapz(tpr, fpr)  # AUC using the trapezoidal rule
    return fpr, tpr, auc

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    targets = []
    probabilities = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            predicted_labels = (outputs >= 0.5).float()  # Threshold at 0.5 for binary classification
            predictions.extend(predicted_labels.cpu().numpy())
            probabilities.extend(outputs.cpu().numpy())  # Save the probability for ROC curve
            targets.extend(labels.cpu().numpy())
    
    # Convert lists to tensors
    predictions = torch.tensor(predictions)
    targets = torch.tensor(targets)
    probabilities = torch.tensor(probabilities)
    
    # Compute metrics
    accuracy, precision, recall, f1, conf_matrix = calculate_metrics(predictions, targets)
    
    print(f"\nTest set accuracy: {accuracy:.4f}")
    print(f"Test set precision: {precision:.4f}")
    print(f"Test set recall: {recall:.4f}")
    print(f"Test set F1 score: {f1:.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # ROC Curve and AUC
    fpr, tpr, roc_auc = calculate_roc_auc(targets, probabilities)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.pdf')  # Save the ROC curve plot
    plt.close()  # Close the figure to free up memory
    
    return predictions, targets

def main():
    try:
        # Input path
        dir = "/user/u/u24gmarujo/SummerLIP24/Machine_Learning/" 
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
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
# Ensure main runs only when the script is executed directly and not when it is imported    
if __name__ == '__main__':
    main()
