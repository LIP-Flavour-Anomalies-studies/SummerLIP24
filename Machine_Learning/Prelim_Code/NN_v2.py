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

def calculate_metrics(predictions, targets):
    tp = ((predictions == 1) & (targets == 1)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1, np.array([[tp, fp], [fn, tn]])

def calculate_roc_auc_thr(targets, probabilities):
    thresholds = sorted(set(probabilities), reverse=True)
    tpr = []  # True positive rate
    fpr = []  # False positive rate
    best_thr = 0.5
    best_J = -1
    best_point = None

    for thr in thresholds:
        predicted = (probabilities >= thr).float()
        _, _, _, _, conf_matrix = calculate_metrics(predicted, targets)
        
        tp = conf_matrix[0, 0]
        fp = conf_matrix[0, 1]
        fn = conf_matrix[1, 0]
        tn = conf_matrix[1, 1]
        
        # Calculate sensitivity and specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        tpr.append(sensitivity)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        
        # Calculate Youden's J statistic
        J = sensitivity + specificity - 1

        if J > best_J:
            best_J = J
            best_thr = thr
            best_point = (fpr[-1], tpr[-1])

    tpr = np.array(tpr)
    fpr = np.array(fpr)
    auc = np.trapz(tpr, fpr)  # Calculate area under the curve based on the trapezoidal rule

    return fpr, tpr, auc, best_thr, best_point

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    targets = []
    probabilities = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            probabilities.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    # Convert lists to tensors
    targets = torch.tensor(targets)
    probabilities = torch.tensor(probabilities)
    
    # Compute ROC Curve and AUC
    fpr, tpr, auc, best_thr, best_point = calculate_roc_auc_thr(targets, probabilities)
    
    # Use the best threshold for the predictions
    predictions = (probabilities >= best_thr).float()
    
    # Compute metrics
    accuracy, precision, recall, f1, conf_matrix = calculate_metrics(predictions, targets)
    
    print(f"\nTest set accuracy: {accuracy:.4f}")
    print(f"Test set precision: {precision:.4f}")
    print(f"Test set recall: {recall:.4f}")
    print(f"Test set F-score: {f1:.4f}")
    print(f"Best threshold: {best_thr:.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    plt.scatter(best_point[0], best_point[1], color="black", label=f"Best Threshold")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.pdf")
    plt.close()
    
    return probabilities, targets, best_thr

def plot_histogram(model, data_loader, labels, best_thr):
    model.eval()
    prob = []
    targets = []
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs).squeeze()
            prob.extend(outputs.cpu().numpy())
    
    prob = np.array(prob)
    targets = np.array(labels)

    plt.figure(figsize=(8, 6))
        
    # Signal predictions
    signal_predict = prob[targets == 1]
    plt.hist(signal_predict, bins=40, density=True, alpha=0.9, label="Signal (MC)", color="blue", range=(0.0, 1.0))

    # Background predictions
    background_predict = prob[targets == 0]
    plt.hist(background_predict, bins=40, density=True, alpha=0.5, label="Background (ED)", color="red", hatch="//", edgecolor="black", range=(0.0, 1.0))
        
    plt.axvline(x=best_thr, color="black", lw=2, linestyle="--", label=f"Threshold = {best_thr:.2f}")
    plt.xlabel("Predicted Probability", fontsize=14, labelpad=15)
    plt.ylabel("Normalized Density", fontsize=14, labelpad=15) 
    plt.legend()
    plt.savefig("prob_distribution.pdf")  # Save the plot as a PDF file
    plt.close()
    
def scatter_plots(dada_loader, probabilities, targets):

    # Convert probabilities to a numpy array
    prob = np.array(probabilities)
    targets = np.array(targets)
    
    # Split probabilities based on target labels
    signal_predict = prob[targets == 1]
    background_predict = prob[targets == 0]
    
    # Define feature columns
    columns = ["kstTMass", "bCosAlphaBS", "bVtxCL", "bLBSs", "bDCABSs", "kstTrkpDCABSs", "kstTrkmDCABSs", "leadingPt", "trailingPt"]

    for i in range(len(columns)):
        variables = []
        with torch.no_grad():
            name = columns[i]

            # Collect the feature values for the i-th column
            for inputs, labels in dada_loader:
                variables.extend(inputs[:, i].cpu().numpy())
                   
            variables = np.array(variables)
           
        # Split feature values based on target label
        signal_variable = variables[targets == 1]
        background_variable = variables[targets == 0]
           
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(background_variable, background_predict, marker='o', color='red', label="Background (ED)")  
        plt.scatter(signal_variable, signal_predict, marker='o', color='blue', label="Signal (MC)")
        plt.xlabel(name, fontsize=14, labelpad=15)
        plt.ylabel("Predicted Probability", fontsize=14, labelpad=15) 
        plt.legend()
        plt.savefig(f"{name}_scatter_plot.pdf")  # Save the plot as a PDF file
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
        train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False)
        val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True)

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
        train_model(model, early_stopping, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=1500)

        # Evaluate on the test set
        probabilities, targets, best_thr = evaluate_model(model, test_dataloader)
        
        # Plot the histograms of predicted probabilities
        plot_histogram(model, test_dataloader, targets, best_thr)
        
        # Plot the predicted probability as a function of each variable
        scatter_plots(test_dataloader, probabilities, targets)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
# Ensure main runs only when the script is executed directly and not when it is imported    
if __name__ == '__main__':
    main()
