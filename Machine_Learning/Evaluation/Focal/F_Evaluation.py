"""
Created on August 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import numpy as np
import sys
import os
import torch
from torch.utils.data import DataLoader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import matplotlib.pyplot as plt

# Add the directory containing NeuralNetwork.py to the Python path
sys.path.append(os.path.abspath("/user/u/u24gmarujo/SummerLIP24/Machine_Learning/Training/"))

from NeuralNetwork import ClassificationModel

def load_model():
    # Load checkpoint
    checkpoint = torch.load("F_model_checkpoint.pth")

    # Load test dataset from checkpoint
    dataset = checkpoint["dataset"]
    test_dataset = checkpoint["test_set"]

    # Create DataLoader for test data
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Recreate the model and load state dict
    input_size = test_dataset.dataset.X.shape[1]
    model = ClassificationModel(input_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, test_loader

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

def save_metrics(conf_matrix, accuracy, precision, recall, f1, best_thr):
    pdf_filename = "F_metrics.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, "Evaluation Metrics")

    # Accuracy, Precision, Recall, F1-Score, Best Threshold
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 100, f"Accuracy: {accuracy:.4f}")
    c.drawString(100, height - 120, f"Precision: {precision:.4f}")
    c.drawString(100, height - 140, f"Recall: {recall:.4f}")
    c.drawString(100, height - 160, f"F-score: {f1:.4f}")
    c.drawString(100, height - 180, f"Best Threshold: {best_thr:.4f}")

    # Confusion Matrix
    c.drawString(100, height - 220, "Confusion Matrix:")
    
    # Define the starting position for the confusion matrix
    matrix_top = height - 280  # Move down to create more space
    matrix_left = 220  # Shift right for better alignment
    
    # Draw the confusion matrix
    cell_width = 100
    cell_height = 40
    c.setStrokeColor(colors.black)
    c.setLineWidth(1)

    for i in range(2):
        for j in range(2):
            x = matrix_left + j * cell_width
            y = matrix_top - i * cell_height
            c.rect(x, y, cell_width, cell_height)
            c.drawCentredString(x + cell_width / 2, y + cell_height / 2 - 6, str(conf_matrix[i, j]))

    # Draw labels for the confusion matrix
    c.setFont("Helvetica-Bold", 12)
    c.drawString(matrix_left + cell_width / 2 - 20, matrix_top + 50, "True: 1")
    c.drawString(matrix_left + 3 * cell_width / 2 - 20, matrix_top + 50, "True: 0")
    c.drawString(matrix_left - 60, matrix_top - cell_height / 2 + 35, "Pred: 1")
    c.drawString(matrix_left - 60, matrix_top - 3 * cell_height / 2 + 35, "Pred: 0")

    # Save the PDF
    c.save()
    print(f"Evaluation metrics saved to {pdf_filename}")

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
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-score: {f1:.4f}")
    print(f"Best threshold: {best_thr:.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Save metrics to a .pdf file
    save_metrics(conf_matrix, accuracy, precision, recall, f1, best_thr)
    
    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    plt.scatter(best_point[0], best_point[1], color="black", label="Best Threshold")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig("F_roc_curve.pdf")
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
    plt.hist(background_predict, bins=40, density=True, alpha=0.5, label="Background (Data)", color="red", hatch="//", edgecolor="black", range=(0.0, 1.0))
        
    plt.axvline(x=best_thr, color="black", lw=2, linestyle="--", label=f"Threshold = {best_thr:.2f}")
    plt.xlabel("Predicted Probability", fontsize=14, labelpad=15)
    plt.ylabel("Normalized Density", fontsize=14, labelpad=15) 
    plt.legend()
    plt.savefig("F_prob_distribution.pdf")  # Save the plot as a PDF file
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
        plt.scatter(background_variable, background_predict, marker="o", color="red", label="Background (Data)")  
        plt.scatter(signal_variable, signal_predict, marker="o", color="blue", label="Signal (MC)")
        plt.xlabel(name, fontsize=14, labelpad=15)
        plt.ylabel("Predicted Probability", fontsize=14, labelpad=15) 
        plt.legend()
        plt.savefig(f"F_{name}_scatter_plot.pdf")  # Save the plot as a PDF file
        plt.close()
        
def main():
    try:
        model, test_loader = load_model()

        # Evaluate on the test set
        probabilities, targets, best_thr = evaluate_model(model, test_loader)
        
        # Plot the histograms of predicted probabilities
        plot_histogram(model, test_loader, targets, best_thr)
        
        # Plot the predicted probability as a function of each variable
        scatter_plots(test_loader, probabilities, targets)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
# Ensure main runs only when the script is executed directly and not when it is imported    
if __name__ == '__main__':
    main()
