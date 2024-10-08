"""
Created on August 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""


import torch
import sys
import os
from fpdf import FPDF
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Add the directory containing NN.py to the Python path
sys.path.append(os.path.abspath('/user/u/u24diogobpereira/LocalRep/Machine_Learning/Diogo/Training/'))

from NN import ClassificationModel 
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



def load_model():
    # Load checkpoint
    checkpoint = torch.load('model_checkpoint.pth')

    # Load test dataset from checkpoint
    dataset = checkpoint['dataset']
    test_dataset = checkpoint['test_set']

    # Create DataLoader for test data
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    '''
    lr = checkpoint['hyperparameters']["lr"]
    n_layers = checkpoint['hyperparameters']["n_layers"]
    n_units = [checkpoint['hyperparameters'][f"n_units_l{i}"] for i in range(best_n_layers)]
    activation_functions = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'LeakyReLU': nn.LeakyReLU}
    activation = activation_functions[checkpoint['hyperparameters']["activation_name"]]
    '''
    
    # Recreate the model and load state dict
    input_size = test_dataset.dataset.X.shape[1]
    alpha_model = ClassificationModel(input_size)
    focal_model = ClassificationModel(input_size)
    # model = ClassificationModel(input_size, n_layers, n_units, activation)
    alpha_model.load_state_dict(checkpoint['alpha_model_state_dict'])
    focal_model.load_state_dict(checkpoint['focal_model_state_dict'])
   
    
    return alpha_model, focal_model, test_loader




def get_targets_probabilities(model, test_loader):
    targets = []
    probabilities = []

    # Loop through dataloader minimizes memory usage
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            probabilities.extend(outputs.cpu().numpy())  # Save the probability for ROC curve
            targets.extend(labels.cpu().numpy())
        
    return targets, probabilities
    
    
    
    
   
def get_predictions(model, test_loader, best_thresh):
    predictions = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            predicted_labels = (outputs >= best_thresh).float()
            predictions.extend(predicted_labels.cpu().numpy())
            
    return predictions
   
   



def plot_histograms(targets, probabilities, best_thresh, name):
    prob = np.array(probabilities)
    targets = np.array(targets)

    plt.figure(figsize=(8, 6))

    # Background predictions
    background_predict = prob[targets == 0]
    plt.hist(background_predict, bins=40, range=(0,1), density=True, alpha=0.5, label="Background (ED)", color="red", hatch="//", edgecolor="black")
    
    # Signal predictions
    signal_predict = prob[targets == 1]
    plt.hist(signal_predict, bins=40, range=(0,1), density=True, alpha=0.7, label="Signal (MC)", color="blue")
    
    plt.axvline(x=best_thresh, color='grey', lw=2, label=f'Threshold = {best_thresh:.2f}')
    plt.xlabel("Predicted Probability", fontsize=14, labelpad=15)
    plt.ylabel("Normalized Density", fontsize=14, labelpad=15) 
    plt.legend()
    plt.savefig(f"{name}_prob_dsitribution.pdf")  # Save the plot as a PDF file
    plt.close()





def scatter_plots(test_loader, probabilities, targets, filename):

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
            for inputs, labels in test_loader:
                variables.extend(inputs[:, i].cpu().numpy())
                   
            variables = np.array(variables)
           
        # Split feature values based on target label
        signal_variable = variables[targets == 1]
        background_variable = variables[targets == 0]
           
   
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(signal_variable, signal_predict, marker='o', color='blue', label="Signal (MC)")
        plt.scatter(background_variable, background_predict, marker='o', color='red', alpha = 0.5, label="Background (DATA)")    
        plt.xlabel(name, fontsize=14, labelpad=15)
        plt.ylabel("Predicted Probability", fontsize=14, labelpad=15) 
        plt.legend()
        plt.savefig(f"{filename}_{name}_scatter_plot.pdf")  # Save the plot as a PDF file
        plt.close()
 
 


def save_metrics(accuracy, precision, recall, f1, c, name):

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add accuracy, precision, recall, and F1 score
    pdf.cell(200, 10, txt=f"Accuracy -> Fraction of correctly classified samples:\n {accuracy:.4f}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Precision -> Ability of the classifier not to label as signal a sample that is background:\n {precision:.4f}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Recall -> Ability of classifier to find all signal samples:\n {recall:.4f}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"F1 score -> Harmonic mean of precision and recall:\n {f1:.4f}\n", ln=True, align='L')

    # Add confusion matrix
    pdf.cell(200, 10, txt="Confusion Matrix:", ln=True, align='L')
    pdf.cell(200, 10, txt="|tn fp|", ln=True, align='L')
    pdf.cell(200, 10, txt="|fn tp|\n", ln=True, align='L')
    pdf.cell(200, 10, txt=f"|{c[0,0]} {c[0,1]}|", ln=True, align='L')
    pdf.cell(200, 10, txt=f"|{c[1,0]} {c[1,1]}|\n", ln=True, align='L')
    
    pdf.output(f"{name}_classification_report.pdf")
    



'''
-------------------------------------MAIN--------------------------------------------
'''




alpha_model, focal_model, test_loader = load_model()       
alpha_model.eval()  # Set model to evaluation mode
focal_model.eval()

models = {
    "alpha_model": alpha_model,
    "focal_model": focal_model
}

# Plot ROC curve
plt.figure()
plt.plot([0, 1], [0, 1], color='darkgreen', lw=2, linestyle='--', label = 'Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')

for model_name, model in models.items():

    # Compute AUC
    targets, probabilities = get_targets_probabilities(model, test_loader)    
    fpr, tpr, thresholds = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)
       
    # Get the best threshold
    '''
    J = tpr - 1.5 * fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    print('\nBest Threshold = %f' % (best_thresh))
    '''
    
    f1_vector=[]
    for thresh in thresholds:
        predictions = get_predictions(model, test_loader, thresh)
        f1_vector.append(f1_score(targets, predictions))
    
    f1_vector = np.array(f1_vector)
    ix = np.argmax(f1_vector)
    best_thresh = thresholds[ix]

    # Get predictions 
    predictions = get_predictions(model, test_loader, best_thresh)

    # Compute metrics with sklearn   
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions) # (tp / (tp + fn)) = tpr[best_thresh] 
    f1 = f1_score(targets, predictions)
    c = confusion_matrix(targets, predictions)
    
    if model_name == "alpha_model":
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Alpha Loss (area = {roc_auc:.4f})')
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Alpha cut')
    
    if model_name == "focal_model":
        plt.plot(fpr, tpr, color='navy', lw=2, alpha = 0.5, label=f'Focal Loss (area = {roc_auc:.4f})')
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='red', label='Focal cut')
   
    save_metrics(accuracy, precision, recall, f1, c, model_name)   
    plot_histograms(targets, probabilities, best_thresh, model_name)
    scatter_plots(test_loader, probabilities, targets, model_name)

# Save the plot as a PDF file
plt.legend(loc='lower right')
plt.savefig(f"roc_curve.pdf")
plt.close()  
