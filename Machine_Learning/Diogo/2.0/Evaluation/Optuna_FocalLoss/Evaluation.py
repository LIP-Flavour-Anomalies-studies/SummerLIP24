"""
Created on August 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import torch
import uproot3 as uproot
import torch.nn as nn
import sys
import os
from fpdf import FPDF
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Add the directory containing NN.py to the Python path
sys.path.append(os.path.abspath('/user/u/u24diogobpereira/LocalRep/Machine_Learning/Diogo/2.0/Training/Optuna_FocalLoss/'))
sys.path.append(os.path.abspath('/user/u/u24diogobpereira/LocalRep/Machine_Learning/Diogo/2.0/Training/'))

from NN import ClassificationModel 
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import Plots as p



def load_model():

    # Load checkpoint
    checkpoint = torch.load('model_checkpoint.pth')

    # Load test dataset from checkpoint
    dataset = checkpoint['dataset']
    test_dataset = checkpoint['test_set']
    test_indices = test_dataset.indices

    # Create DataLoader for test data
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Save hyperparameters
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Load hyperparameters
    activation_functions = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'LeakyReLU': nn.LeakyReLU}
    models_params = {}
    for model_name in ['alpha', 'focal']:
        hyperparams = checkpoint[f'{model_name}_hyperparameters']
        models_params[model_name] = {
        'lr': hyperparams["lr"],
        'n_layers': hyperparams["n_layers"],
        'n_units': [hyperparams[f"n_units_l{i}"] for i in range(hyperparams["n_layers"])],
        'activation': activation_functions[hyperparams["activation"]]}
        
        pdf.cell(200, 10, txt=f"\n{model_name} model:", ln=True)
        pdf.cell(200, 10, txt=f'Learning rate -> {hyperparams["lr"]}', ln=True)
        pdf.cell(200, 10, txt=f'Number of layers -> {hyperparams["n_layers"]}', ln=True)
        pdf.cell(200, 10, txt=f'Activation function -> {hyperparams["activation"]}', ln=True)
        for i in range(hyperparams["n_layers"]):
            pdf.cell(200, 10, txt=f'Number of neurons in layer {i}: -> {hyperparams[f"n_units_l{i}"]:}', ln=True)

    pdf.output(f"Hyperparameters.pdf")
    
    # Recreate the model and load state dict
    input_size = test_dataset.dataset.X.shape[1]
    alpha_model = ClassificationModel(input_size, models_params['alpha']['n_layers'], models_params['alpha']['n_units'], models_params['alpha']['activation'])
    focal_model = ClassificationModel(input_size, models_params['focal']['n_layers'], models_params['focal']['n_units'], models_params['focal']['activation'])
    alpha_model.load_state_dict(checkpoint['alpha_model_state_dict'])
    focal_model.load_state_dict(checkpoint['focal_model_state_dict'])
    
    return alpha_model, focal_model, test_loader, test_indices




def get_targets_probabilities(model, test_loader):
    targets = []
    probabilities = []

    # Loop through dataloader minimizes memory usage
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            probabilities.extend(outputs.cpu().numpy())
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
   
  
  
  

def get_Bmass(test_indices):

    dir = "/user/u/u24diogobpereira/Data/" #Diogo
    MC_file = "LargerMC.root"
    ED_file = "LargerED.root"

    data = uproot.open(dir + ED_file)
    data_mc = uproot.open(dir + MC_file)
    
    TreeS = data_mc["signal_tree"]
    TreeB = data["background_tree"]
    
    word = "bTMass"
    variab = word.encode('utf-8')

    # Extract the data from the tree and returns it as a 2D pandas array
    signal = TreeS.arrays(branches=variab)
    background = TreeB.arrays(branches=variab)
   
    signal_array = list(signal.values())[0]  # Extract the first (and only) array from the dictionary
    background_array = list(background.values())[0]  # Extract the first (and only) array from the dictionary

    nsignal = len(signal_array)
    nback = len(background_array)
    nevents = nsignal + nback

    x = np.zeros(nevents)
    x[:nsignal] = signal_array
    x[nsignal:] = background_array
    x = x[test_indices]
 
    return x





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




alpha_model, focal_model, test_loader, test_indices = load_model()       
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
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
 
    '''   
    f1_vector=[]
    for thresh in thresholds:
        predictions = get_predictions(model, test_loader, thresh)
        f1_vector.append(f1_score(targets, predictions))
    
    f1_vector = np.array(f1_vector)
    ix = np.argmax(f1_vector)
    best_thresh = thresholds[ix]
    '''
    
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
    p.plot_prob_distribution(targets, probabilities, best_thresh, model_name)
    # p.scatter_plots(test_loader, probabilities, targets, model_name)
    x = get_Bmass(test_indices)
    p.variable_histograms(test_loader, predictions, x, model_name)

# Save the plot as a PDF file
plt.legend(loc='lower right')
plt.savefig(f"roc_curve.pdf")
plt.close()  
