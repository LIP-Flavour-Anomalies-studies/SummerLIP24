"""
Created on August 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

from collections import defaultdict
import torch
import uproot
import torch.nn as nn
import sys
import os
from fpdf import FPDF
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Add the directory containing NN.py to the Python path
sys.path.append(os.path.abspath('/user/u/u24diogobpereira/LocalRep/Machine_Learning/Diogo/2.0/Training/Time_reduced_5/'))
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

    # Recreate the model and load state dict
    input_size = test_dataset.dataset.X.shape[1]
    alpha_model = ClassificationModel(input_size, models_params['alpha']['n_layers'], models_params['alpha']['n_units'], models_params['alpha']['activation'])
    focal_model = ClassificationModel(input_size, models_params['focal']['n_layers'], models_params['focal']['n_units'], models_params['focal']['activation'])
    alpha_model.load_state_dict(checkpoint['alpha_model_state_dict'])
    focal_model.load_state_dict(checkpoint['focal_model_state_dict'])
    
    return alpha_model, focal_model



  
  
  

def get_Bmasses():

    # Input path
    dir = "/user/u/u24diogobpereira/Data/"
    data_file = "Tagged_mass.root" # Data
    data = uproot.open(dir + data_file)    
    Tree = data["mass_tree"]
       
    columns = ["kstTMass", "bCosAlphaBS", "bVtxCL", "bLBSs", "bDCABSs", "kstTrkpDCABSs", "kstTrkmDCABSs", "leadingPt", "trailingPt"]
    variab1 = ["bTMass"]
    variab2 = ["eventN"]

    ins = Tree.arrays(columns, library="np")
    column_arrays = [ins[column] for column in columns]
    np_array = np.column_stack(column_arrays) 
    inputs = torch.tensor(np_array, dtype=torch.float32)
    
    Bmasses = Tree.arrays(variab1, library="np")[variab1[0]]
    eventN = Tree.arrays(variab2, library="np")[variab2[0]]
 
    return inputs, Bmasses, eventN






'''
-------------------------------------MAIN--------------------------------------------
'''




alpha_model, focal_model = load_model()       
alpha_model.eval()  # Set model to evaluation mode
focal_model.eval()

inputs, Bmasses, eventN = get_Bmasses()
print(f"Bmasses.shape: {Bmasses.shape}")
print(f"eventN.shape: {eventN.shape}")

models = {
    "alpha_model": alpha_model,
    "focal_model": focal_model
}

for model_name, model in models.items(): 
    
    if model_name == "alpha_model":
        best_thresh = 0.78768
    if model_name == "focal_model":
        best_thresh = 0.47106
    
    outputs = model(inputs).squeeze()
    probabilities = torch.sigmoid(outputs)
    predictions = (probabilities >= best_thresh).float()
    predictions = np.array(predictions)
    print(f"predictions_{model_name}.shape: {predictions.shape}")    
  
    Bmass_sign = Bmasses[predictions == 1]
    event_number = eventN[predictions == 1]
    probabilities = probabilities.detach().numpy()
    probabilities = probabilities[predictions == 1]
    
    # Create a dictionary to store the maximum probability and corresponding Bmass_sign for each eventN
    max_prob_dict = defaultdict(lambda: (-np.inf, None))
 
    for mass, evt, prob in zip(Bmass_sign, event_number, probabilities):
        if prob > max_prob_dict[evt][0]:
            max_prob_dict[evt] = (prob, mass)

    Bmass_sign_max_prob = np.array([mass for _, mass in max_prob_dict.values()])
        

    p.plot_hist(Bmass_sign_max_prob, "Tagged B0 mass", model_name)

