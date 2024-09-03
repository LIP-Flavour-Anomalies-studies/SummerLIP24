"""
Created on August 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import optuna
import uproot3 as uproot
import awkward0 as ak
import numpy as np
import os
import sys
# Add the directory containing prepdata.py to the Python path
sys.path.append(os.path.abspath('/user/u/u24diogobpereira/LocalRep/Machine_Learning/Diogo/2.0/Training/'))
import prepdata as prep
import torch
import NN
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split


                  
      
  
  
  
def regularisation(val_loader, model, criterion, epoch, num_epochs, early_stopping):  
  
    model.eval()
    v_loss = 0.0
    
    # Compute validation loss for 1 epoch
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs).squeeze()
            val_loss = criterion(val_outputs, val_targets) 
            v_loss += val_loss.item() * val_inputs.size(0)          
        v_loss /= len(val_loader.dataset)              
        print(f"Epoch {epoch+1}/{num_epochs}, v_loss: {v_loss}")  
        
        # Check if validation loss has reached its minimum
        early_stopping(v_loss, model)
    
    return v_loss
  
  
  
      
        
'''        
-------------------------------------------||-------------------------------------------        
'''       
      
      
      
        

def train_model(model, early_stopping, train_loader, val_loader, criterion, optimizer, num_epochs=100):

    stop_flag = 0
    train_losses = []
    validation_losses = []
    idx = num_epochs - 1
    max_epoch = num_epochs

    for epoch in range(num_epochs):
        model.train() # Switch model to training mode
        t_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()                # Zero gradients to avoid accumulation between batches
            outputs = model(inputs).squeeze() 
            loss = criterion(outputs, targets)   # Average loss for the current batch 
            loss.backward()                      # Computes accumulated loss gradient
            optimizer.step()                     # Updates parameters in the direction that reduces the loss
            t_loss += loss.item() * inputs.size(0)
            
        t_loss /= len(train_loader.dataset)            
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {t_loss}")       
        
        v_loss = regularisation(val_loader, model, criterion, epoch, num_epochs, early_stopping)
        
        # Save losses in vector for loss over epochs plot
        train_losses.append(t_loss)   
        validation_losses.append(v_loss) 
        
        # Save best epoch number
        if early_stopping.early_stop and stop_flag == 0:
            idx = epoch - early_stopping.patience
            max_epoch = epoch
            print(f"Early stopping at epoch {idx+1}\n Lowest loss: {-early_stopping.best_score}")
            stop_flag = 1
            break
        
        # Check if training is unstable
        if early_stopping.brk:
            break
        
    # Load the best model
    early_stopping.load_best_model(model)
        
    return train_losses, validation_losses, idx, max_epoch



      
        
'''        
-------------------------------------------||-------------------------------------------        
'''       
        




def loss_plot(num_epochs, train_losses, validation_losses, idx, name):

    indices = range(80, num_epochs + 1) 
        
    # Plot the training loss
    plt.figure()
    plt.plot(indices[::2], train_losses, marker='o', color='navy', label='Training', markersize=1)
    plt.plot(indices[::2], validation_losses, marker='o', color='darkorange', label='Validation', markersize=1)
    plt.scatter(idx + 1, validation_losses[(idx-79) // 2], marker='o', color='black', label='Early Stop', s=64)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.savefig(f"{name}_training_validation_loss.pdf")  
    plt.close()  




      
        
'''        
-------------------------------------------||-------------------------------------------        
'''       
        







def objective(trial, epochs, train_loader, val_loader, input_size, class_wght, models, focal, study):

    # Define hyperparameters to be optimized
    lr = trial.suggest_float("lr", 1e-7, 1e-5, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 5)
    n_units = [trial.suggest_int(f"n_units_l{i}", 4, 128) for i in range(n_layers)]
    activation_functions = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'LeakyReLU': nn.LeakyReLU}
    activation_name = trial.suggest_categorical("activation", ['ReLU', 'Tanh', 'LeakyReLU'])
    activation = activation_functions[activation_name]

    model = NN.ClassificationModel(input_size, n_layers, n_units, activation)
    early_stopping = NN.EarlyStopping(patience=200, delta=0)
   
    if focal:
        criterion = NN.FocalLoss(alpha=class_wght)
        early_stopping.stability = 0.01
    else:
        criterion = NN.AlphaLoss(alpha=class_wght)
        early_stopping.stability = 0.2
        
    optimizer = optim.Adam(model.parameters(), lr=lr)

    tl_vector, vl_vector, idx, max_epoch = train_model(model, early_stopping, train_loader, val_loader, criterion, optimizer, num_epochs = epochs)
   
    train_losses = tl_vector[79:][::2]
    validation_losses = vl_vector[79:][::2]

    if len(study.trials) == 1 or vl_vector[idx] < study.best_trial.value:
        models.clear()
        models.extend([early_stopping.best_model_state, train_losses, validation_losses, idx, max_epoch])

    return vl_vector[idx]







'''
-----------------------------------------MAIN--------------------------------------------------
'''



def main():
    try:
        # Input path
        dir = "/user/u/u24diogobpereira/Data/" #Diogo
        MC_file = "LargerMC.root"
        ED_file = "LargerED.root"
        
        # Prepare data
        x, y, branches = prep.prepdata(dir, MC_file, ED_file)
        
        # Create dataset
        dataset = prep.ClassificationDataset(x, y)

        # Calculate lengths based on dataset size
        total_length = len(dataset)
        train_length = int(0.5 * total_length)
        test_length = int(0.25 * total_length)
        val_length = total_length - train_length - test_length

        # Create random splits
        train_set, test_set, val_set = random_split(dataset, [train_length, test_length, val_length])

        # Create DataLoader for training and validation
        train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True)

        # Compute input size and class weights
        input_size = x.shape[1]
        class_wght = torch.tensor([1 / np.sum(y == 0), 1 / np.sum(y == 1)], dtype=torch.float32)
        class_wght = class_wght / class_wght.sum()
        
        
        '''
        ----------------------------------||-----------------------------------------
        '''
        # Define number of epochs (at least 80!!!!!)
        epochs = 7000
        
        # Create a hyperparameter optimization study
        
        # Alpha
        alpha_models=[]
        alpha_study = optuna.create_study(direction="minimize")
        alpha_study.optimize(lambda trial: objective(trial, epochs, train_dataloader, val_dataloader, input_size, class_wght, alpha_models, False, alpha_study), n_trials=30)
        
        print("Alpha best trial found!")
        alpha_trial = alpha_study.best_trial
        alpha_model_state, alpha_train_losses, alpha_val_losses, alpha_idx, alpha_max_epoch = alpha_models
        loss_plot(num_epochs = alpha_max_epoch + 1, train_losses = alpha_train_losses, validation_losses = alpha_val_losses, idx = alpha_idx, name = "alpha")

        # Focal
        focal_models=[]
        focal_study = optuna.create_study(direction="minimize")
        focal_study.optimize(lambda trial: objective(trial, epochs, train_dataloader, val_dataloader, input_size, class_wght, focal_models, True, focal_study), n_trials=30)
        
        print("Focal best trial found!")
        focal_trial = focal_study.best_trial
        focal_model_state, focal_train_losses, focal_val_losses, focal_idx, focal_max_epoch = focal_models
        loss_plot(num_epochs = focal_max_epoch + 1, train_losses = focal_train_losses, validation_losses = focal_val_losses, idx = focal_idx, name = "focal")

        # Define the directory and filename
        checkpoint_dir = '/user/u/u24diogobpereira/LocalRep/Machine_Learning/Diogo/2.0/Evaluation/Optuna_FocalLoss_30/' #Diogo
        checkpoint_file = 'model_checkpoint.pth'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

        # Create the directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model and optimizer state dict and test dataset
        torch.save({'alpha_model_state_dict': alpha_model_state,
                    'focal_model_state_dict': focal_model_state,
                    'dataset': dataset,
                    'test_set': test_set,
                    'focal_hyperparameters': focal_trial.params,
                    'alpha_hyperparameters': alpha_trial.params}, checkpoint_path)        
       
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
        
        
        
# Ensure main runs only when the script is executed directly and not when it is imported    
if __name__ == '__main__':
    main()
    
    
    
    
