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
import sys
# Add the directory containing prepdata.py to the Python path
sys.path.append(os.path.abspath('/user/u/u24diogobpereira/LocalRep/Machine_Learning/Diogo/Training/'))
import prepdata as prep
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import NN
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
  
  





def loss_plot(num_epochs, train_losses, validation_losses, idx, name):
    indices = range(100, num_epochs + 1) 
    train_losses = train_losses[99:]   
    validation_losses = validation_losses[99:]
        
    # Plot the training loss
    plt.figure()
    plt.plot(indices[::2], train_losses[::2], marker='o', color='navy', label='Training', markersize=1)
    plt.plot(indices[::2], validation_losses[::2], marker='o', color='darkorange', label='Validation', markersize=1)
    plt.scatter(idx + 1, validation_losses[idx-100], marker='o', color='black', label='Early Stop', s=64)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.savefig(name)  
    plt.close()  
        



      
        
'''        
-------------------------------------------||-------------------------------------------        
'''       
        

def train_model(model, early_stopping, train_loader, val_loader, criterion, optimizer, num_epochs=100):

    stop_flag = 0
    train_losses = []
    validation_losses = []
    idx = num_epochs - 1 

    for epoch in range(num_epochs):
        model.train() # Switch model to training mode
        t_loss = 0.0
        
        for inputs, targets in train_loader:
            
            # Zero gradients to avoid accumulation between batches
            optimizer.zero_grad() 
            
            # Squeeze() removes dimension of size 1, adjusting shape of outputs to match the shape of targets
            outputs = model(inputs).squeeze() 
            
            # Average loss for the current batch
            loss = criterion(outputs, targets) 
            
            # Computes accumulated loss gradient
            loss.backward() 
            
            # Updates parameters in the direction that reduces the loss
            optimizer.step() 
            
            # .item() converts scalar tensor 'loss' into a standard Python float
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
            print(f"Early stopping at epoch {idx}\n Lowest loss: {-early_stopping.best_score}")
            stop_flag = 1
        
    # Load the best model
    early_stopping.load_best_model(model)
        
    return train_losses, validation_losses, idx


'''
-----------------------------------------MAIN--------------------------------------------------
'''



def main():
    try:
        # Input path
        dir = "/user/u/u24diogobpereira/Data/" #Diogo
        # dir = "/user/u/u24gmarujo/root_fl/" #Gonçalo
        MC_file = "MC.root"
        ED_file = "ED.root"
        
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

        # Create DataLoader for training and testing
        train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False)

        # Compute weights of class imbalance
        class_wght = torch.tensor([1 / np.sum(y == 0), 1 / np.sum(y == 1)], dtype=torch.float32)
        class_wght = class_wght / class_wght.sum()

        # Create an instance of 'ClassificationModel' called model
        input_size = x.shape[1]
        alpha_model = NN.ClassificationModel(input_size)
        focal_model = NN.ClassificationModel(input_size)

        # Define loss function and optimizer
        alpha_loss = NN.AlphaLoss(alpha=class_wght)
        focal_loss = NN.FocalLoss(alpha=class_wght)
        
        # Pass model parameters (weights and biases) to the optimizer
        alpha_optimizer = optim.Adam(alpha_model.parameters(), lr=0.001)
        focal_optimizer = optim.Adam(focal_model.parameters(), lr=0.001)
        
        # Early stopping (delta should be positive quantity)
        alpha_early_stopping = NN.EarlyStopping(patience=100, delta=0)
        focal_early_stopping = NN.EarlyStopping(patience=100, delta=0)

        # Train the models
        epochs = 1500
        print("Training model with alpha loss:")
        alpha_t_losses, alpha_v_losses, alpha_idx = train_model(alpha_model, alpha_early_stopping, train_dataloader, val_dataloader, alpha_loss, alpha_optimizer, num_epochs = epochs)
        print("Training model with focal loss:")
        focal_t_losses, focal_v_losses, focal_idx = train_model(focal_model, focal_early_stopping, train_dataloader, val_dataloader, focal_loss, focal_optimizer, num_epochs = epochs)
        
        loss_plot(epochs, alpha_t_losses, alpha_v_losses, alpha_idx, 'alpha_training_validation_loss.pdf')
        loss_plot(epochs, focal_t_losses, focal_v_losses, focal_idx, 'focal_training_validation_loss.pdf')
     
        # Define the directory and filename
        checkpoint_dir = '/user/u/u24diogobpereira/LocalRep/Machine_Learning/Diogo/Evaluation/' #Diogo
        checkpoint_file = 'model_checkpoint.pth'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

        # Create the directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model and optimizer state dict and test dataset
        torch.save({'alpha_model_state_dict': alpha_model.state_dict(),
                    'focal_model_state_dict': focal_model.state_dict(),
                    'dataset': dataset,
                    'test_set': test_set}, checkpoint_path)        
       
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
        
        
        
# Ensure main runs only when the script is executed directly and not when it is imported    
if __name__ == '__main__':
    main()
    
    
    
    
