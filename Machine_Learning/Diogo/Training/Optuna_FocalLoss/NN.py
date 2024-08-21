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
import prepdata as prep
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split




class ClassificationModel(nn.Module):
    def __init__(self, input_size, n_layers, n_units, activation):
    
        super(ClassificationModel, self).__init__()
        layers = []
        current_iz = input_size

        for i in range(n_layers):
            layers.append(nn.Linear(current_iz, n_units[i]))
            layers.append(activation())
            current_iz = n_units[i]

        layers.append(nn.Linear(current_iz, 1))
        layers.append(nn.Sigmoid())  

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

   
   
   
   
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)   
   
   
   
   
      
      
      
class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

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
            print(f"Early stopping at epoch {idx}\n Lowest loss: {-early_stopping.best_score}")
            stop_flag = 1
        
    # Load the best model
    early_stopping.load_best_model(model)
        
    return train_losses, validation_losses, idx


def loss_plot(num_epochs, train_losses, validation_losses, idx):

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
    plt.savefig('training_validation_loss.pdf')  
    plt.close()  


def objective(trial, train_loader, val_loader, input_size, class_wght, models):

    # Define hyperparameters to be optimized
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 5)
    n_units = [trial.suggest_int(f"n_units_l{i}", 4, 128) for i in range(n_layers)]
    activation_functions = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'LeakyReLU': nn.LeakyReLU}
    activation_name = trial.suggest_categorical("activation", ['ReLU', 'Tanh', 'LeakyReLU'])
    activation = activation_functions[activation_name]

    model = ClassificationModel(input_size, n_layers, n_units, activation)

    criterion = FocalLoss(alpha=class_wght[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(patience=200, delta=0)

    _, vl_vector, idx = train_model(model, early_stopping, train_loader, val_loader, criterion, optimizer, num_epochs=1000)
    
    models.append(early_stopping.best_model_state)

    return vl_vector[idx]







'''
-----------------------------------------MAIN--------------------------------------------------
'''



def main():
    try:
        # Input path
        dir = "/user/u/u24diogobpereira/Data/" #Diogo
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

        # Create DataLoader for training and validation
        train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True)

        # Compute input size and class weights
        input_size = x.shape[1]
        total_sampl = len(y)
        class_wght = torch.tensor([total_sampl / (2 * np.sum(y == 0)), total_sampl / (2 * np.sum(y == 1))], dtype=torch.float32)
        
        # Create a hyperparameter optimization study
        models=[]
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, train_dataloader, val_dataloader, input_size, class_wght, models), n_trials=10)
        
        print("Best trial:")
        trial = study.best_trial

        print(f"\nValue: {trial.value}")
        print("Param.: ")
        for key, value in trial.params.items():
            print(f"{key}: {value}")

        
        # Define the directory and filename
        checkpoint_dir = '/user/u/u24diogobpereira/LocalRep/Machine_Learning/Diogo/Evaluation/' #Diogo
        checkpoint_file = 'model_checkpoint.pth'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

        # Create the directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model and optimizer state dict and test dataset
        torch.save({'model_state_dict': models[trial.number],
                    'dataset': dataset,
                    'test_set': test_set,
                    'hyperparameters': trial.params}, checkpoint_path)        
       
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
        
        
        
# Ensure main runs only when the script is executed directly and not when it is imported    
if __name__ == '__main__':
    main()
    
    
    
    
