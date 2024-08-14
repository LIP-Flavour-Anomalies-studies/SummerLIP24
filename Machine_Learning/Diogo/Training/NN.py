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
import prepdata as prep
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split




class ClassificationModel(nn.Module):

    def __init__(self, input_size):
    
        super(ClassificationModel, self).__init__()
        self.first_layer = nn.Linear(input_size, 1)  # Single output for binary classification
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = self.first_layer(x)
        x = self.sigmoid(x)  # Apply sigmoid to get output in [0, 1]
        return x
      
      
      
      
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
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
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
     
 
      
      
      
      
      
      
        
'''        
-------------------------------------------||-------------------------------------------        
'''       
        

def train_model(model, early_stopping, train_loader, val_loader, criterion, optimizer, num_epochs=100):

    train_losses = []
    validation_losses = []

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
        train_losses.append(t_loss)          
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {t_loss}")
        
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_outputs = model(val_inputs).squeeze()
                val_loss = criterion(val_outputs, val_targets) 
                v_loss += val_loss.item() * val_inputs.size(0)
           
        v_loss /= len(val_loader.dataset)         
        validation_losses.append(v_loss) 
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
        
    # Load the best model
    early_stopping.load_best_model(model)
        
    # Plot the training loss
    plt.figure()
    plt.plot(range(20, num_epochs + 1), train_losses[19:], marker='o', color='navy', label='Training')
    plt.plot(range(20, num_epochs + 1), validation_losses[19:], marker='o', color='darkorange', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.savefig('training_validation_loss.pdf')  
    plt.close()  
        



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

        # Create DataLoader for training and testing
        train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False)

        # Create an instance of 'ClassificationModel' called model
        input_size = x.shape[1]
        model = ClassificationModel(input_size)
        print(model)

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001) # Pass model parameters (weights and biases) to the optimizer

        # Early stopping
        early_stopping = EarlyStopping(patience=30, delta=-0.01)

        # Train the model
        train_model(model, early_stopping, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=500)
        
        
        # Define the directory and filename
        checkpoint_dir = '/user/u/u24diogobpereira/LocalRep/Machine_Learning/Diogo/Evaluation/' #Diogo
        checkpoint_file = 'model_checkpoint.pth'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

        # Create the directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model and optimizer state dict and test dataset
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dataset': dataset,
                    'test_set': test_set}, checkpoint_path)        
       
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
        
        
        
# Ensure main runs only when the script is executed directly and not when it is imported    
if __name__ == '__main__':
    main()
    
    
    
    
