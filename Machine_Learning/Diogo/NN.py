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
        
'''        
-------------------------------------------||-------------------------------------------        
'''       
        

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):

    train_losses = []

    for epoch in range(num_epochs):
        model.train() # Switch model to training mode
        running_loss = 0.0
        
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
    plt.savefig('training_loss.pdf')  
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
        test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False)

        # Create an instance of 'ClassificationModel' called model
        input_size = x.shape[1]
        model = ClassificationModel(input_size)
        print(model)

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001) # Pass model parameters (weights and biases) to the optimizer

        # Train the model
        train_model(model, train_dataloader, criterion, optimizer, num_epochs=10)
        
        # Save model and optimizer state dict and test dataset
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dataset': dataset,
                    'test_set': test_set},
                    'model_checkpoint.pth')

    except Exception as e:
        print(f"An error occurred: {e}")
        
        
        
        
        
# Ensure main runs only when the script is executed directly and not when it is imported    
if __name__ == '__main__':
    main()
    
    
    
    
