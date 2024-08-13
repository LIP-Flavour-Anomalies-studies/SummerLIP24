"""
Created on August 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""


import torch
from torch.utils.data import DataLoader, TensorDataset
from NN import ClassificationModel 

# Load checkpoint
checkpoint = torch.load('model_checkpoint.pth')

# Load test dataset from checkpoint
dataset = checkpoint['dataset']
test_dataset = checkpoint['test_set']

# Create DataLoader for test data
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Recreate the model and load state dict
input_size = test_dataset.dataset.X.shape[1]
model = ClassificationModel(input_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set model to evaluation mode

# Evaluation loop
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs).squeeze()
        predicted = (outputs > 0.5).float()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')

