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
      



class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)

        if self.alpha is not None:
            alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)
            F_loss = alpha_t * (1-pt)**self.gamma * BCE_loss
        else:
            F_loss = (1-pt)**self.gamma * BCE_loss

        return torch.mean(F_loss)





class AlphaLoss(nn.Module):
    def __init__(self, alpha=None):
        super(AlphaLoss, self).__init__()
        self.alpha = alpha  

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")

        if self.alpha is not None:
            alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)
            F_loss = alpha_t * BCE_loss
        else:
            F_loss = BCE_loss

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
     
                  
      
  
  
  

    
