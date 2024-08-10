# -*- coding: utf-8 -*-
"""
Created on August 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import uproot3 as uproot
import numpy as np
import torch

# dir = ... #Diogo
dir = "/user/u/u24gmarujo/SummerLIP24/Machine_Learning/" #Gonçalo
MC_file = "MC.root"
ED_file = "ED.root"

# Function to load data and labels from a .root file using uproot
def load_data(dir, file_name, branches):

    # Convert branch names to byte strings for uproot3
    branches = [branch.encode('utf-8') for branch in branches]

    with uproot.open(dir + file_name) as file:
        if file_name == "MC.root":
            tree = file["signal_tree"]
            sign_condit = True
        elif file_name == "ED.root":
            tree = file["background_tree"]
            sign_condit = False
        
        # Extract the data for the specified branches
        data = tree.arrays(branches)
    
    # Prepare the data and labels arrays
    rows = []
    labels = []
    
    # Loop over the events and process the data
    for i in range(len(data[branches[0]])):
        current_row = [
            data[b"kstTMass"][i],  # Accessing branch as byte string
            data[b"bCosAlphaBS"][i],
            data[b"bVtxCL"][i],
            data[b"bLBSs"][i],
            data[b"bDCABSs"][i],
            data[b"kstTrkpDCABSs"][i],
            data[b"kstTrkmDCABSs"][i],
            max(data[b"leadingPt"][i], data[b"trailingPt"][i]),
            min(data[b"leadingPt"][i], data[b"trailingPt"][i]),
        ]

        labels.append(1 if sign_condit else 0)
        rows.append(current_row)
    
    data_array = np.array(rows, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.float32)
    
    return data_array, labels_array

# Function to prepare the dataset for PyTorch
def prepdata(dir, MC_file, ED_file):
    
    branches = ["kstTMass", "bCosAlphaBS", "bVtxCL", "bLBSs", "bDCABSs", "kstTrkpDCABSs", "kstTrkmDCABSs", "leadingPt", "trailingPt"]
    
    # Load data from Monte Carlo and Experimental Data .root files
    x_mc, y_mc = load_data(dir, MC_file, branches)
    x_data, y_data = load_data(dir, ED_file, branches)
    
    # Combine the Monte Carlo and Experimental Data arrays
    x = np.concatenate([x_mc, x_data], axis = 0)
    y = np.concatenate([y_mc, y_data], axis = 0)
    
    return x, y

# Dataset class for PyTorch
class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.X = torch.tensor(data, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

"""    
def test_prepdata():
    x, y = prepdata(dir, MC_file, ED_file)
    dataset = RegressionDataset(x, y)
    
    print(f"Data shape: {x.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"First data sample: {dataset[0]}")

    # Optional: check the length of the dataset
    print(f"Dataset length: {len(dataset)}")
    
    # Optionally, you can also create a DataLoader and iterate through it
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    for batch_idx, (data, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: data shape = {data.shape}, labels shape = {labels.shape}")
        if batch_idx == 1:  # Test first two batches
            break

# Run the test function
test_prepdata()
"""
