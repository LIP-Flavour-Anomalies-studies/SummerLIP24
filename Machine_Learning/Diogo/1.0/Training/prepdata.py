"""
Created on July 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import numpy as np
import uproot3 as uproot
import torch



def prepdata(dir, MC_file, ED_file):

    data = uproot.open(dir + ED_file)
    data_mc = uproot.open(dir + MC_file)
 
    columns = ["kstTMass", "bCosAlphaBS", "bVtxCL", "bLBSs", "bDCABSs", "kstTrkpDCABSs", "kstTrkmDCABSs", "leadingPt", "trailingPt"]

    # Convert branch names to byte strings for uproot3
    branches = [branch.encode('utf-8') for branch in columns]

    TreeS = data_mc["signal_tree"]
    TreeB = data["background_tree"]

    # Extract the data from the tree and returns it as a 2D pandas array
    signal = TreeS.arrays(branches=branches)
    background = TreeB.arrays(branches=branches)

    nsignal = len(signal[branches[0]])
    nback = len(background[branches[0]])
    nevents = nsignal + nback
    
    x = np.zeros([nevents, len(branches)]) # Modifiable array of shape (nevents, len(stages)) 
    y = np.zeros(nevents) 
    y[:nsignal] = 1 
    for i, j in enumerate(branches):
        x[:nsignal, i] = signal[j]
        x[nsignal:, i] = background[j]
    
    return x, y, columns




# Dataset class for PyTorch
class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.X = torch.tensor(data, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
        
        
                        
def test_prepdata():

    dir = "/user/u/u24diogobpereira/Data/" #Diogo
    # dir = "/user/u/u24gmarujo/root_fl/" #Gonçalo
    MC_file = "MC.root"
    ED_file = "ED.root"
    
    x, y, branches = prepdata(dir, MC_file, ED_file)
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


if __name__ == '__test_prepdata__':
    test_prepdata()
