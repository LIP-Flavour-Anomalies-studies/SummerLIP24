"""
Created on August 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import ROOT
from ROOT import TFile, TH1F, TCanvas
import numpy as np
import prepdata as prep
import torch
from torch.utils.data import random_split

class CustomDataLoader:
    def __init__(self, dataset, batch_size = 1, shuffle = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch = [self.dataset[idx] for idx in batch_indices]
        self.current_idx += self.batch_size
        batch_x, batch_y = zip(*batch)
        return torch.stack(batch_x), torch.stack(batch_y)

def main():
    try:
        # Input path
        dir = "/user/b/boletti/Run3-samples/"
        filename_mc = "reco_ntuple_LMNR_1.root" # Monte Carlo
        filename = "ntuple_flat_LMNR_PostRefitMomenta_test_2022F_skimSoftMu_1.root" # Data

        # Open data files
        print("Opening data files...")
        data = TFile.Open(dir + filename)
        data_mc = TFile.Open(dir + filename_mc)
        
        # Check if files are successfully opened
        if not data or not data_mc or data.IsZombie() or data_mc.IsZombie():
            print("Error opening one of the data files.")
            return

        print("Data files opened successfully.")
        
        # Prepare data
        print("Preparing data...")
        x, y, branches = prep.prepdata(data, data_mc)
        
        print("Shape of x:", x.shape)
        print("Shape of y:", y.shape)
        print("Branches:", branches)
        
        # Create dataset
        dataset = prep.RegressionDataset(x, y)

        # Calculate lengths based on dataset size
        total_length = len(dataset)
        train_length = int(0.5 * total_length)
        test_length = int(0.25 * total_length)
        val_length = total_length - train_length - test_length

        # Create random splits
        train_set, test_set, val_set = random_split(dataset, [train_length, test_length, val_length])

        # Verify lengths
        print("Training set length:", len(train_set))
        print("Testing set length:", len(test_set))
        print("Validation set length:", len(val_set))

        # Create CustomDataLoader for the training set
        train_dataloader = CustomDataLoader(train_set, batch_size = 600, shuffle = True)

        # Avoid unnecessary plots during runtime
        ROOT.gROOT.SetBatch(True)

        h_bVtxCL_Sign = TH1F("h_bVtxCL_Sign", "B VtxCL1", 50, 0, 1.0)
        h_bVtxCL_Bck = TH1F("h_bVtxCL_Bck", "B VtxCL2", 50, 0, 1.0)

        VtxCl = x[:, 2]
        for idx, value in enumerate(VtxCl):
            if y[idx] == 1: # Assuming 'y' is 1D with 1 for signal and 0 for background
                h_bVtxCL_Sign.Fill(value)
            else:
                h_bVtxCL_Bck.Fill(value)

        h_bVtxCL_Sign.Scale(1./h_bVtxCL_Sign.Integral())
        h_bVtxCL_Bck.Scale(1./h_bVtxCL_Bck.Integral())

        canvas = ROOT.TCanvas("canvas", "Canvas", 800, 600)

        h_bVtxCL_Sign.SetLineColor(ROOT.kRed)
        h_bVtxCL_Bck.SetLineColor(ROOT.kBlue)    
        h_bVtxCL_Sign.SetTitle("B VtxCL")
        h_bVtxCL_Sign.Draw("HIST")
        h_bVtxCL_Bck.Draw("HIST SAME")
        canvas.SaveAs("bVtxCl.pdf")
        canvas.Close()    

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if data:
            data.Close()
        if data_mc:
            data_mc.Close()

# Ensure main runs only when the script is executed directly and not when it is imported    
if __name__ == '__main__':
    main()
