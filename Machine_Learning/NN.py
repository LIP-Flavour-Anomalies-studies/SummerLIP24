"""
Created on July 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import ROOT
from ROOT import TFile, TH1F, TCanvas, TLegend, TLorentzVector
import numpy as np
import prepdata as prep




def main():

    # Input path
    dir = "/user/b/boletti/Run3-samples/"
    filename_mc = "reco_ntuple_LMNR_1.root" # Monte Carlo
    filename = "ntuple_flat_LMNR_PostRefitMomenta_test_2022F_skimSoftMu_1.root" # Data

    data = TFile.Open(dir + filename)
    data_mc = TFile.Open(dir + filename_mc)
                
    x, y, branches = prep.prepdata(data, data_mc)

    dataset = prep.RegressionDataset(x, y)

    print(variable_names)

    
    # Test whether x,y are saving correct values

    # Avoid unnecessary plots during runtime
    ROOT.gROOT.SetBatch(True)

    h_bVtxCL_Sign = TH1F("h_bVtxCL_Sign", "B VtxCL1", 50, 0, 1.0)
    h_bVtxCL_Bck = TH1F("h_bVtxCL_Bck", "B VtxCL2", 50, 0, 1.0)

    VtxCl = x[:, 1]
    for idx, value in enumerate(VtxCl):
        if y[idx, 1] == 1:
            h_bVtxCL_Sign.Fill(value)
        if y[idx, 1] == 0:
            h_bVtxCL_Bck.Fill(value)

    h_bVtxCL_Sign.Scale(1./h_bVtxCL_Sign.GetMaximum())
    h_bVtxCL_Bck.Scale(1./h_bVtxCL_Bck.GetMaximum())

    canvas = ROOT.TCanvas("canvas", "Canvas", 800, 600)

    h_bVtxCL_Sign.SetLineColor(ROOT.kRed)
    h_bVtxCL_Bck.SetLineColor(ROOT.kBlue)    
    h_bVtxCL_Sign.SetTitle("B VtxCL")
    h_bVtxCL_Sign.Draw("HIST")
    h_bVtxCL_Bck.Draw("HIST SAME")
    canvas.SaveAs("bVtxCl.pdf")
    canvas.Close()
    

# Ensure main runs only when the script is executed directly and not when it is imported    
if __name__ == '__main__':
    main()
