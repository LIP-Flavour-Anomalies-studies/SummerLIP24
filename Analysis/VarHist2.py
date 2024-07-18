# -*- coding: utf-8 -*-
"""
Created on July 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import ROOT
from ROOT import TFile, TH1F, TCanvas, TLegend

# Input path
dir = "~/LocalRep/Data/" # Diogo
#dir = "/user/u/u24gmarujo/root_folder/" #Gonçalo
data_file1 = "B0ToKstMuMu_JpsiMC_22F_0-1531_miniaod.root"
data_file2 = "B0ToKstMuMu_22F_0-812_miniaod.root"

# Open .root files
root_file1 = TFile.Open(dir + data_file1)
root_file2 = TFile.Open(dir + data_file2)

if not root_file1 or root_file1.IsZombie():
    print(f"Error: Could not open file {dir + data_file1}")
    exit()
    
if not root_file2 or root_file2.IsZombie():
    print(f"Error: Could not open file {dir + data_file2}")
    exit()

tree1 = root_file1.Get("B0KstMuMu/B0KstMuMuNTuple")
tree2 = root_file2.Get("B0KstMuMu/B0KstMuMuNTuple")

if not tree1:
    print("Error: Could not find tree 'B0KstMuMuNTuple' in the first file.")
    exit()
    
if not tree2:
    print("Error: Could not find tree 'B0KstMuMuNTuple' in the second file.")
    exit()

# Create histograms for each file
hist1 = {
    "h_bMass": TH1F("h_bMass1", "B Mass", 100, 4.5, 6.0),
    "h_bBarMass": TH1F("h_bBarMass1", "B Bar Mass", 100, 4.5, 6.0),
    "h_kstMass": TH1F("h_kstMass1", "K* Mass", 100, 0, 3),
    "h_kstBarMass": TH1F("h_kstBarMass1", "K* Bar Mass", 100, 0, 3),
    "h_mumuMass": TH1F("h_mumuMass1", "MuMu Mass", 100, 0.8, 11),
    "h_bCosAlphaBS": TH1F("h_bCosAlphaBS1", "B CosAlphaBS", 100, -1.0, 1.0),
    "h_bVtxCL": TH1F("h_bVtxCL1", "B VtxCL", 100, 0, 1.0),
    "h_bLBSs": TH1F("h_bLBSs1", "Significance B LBS", 100, 0, 20),
    "h_bDCABSs": TH1F("h_bDCABSs1", "Significance B DCABS", 100, 0, 20),
    "h_kstTrkpDCABSs": TH1F("h_kstTrkpDCABSs1", "Significance K* TrkpDCABS", 100, 0, 10),
    "h_kstTrkmDCABSs": TH1F("h_kstTrkmDCABSs1", "Significance K* TrkmDCABS", 100, 0, 10),
}

hist2 = {
    "h_bMass": TH1F("h_bMass2", "B Mass", 100, 4.5, 6.0),
    "h_bBarMass": TH1F("h_bBarMass2", "B Bar Mass", 100, 4.5, 6.0),
    "h_kstMass": TH1F("h_kstMass2", "K* Mass", 100, 0, 3),
    "h_kstBarMass": TH1F("h_kstBarMass2", "K* Bar Mass", 100, 0, 3),
    "h_mumuMass": TH1F("h_mumuMass2", "MuMu Mass", 100, 0.8, 11),
    "h_bCosAlphaBS": TH1F("h_bCosAlphaBS2", "B CosAlphaBS", 100, -1.0, 1.0),
    "h_bVtxCL": TH1F("h_bVtxCL2", "B VtxCL", 100, 0, 1.0),
    "h_bLBSs": TH1F("h_bLBSs2", "Significance B LBS", 100, 0, 20),
    "h_bDCABSs": TH1F("h_bDCABSs2", "Significance B DCABS", 100, 0, 20),
    "h_kstTrkpDCABSs": TH1F("h_kstTrkpDCABSs2", "Significance K* TrkpDCABS", 100, 0, 10),
    "h_kstTrkmDCABSs": TH1F("h_kstTrkmDCABSs2", "Significance K* TrkmDCABS", 100, 0, 10),
}

# Function to fill histograms from a tree
def fill_hist(tree, histograms):
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)

        bMass = getattr(tree, "bMass")
        bBarMass = getattr(tree, "bBarMass")
        kstMass = getattr(tree, "kstMass")
        kstBarMass = getattr(tree, "kstBarMass")
        mumuMass = getattr(tree, "mumuMass")
        bCosAlphaBS = getattr(tree, "bCosAlphaBS")
        bVtxCL = getattr(tree, "bVtxCL")
        bLBS = getattr(tree, "bLBS")
        bLBSE = getattr(tree, "bLBSE")
        bDCABS = getattr(tree, "bDCABS")
        bDCABSE = getattr(tree, "bDCABSE")
        kstTrkpDCABS = getattr(tree, "kstTrkpDCABS")
        kstTrkpDCABSE = getattr(tree, "kstTrkpDCABSE")
        kstTrkmDCABS = getattr(tree, "kstTrkmDCABS")
        kstTrkmDCABSE = getattr(tree, "kstTrkmDCABSE")

        for value in bMass:
            histograms["h_bMass"].Fill(value)

        for value in bBarMass:
            histograms["h_bBarMass"].Fill(value)

        for value in kstMass:
            histograms["h_kstMass"].Fill(value)
        
        for value in kstBarMass:
            histograms["h_kstBarMass"].Fill(value)

        for value in mumuMass:
            histograms["h_mumuMass"].Fill(value)

        for value in bCosAlphaBS:
            histograms["h_bCosAlphaBS"].Fill(value)

        for value in bVtxCL:
            histograms["h_bVtxCL"].Fill(value)

        for value1, value2 in zip(bLBS, bLBSE):
            if value2 != 0:
                histograms["h_bLBSs"].Fill(abs(value1 / value2))

        for value1, value2 in zip(bDCABS, bDCABSE):
            if value2 != 0:
                histograms["h_bDCABSs"].Fill(abs(value1 / value2))

        for value1, value2 in zip(kstTrkpDCABS, kstTrkpDCABSE):
            if value2 != 0:          
                histograms["h_kstTrkpDCABSs"].Fill(value1 / value2)

        for value1, value2 in zip(kstTrkmDCABS, kstTrkmDCABSE):
            if value2 != 0:
                histograms["h_kstTrkmDCABSs"].Fill(value1 / value2)

# Fill histograms for both trees
fill_hist(tree1, hist1)
fill_hist(tree2, hist2)
           
# Function to plot two histograms on the same canvas and save as .pdf def plot_hist(hist1, hist2, file_name, title):
def plot_hist(hist1, hist2, file_name, title):
    canvas = TCanvas()
     hist1.SetLineColor(ROOT.kRed)
     hist2.SetLineColor(ROOT.kBlue)
     hist1.SetTitle(title)
      
     # Set the y-axis maximum to include all data
     max1 = hist1.GetMaximum()
     max2 = hist2.GetMaximum()
     hist1.SetMaximum(1.1 * max(max1, max2))
       
     # ...
     hist1.Draw()
     hist2.Draw("SAME")
   
     # Add legend
     legend = TLegend(0.7, 0.65, 0.9, 0.76)
     legend.AddEntry(hist1, "JpsiMC_22F_0-1531", "l")
     legend.AddEntry(hist2, "22F_0-812", "l")
     legend.Draw()
   
     canvas.SaveAs(file_name)
     canvas.Close()
   
# Plot histograms and save them
plot_hist(hist1["h_bMass"], hist2["h_bMass"], "h_bMass.pdf", "B Mass")
plot_hist(hist1["h_bBarMass"], hist2["h_bBarMass"], "h_bBarMass.pdf", "B Bar Mass")
plot_hist(hist1["h_kstMass"], hist2["h_kstMass"], "h_kstMass.pdf", "K* Mass")
plot_hist(hist1["h_kstBarMass"], hist2["h_kstBarMass"], "h_kstBarMass.pdf", "K* Bar Mass")
plot_hist(hist1["h_mumuMass"], hist2["h_mumuMass"], "h_mumuMass.pdf", "MuMu Mass")
plot_hist(hist1["h_bCosAlphaBS"], hist2["h_bCosAlphaBS"], "h_bCosAlphaBS.pdf", "B CosAlphaBS")
plot_hist(hist1["h_bVtxCL"], hist2["h_bVtxCL"], "h_bVtxCL.pdf", "B VtxCL")
plot_hist(hist1["h_bLBSs"], hist2["h_bLBSs"], "h_bLBSs.pdf", "Significance B LBS")
plot_hist(hist1["h_bDCABSs"], hist2["h_bDCABSs"], "h_bDCABSs.pdf", "Significance B DCABS")
plot_hist(hist1["h_kstTrkpDCABSs"], hist2["h_kstTrkpDCABSs"], "h_kstTrkpDCABSs.pdf", "Significance K* TrkpDCABS")
plot_hist(hist1["h_kstTrkmDCABSs"], hist2["h_kstTrkmDCABSs"], "h_kstTrkmDCABSs.pdf", "Significance K* TrkmDCABS")
  
# Save histograms to a new .root file
output_file = TFile("VariabHist2.root", "RECREATE")
  
for key, hist in hist1.items():
    hist.Write()
      
for key, hist in hist2.items():
    hist.Write()
  
output_file.Close()
  
# Close .root files
root_file1.Close()
root_file2.Close()
