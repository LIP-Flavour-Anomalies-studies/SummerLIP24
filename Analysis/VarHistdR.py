# -*- coding: utf-8 -*-
"""
Created on July 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import ROOT
from ROOT import TFile, TH1F, TCanvas, TLegend, TLorentzVector
import numpy as np

# Input path
dir = "~/Data/" # Diogo
#dir = "/user/u/u24gmarujo/root_fl/" #Gonçalo
data_file1 = "B0ToKstMuMu_JpsiMC_22F_0-1531_miniaod.root"

# Open .root files
root_file1 = TFile.Open(dir + data_file1)

if not root_file1 or root_file1.IsZombie():
    print(f"Error: Could not open file {dir + data_file1}")
    exit()

tree1 = root_file1.Get("B0KstMuMu/B0KstMuMuNTuple")

if not tree1:
    print("Error: Could not find tree 'B0KstMuMuNTuple' in the first file.")
    exit()
    
# Create histograms for each parameter
hist1 = {
    "h_bTMass": TH1F("h_bTMass1", "True B Mass", 100, 4.5, 6.0),
    "h_kstTMass": TH1F("h_kstTMass1", "True K* Mass", 100, 0, 3),
    "h_mumuMass": TH1F("h_mumuMass1", "MuMu Mass", 100, 0.8, 11),
    "h_bCosAlphaBS": TH1F("h_bCosAlphaBS1", "B CosAlphaBS", 100, -1.0, 1.0),
    "h_bVtxCL": TH1F("h_bVtxCL1", "B VtxCL", 40, 0, 1.0),
    "h_bLBSs": TH1F("h_bLBSs1", "Significance B LBS", 40, 0, 20),
    "h_bDCABSs": TH1F("h_bDCABSs1", "Significance B DCABS", 100, -20, 20),
    "h_kstTrkpDCABSs": TH1F("h_kstTrkpDCABSs1", "Significance K* TrkpDCABS", 40, -10, 10),
    "h_kstTrkmDCABSs": TH1F("h_kstTrkmDCABSs1", "Significance K* TrkmDCABS", 40, -10, 10),
}

# Function that initializes a Lorentzvector
def create_TLvector(m, Px, Py, Pz):

    # Create and set the TLorentzVector for the candidate
    lv = TLorentzVector()
    P = np.sqrt(Px**2 + Py**2 + Pz**2)
    E = np.sqrt(P**2 + m**2) 
    lv.SetPxPyPzE(Px, Py, Pz, E)
    
    return lv

# Function that computes Delta R (between genparticles and candidates) and appends these values to a list
def calc_dR(m, particle, gparticle, tree):

    dR_list = [] # Initialize an empty list to store dR values

    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        
        entry_dR_list = []
        
        Px = getattr(tree, particle + "Px")
        Py = getattr(tree, particle + "Py")
        Pz = getattr(tree, particle + "Pz")
        
        gPx = getattr(tree, gparticle + "Px")
        gPy = getattr(tree, gparticle + "Py")
        gPz = getattr(tree, gparticle + "Pz")
        
        glv = create_TLvector(m, gPx, gPy, gPz)
        
        # Loop over each candidate in the event
        for j in range(len(Px)):
        
           lv = create_TLvector(m, Px[j], Py[j], Pz[j])
           dR = lv.DeltaR(glv)
           entry_dR_list.append(dR)  # Append dR value to the list
           
        dR_list.append(entry_dR_list)

    return dR_list

# Function to fill histograms from a tree
def fill_hist(tree, histograms):

    dR_list_mum = calc_dR(0.105, "mum", "genMum", tree1)
    dR_list_mup = calc_dR(0.105, "mup", "genMup", tree1)
    dR_list_Trkp = calc_dR(0.0, "kstTrkp", "genKstTrkp", tree1)
    dR_list_Trkm = calc_dR(0.0, "kstTrkm", "genKstTrkm", tree1)

    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        
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
        genSignal = getattr(tree, "genSignal")
        bMass = getattr(tree, "bMass")
        kstMass = getattr(tree, "kstMass")
        bBarMass = getattr(tree, "bBarMass")
        kstBarMass = getattr(tree, "kstBarMass")
      
        truthMatchSignal = [False] * len(dR_list_mum[i])
 
        for idx in range(len(dR_list_mum[i])):
        
            if (dR_list_mum[i][idx] < 0.004 and
                dR_list_mup[i][idx]< 0.004 and
                dR_list_Trkp[i][idx] < 0.01 and
                dR_list_Trkm[i][idx] < 0.01):
                   
                truthMatchSignal[idx] = True
         
#                print(f" truthMatchSignal[{idx}] = {bool(truthMatchSignal[idx])}")

                if genSignal == 1:
                    histograms["h_bTMass"].Fill(bMass[idx])
                    histograms["h_kstTMass"].Fill(kstMass[idx])
                elif genSignal == 2:
                    histograms["h_bTMass"].Fill(bBarMass[idx])
                    histograms["h_kstTMass"].Fill(kstBarMass[idx])

                histograms["h_mumuMass"].Fill(mumuMass[idx])
                histograms["h_bCosAlphaBS"].Fill(bCosAlphaBS[idx])
                histograms["h_bVtxCL"].Fill(bVtxCL[idx])

                if bLBSE[idx] != 0:
                    histograms["h_bLBSs"].Fill(bLBS[idx] / bLBSE[idx])

                if bDCABSE != 0:
                    histograms["h_bDCABSs"].Fill(bDCABS[idx] / bDCABSE[idx])

                if kstTrkpDCABSE != 0:          
                    histograms["h_kstTrkpDCABSs"].Fill(kstTrkpDCABS[idx] / kstTrkpDCABSE[idx])

                if kstTrkmDCABSE != 0:
                    histograms["h_kstTrkmDCABSs"].Fill(kstTrkmDCABS[idx] / kstTrkmDCABSE[idx])

# Fill histograms for both trees
fill_hist(tree1, hist1)
           
# Function to plot a histogram and save as .pdf
def plot_hist(hist1, file_name, title):

    canvas = ROOT.TCanvas("canvas", "Canvas", 800, 600)
    
    hist1.SetTitle(title)

    hist1.Draw("HIST")

    # Create a legend
    legend = ROOT.TLegend(0.8, 0.2, 1.0, 0.3)  # Adjusted to not overlap with stats boxes
    legend.AddEntry(hist1, "Monte Carlo", "l")
    legend.Draw()
    
    # Update the main canvas to reflect changes
    canvas.Update()
    
    # Save the canvas to a file
    canvas.SaveAs(file_name)
    canvas.Close()

# Plot histograms and save them
plot_hist(hist1["h_bTMass"], "h_bTMass.pdf", "True B Mass")
plot_hist(hist1["h_kstTMass"], "h_kstTMass.pdf", "True K* Mass")
plot_hist(hist1["h_mumuMass"], "h_mumuMass.pdf", "MuMu Mass")
plot_hist(hist1["h_bCosAlphaBS"], "h_bCosAlphaBS.pdf", "B CosAlphaBS")
plot_hist(hist1["h_bVtxCL"], "h_bVtxCL.pdf", "B VtxCL")
plot_hist(hist1["h_bLBSs"], "h_bLBSs.pdf", "Significance B LBS")
plot_hist(hist1["h_bDCABSs"], "h_bDCABSs.pdf", "Significance B DCABS")
plot_hist(hist1["h_kstTrkpDCABSs"], "h_kstTrkpDCABSs.pdf", "Significance K* TrkpDCABS")
plot_hist(hist1["h_kstTrkmDCABSs"], "h_kstTrkmDCABSs.pdf", "Significance K* TrkmDCABS")
  
# Save histograms to a new .root file
output_file = TFile("TrueHist.root", "RECREATE")
  
for key, hist in hist1.items():
    hist.Write()
      
output_file.Close()
  
# Close .root file
root_file1.Close()
