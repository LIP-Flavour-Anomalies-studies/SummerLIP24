# -*- coding: utf-8 -*-
"""
Created on July 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import ROOT
from ROOT import TFile, TH1F, TLorentzVector
import numpy as np

# Input path
dir = "~/Data/" # Diogo
# dir = "/user/u/u24gmarujo/root_fl/" #Gonçalo
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

# Create histogram for DeltaR
hist_bdR = TH1F("h_bdR", "B Delta R", 60, 0, 0.01)
hist_kstdR = TH1F("h_kstdR", "K* Delta R", 100, 0, 0.02)
hist_mumdR = TH1F("h_mumdR", "mum Delta R", 100, 0, 0.004)
hist_mupdR = TH1F("h_mupdR", "mup Delta R", 100, 0, 0.004)
hist_kstTrkpdR = TH1F("h_kstTrkpdR", "Track+ Delta R", 100, 0, 0.02)
hist_kstTrkmdR = TH1F("h_kstTrkmdR", "Track- Delta R", 100, 0, 0.02)
    
def create_TLvector(m, Px, Py, Pz):

    # Create and set the TLorentzVector for the candidate
    lv = TLorentzVector()
    P = np.sqrt(Px**2 + Py**2 + Pz**2)
    E = np.sqrt(P**2 + m**2) 
    lv.SetPxPyPzE(Px, Py, Pz, E)
    
    return lv

def fill_hist_dR(m, particle, gparticle, tree, hist):

    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        
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
           hist.Fill(dR)
        
# Possible values for m: 5.278, 0.892 and 0.105 
# Possible values for particle and gparticle: "b", "genB0", "kst", "genKst", "mum", "genMum", "mup", "genMup", 
# "kstTrkp", "genKstTrkp", "kstTrkm" and "genKstTrkm"

# Fill histogram
fill_hist_dR(5.278, "b", "genB0", tree1, hist_bdR)
fill_hist_dR(0.892, "kst", "genKst", tree1, hist_kstdR)
fill_hist_dR(0.105, "mum", "genMum", tree1, hist_mumdR)
fill_hist_dR(0.105, "mup", "genMup", tree1, hist_mupdR)
fill_hist_dR(0.0, "kstTrkp", "genKstTrkp", tree1, hist_kstTrkpdR)        
fill_hist_dR(0.0, "kstTrkm", "genKstTrkm", tree1, hist_kstTrkmdR)

# Function to plot two histograms on the same canvas and save as .pdf
def plot_hist(hist, file_name, title):
    canvas = ROOT.TCanvas("canvas", "Canvas", 800, 600)
    
    # Set logarithmic scale for x and y-axis
    # canvas.SetLogx()
    # canvas.SetLogy()
       
    hist.SetTitle(title)
    hist.Draw("HIST")
    
    # Save the canvas to a file
    canvas.SaveAs(file_name)
    canvas.Close()

# Plot histograms and save them
plot_hist(hist_bdR, "h_bdR.pdf", "B Delta R")
plot_hist(hist_kstdR, "h_kstdR.pdf", "K* Delta R")
plot_hist(hist_mumdR, "h_mumdR.pdf", "mum Delta R")
plot_hist(hist_mupdR, "h_mupdR.pdf", "mup Delta R")
plot_hist(hist_kstTrkpdR, "h_kstTrkpdR.pdf", "Track+ Delta R")
plot_hist(hist_kstTrkmdR, "h_kstTrkmdR.pdf", "Track- Delta R")

# Save histograms to a new .root file
output_file = TFile("DeltaR.root", "RECREATE")
  
hist_bdR.Write()
hist_kstdR.Write()
hist_mumdR.Write()
hist_mupdR.Write()
hist_kstTrkpdR.Write()
hist_kstTrkmdR.Write()
      
# for key, hist in hist2.items():
#    hist.Write()
  
output_file.Close()
  
# Close .root file
root_file1.Close()
