# -*- coding: utf-8 -*-
"""
Created on July 2024
 
@author: Diogo Pereira
Gonçalo Marujo
 
LIP Internship Program | Flavour Anomalies
"""
 
import ROOT
from ROOT import TFile, TH1F, TCanvas
 
# Input path
dir = "~/Data/" # Diogo
# dir = "/user/u/u24gmarujo/root_folder/" #Gonçalo
data_file = "B0ToKstMuMu_JpsiMC_22F_0-1531_miniaod.root"
 
# Opening data ROOT file
root_file = TFile.Open(dir + data_file)
 
if not root_file or root_file.IsZombie():
    print(f"Error: Could not open file {dir + data_file}")
    exit()
 
tree = root_file.Get("B0KstMuMu/B0KstMuMuNTuple")
 
if not tree:
    print("Error: Could not find tree 'B0KstMuMuNTuple' in the file.")
    exit()
 
# Create histograms
h_bMass = TH1F("h_bMass", "B Mass", 100, 4.5, 6.0)
h_bBarMass = TH1F("h_bBarMass", "B Bar Mass", 100, 4.5, 6.0)
h_kstMass = TH1F("h_kstMass", "K* Mass", 100, 0, 3)
h_kstBarMass = TH1F("h_kstBarMass", "K* Bar Mass", 100, 0, 3)
h_mumuMass = TH1F("h_mumuMass", "MuMu Mass", 100, 0.8, 11)
h_bCosAlphaBS = TH1F("h_bCosAlphaBS", "B CosAlphaBS", 100, -1.0, 1.0)
h_bVtxCL = TH1F("h_bVtxCL", "B VtxCL", 100, 0, 1.0)
h_sbLBS = TH1F("h_sbLBS", "Significance B LBS", 100, 0, 20)
h_sbDCABS = TH1F("h_sbDCABS", "Significance B DCABS", 100, 0, 20)
h_skstTrkpDCABS = TH1F("h_skstTrkpDCABS", "Significance K* TrkpDCABS", 100, 0, 20)
h_skstTrkmDCABS = TH1F("h_skstTrkmDCABS", "Significance K* TrkmDCABS", 100, 0, 20)
 
# Loop over the entries in the TTree and fill histograms
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
        h_bMass.Fill(value)
        
    for value in bBarMass:
        h_bBarMass.Fill(value)
        
    for value in kstMass:
         h_kstMass.Fill(value)
         
    for value in kstMass:
         h_kstBarMass.Fill(value)
         
    for value in mumuMass:
        h_mumuMass.Fill(value)
 
    for value in bCosAlphaBS:
        h_bCosAlphaBS.Fill(value)
        
    for value in bVtxCL:
        h_bVtxCL.Fill(value)
        
    for value1, value2 in zip(bLBS, bLBSE):
        if value2 != 0:       
            h_sbLBS.Fill(value1/value2)
            
    for value1, value2 in zip(bDCABS, bDCABSE):
        if value2 != 0:       
            h_sbDCABS.Fill(value1/value2)
            
    for value1, value2 in zip(kstTrkpDCABS, kstTrkpDCABSE):
        if value2 != 0:       
            h_skstTrkpDCABS.Fill(value1/value2)
            
    for value1, value2 in zip(kstTrkmDCABS, kstTrkmDCABSE):
        if value2 != 0:       
            h_skstTrkmDCABS.Fill(value1/value2)
            
def hist_to_pdf(hist, file_name):
    canvas = TCanvas()
    hist.Draw()
    canvas.SaveAs(file_name)
    canvas.Close()

# Save histograms to separate PDF files
hist_to_pdf(h_bMass, "h_bMass.pdf")
hist_to_pdf(h_bBarMass, "h_bBarMass.pdf")
hist_to_pdf(h_kstMass, "h_kstMass.pdf")
hist_to_pdf(h_kstBarMass, "h_kstBarMass.pdf")
hist_to_pdf(h_mumuMass, "h_mumuMass.pdf")
hist_to_pdf(h_bCosAlphaBS, "h_bCosAlphaBS.pdf")
hist_to_pdf(h_bVtxCL, "h_bVtxCL.pdf")
hist_to_pdf(h_sbLBS, "h_sbLBS.pdf")
hist_to_pdf(h_sbDCABS, "h_sbDCABS.pdf")
hist_to_pdf(h_skstTrkpDCABS, "h_skstTrkpDCABS.pdf")
hist_to_pdf(h_skstTrkmDCABS, "h_skstTrkmDCABS.pdf")

# Save histograms to a new ROOT file
output_file = TFile("VariabHist.root", "RECREATE")

h_bMass.Write()
h_bBarMass.Write()
h_kstMass.Write()
h_kstBarMass.Write()
h_mumuMass.Write()
h_bCosAlphaBS.Write()
h_bVtxCL.Write()
h_sbLBS.Write()
h_sbDCABS.Write()
h_skstTrkpDCABS.Write()
h_skstTrkmDCABS.Write()

output_file.Close()

# Close the input ROOT file
root_file.Close()
