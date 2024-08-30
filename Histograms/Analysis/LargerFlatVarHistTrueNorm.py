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
dir = "/lstore/cms/boletti/Run3-ntuples/"
data_file1 = "reco_ntuple2_LMNR_1.root" # Monte Carlo
data_file2 = "ntuple2_flat_LMNR_PostRefitMomenta_test_2022F_skimSoftMu_1.root" # Data

# Avoid unnecessary plots during runtime
ROOT.gROOT.SetBatch(True)

# Open .root files
root_file1 = TFile.Open(dir + data_file1)
root_file2 = TFile.Open(dir + data_file2)

if not root_file1 or root_file1.IsZombie():
    print(f"Error: Could not open file {dir + data_file1}")
    exit()
    
if not root_file2 or root_file2.IsZombie():
    print(f"Error: Could not open file {dir + data_file2}")
    exit()

tree1 = root_file1.Get("ntuple")
tree2 = root_file2.Get("ntuple")

if not tree1:
    print("Error: Could not find tree 'ntuple' in the first file.")
    exit()
    
if not tree2:
    print("Error: Could not find tree 'ntuple' in the second file.")
    exit()
    
# Create histograms for each parameter
hist1 = {
    "h_bTMass": TH1F("h_bTMass1", "Tagged B Mass", 100, 4.5, 6.0),
    "h_kstTMass": TH1F("h_kstTMass1", "Tagged K* Mass", 100, 0.4, 1.6),
    "h_mumuMass": TH1F("h_mumuMass1", "MuMu Mass", 100, 0.8, 6.0),
    "h_bCosAlphaBS": TH1F("h_bCosAlphaBS1", "B CosAlphaBS", 100, -1.0, 1.0),
    "h_bVtxCL": TH1F("h_bVtxCL1", "B VtxCL", 40, 0, 1.0),
    "h_bLBSs": TH1F("h_bLBSs1", "Significance B LBS", 40, 0, 100),
    "h_bDCABSs": TH1F("h_bDCABSs1", "Significance B DCABS", 100, -20, 20),
    "h_kstTrkpDCABSs": TH1F("h_kstTrkpDCABSs1", "Significance K* TrkpDCABS", 100, -10, 10),
    "h_kstTrkmDCABSs": TH1F("h_kstTrkmDCABSs1", "Significance K* TrkmDCABS", 100, -10, 10),
    "h_leadingPt": TH1F("h_leadingPt1", "Leading Pt", 100, 0, 40),
    "h_trailingPt": TH1F("h_trailingPt1", "Trailing Pt", 100, 0, 40),
}

hist2 = {
    "h_bTMass": TH1F("h_bTMass2", "Tagged B Mass", 100, 4.5, 6.0),
    "h_kstTMass": TH1F("h_kstTMass2", "Tagged K* Mass", 100, 0.4, 1.6),
    "h_mumuMass": TH1F("h_mumuMass2", "MuMu Mass", 100, 0.8, 6.0),
    "h_bCosAlphaBS": TH1F("h_bCosAlphaBS2", "B CosAlphaBS", 100, -1.0, 1.0),
    "h_bVtxCL": TH1F("h_bVtxCL2", "B VtxCL", 40, 0, 1.0),
    "h_bLBSs": TH1F("h_bLBSs2", "Significance B LBS", 40, 0, 100),
    "h_bDCABSs": TH1F("h_bDCABSs2", "Significance B DCABS", 100, -20, 20),
    "h_kstTrkpDCABSs": TH1F("h_kstTrkpDCABSs2", "Significance K* TrkpDCABS", 100, -10, 10),
    "h_kstTrkmDCABSs": TH1F("h_kstTrkmDCABSs2", "Significance K* TrkmDCABS", 100, -10, 10),
    "h_leadingPt": TH1F("h_leadingPt2", "Leading Pt", 100, 0, 40),
    "h_trailingPt": TH1F("h_trailingPt2", "Trailing Pt", 100, 0, 40),
}

# Function that initializes a Lorentzvector
def TLvector(m, pt, eta, phi):

    # Create and set the TLorentzVector for the candidate
    lv = TLorentzVector()
    lv.SetPtEtaPhiM(pt, eta, phi, m)
    
    return lv

# Function that returns a vector with the cylindrical coordinates (pt, eta and phi), for each candidate in event i  
def cyl_coord(particle, tree, i):

    C = []
    tree.GetEntry(i)
    
    # Retrieve the cylindrical coordinates for the particle candidates
    pt = getattr(tree, particle + "Pt")
    eta = getattr(tree, particle + "Eta")
    phi = getattr(tree, particle + "Phi")
   
    C.append(pt)
    C.append(eta)
    C.append(phi)
  
    return C
    
def flavour_tag(Pm, Pp):

    # Create TLorentzVectors for both hypotheses
    lvm1 = TLvector(0.140, Pm[0], Pm[1], Pm[2])  # π-
    lvp1 = TLvector(0.494, Pp[0], Pp[1], Pp[2])  # K+

    lvm2 = TLvector(0.494, Pm[0], Pm[1], Pm[2])  # K-
    lvp2 = TLvector(0.140, Pp[0], Pp[1], Pp[2])  # π+
    
    # Calculate invariant masses for each candidate pair
    m1 = (lvm1 + lvp1).M()
    m2 = (lvm2 + lvp2).M()
    
    # Determine the flavour tag based on proximity to the K* mass (0.896 GeV/c²)
    if abs(m1 - 0.896) < abs(m2 - 0.896):
        fl_tag = 1  # K*
    else:
        fl_tag = 2  # K* Bar
        
    return fl_tag
    
    
# Function to fill histograms from a tree
def fill_hist(tree, histograms, Data):

    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        
        # Retrieve variables of interest
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
        bMass = getattr(tree, "bMass")
        kstMass = getattr(tree, "kstMass")
        bBarMass = getattr(tree, "bBarMass")
        kstBarMass = getattr(tree, "kstBarMass")
        mumPt = getattr(tree, "mumPt")
        mupPt = getattr(tree, "mupPt")
       
        # Retrieve the momentum components for the track candidates
        Cm = cyl_coord("kstTrkm", tree, i) 
        Cp = cyl_coord("kstTrkp", tree, i)  
        
        fl_tag = flavour_tag(Cm, Cp)
        tmatch = True
 
        if not Data:
            tMum = getattr(tree, "truthMatchMum")
            tMup = getattr(tree, "truthMatchMup")
            tTrkm = getattr(tree, "truthMatchTrkm")
            tTrkp = getattr(tree, "truthMatchTrkp")
            tmatch = (tMum and tMup and tTrkm and tTrkp)

        if tmatch:
                                
            if fl_tag == 1:
                histograms["h_bTMass"].Fill(bMass)
                histograms["h_kstTMass"].Fill(kstMass)
                
            elif fl_tag == 2:
                histograms["h_bTMass"].Fill(bBarMass)
                histograms["h_kstTMass"].Fill(kstBarMass)
                
            histograms["h_mumuMass"].Fill(mumuMass)
            histograms["h_bCosAlphaBS"].Fill(bCosAlphaBS)
            histograms["h_bVtxCL"].Fill(bVtxCL)

            if bLBSE != 0:
                histograms["h_bLBSs"].Fill(bLBS / bLBSE)

            if bDCABSE != 0:
                histograms["h_bDCABSs"].Fill(bDCABS / bDCABSE)

            if kstTrkpDCABSE != 0:          
                histograms["h_kstTrkpDCABSs"].Fill(kstTrkpDCABS / kstTrkpDCABSE)

            if kstTrkmDCABSE != 0:
                histograms["h_kstTrkmDCABSs"].Fill(kstTrkmDCABS / kstTrkmDCABSE)
                
            if mumPt > mupPt:
                histograms["h_leadingPt"].Fill(mumPt)
                histograms["h_trailingPt"].Fill(mupPt)
            else:
                histograms["h_leadingPt"].Fill(mupPt)
                histograms["h_trailingPt"].Fill(mumPt)
                
# Fill histograms for both trees
fill_hist(tree1, hist1, False)
fill_hist(tree2, hist2, True)
           
# Normalize the Histograms
for key, hist in hist1.items():
    if hist.Integral() != 0:    
        hist.Scale(1./hist.Integral())
 
for key, hist in hist2.items():
    if hist.Integral() != 0:
        hist.Scale(1./hist.Integral())
           
# Function to plot two histograms on the same canvas and save as .pdf
def plot_hist(hist1, hist2, file_name, title):
    canvas = ROOT.TCanvas("canvas", "Canvas", 800, 600)
    
    # Set line colors for histograms
    hist1.SetLineColor(ROOT.kRed)
    hist2.SetLineColor(ROOT.kBlue)
    
    # Set titles for histograms
    hist1.SetTitle(title)
    hist2.SetTitle(title)
    
    # Set the y-axis maximum to include all data
    max_y = max(hist1.GetMaximum(), hist2.GetMaximum())
    hist1.SetMaximum(1.05 * max_y)
    hist2.SetMaximum(1.05 * max_y)

    # Draw histograms on the main canvas
    hist1.Draw("HIST")
    hist2.Draw("HIST SAME")

    hist1.SetStats(0)
    hist2.SetStats(0)
    
    # Calculate statistics for hist1
    entries1 = hist1.GetEntries()
    mean1 = hist1.GetMean()
    stddev1 = hist1.GetStdDev()
    
    # Calculate statistics for hist2
    entries2 = hist2.GetEntries()
    mean2 = hist2.GetMean()
    stddev2 = hist2.GetStdDev()
    
    # Create custom statistics box for hist1
    stats1 = ROOT.TPaveText(0.6, 0.8, 0.8, 0.9, "NDC")
    stats1.SetBorderSize(1)
    stats1.SetTextColor(ROOT.kRed)
    stats1.AddText(f"Entries = {entries1:.0f}")
    stats1.AddText(f"Mean = {mean1:.3f}")
    stats1.AddText(f"Std. Dev. = {stddev1:.3f}")
    stats1.Draw()
    
    # Create custom statistics box for hist2
    stats2 = ROOT.TPaveText(0.8, 0.8, 1.0, 0.9, "NDC")
    stats2.SetBorderSize(1)
    stats2.SetTextColor(ROOT.kBlue)
    stats2.AddText(f"Entries = {entries2:.0f}")
    stats2.AddText(f"Mean = {mean2:.3f}")
    stats2.AddText(f"Std. Dev. = {stddev2:.3f}")
    stats2.Draw()
    
    # Create a legend and add entries with statistics
    legend = ROOT.TLegend(0.8, 0.4, 1.0, 0.5)  # Adjusted to not overlap with stats boxes
    legend.AddEntry(hist1, "Monte Carlo", "l")
    legend.AddEntry(hist2, "Data", "l")
    legend.Draw()
    
    # Save the canvas to a file
    canvas.SaveAs(file_name)
    canvas.Close()

#histograms and save them
plot_hist(hist1["h_bTMass"], hist2["h_bTMass"], "h_bTMass.pdf", "Tagged B Mass")
plot_hist(hist1["h_kstTMass"], hist2["h_kstTMass"], "h_kstTMass.pdf", "Tagged K* Mass")
plot_hist(hist1["h_mumuMass"], hist2["h_mumuMass"], "h_mumuMass.pdf", "MuMu Mass")
plot_hist(hist1["h_bCosAlphaBS"], hist2["h_bCosAlphaBS"], "h_bCosAlphaBS.pdf", "B CosAlphaBS")
plot_hist(hist1["h_bVtxCL"], hist2["h_bVtxCL"], "h_bVtxCL.pdf", "B VtxCL")
plot_hist(hist1["h_bLBSs"], hist2["h_bLBSs"], "h_bLBSs.pdf", "Significance B LBS")
plot_hist(hist1["h_bDCABSs"], hist2["h_bDCABSs"], "h_bDCABSs.pdf", "Significance B DCABS")
plot_hist(hist1["h_kstTrkpDCABSs"], hist2["h_kstTrkpDCABSs"], "h_kstTrkpDCABSs.pdf", "Significance K* TrkpDCABS")
plot_hist(hist1["h_kstTrkmDCABSs"], hist2["h_kstTrkmDCABSs"], "h_kstTrkmDCABSs.pdf", "Significance K* TrkmDCABS")
plot_hist(hist1["h_leadingPt"], hist2["h_leadingPt"], "h_leadingPt.pdf", "Leading Pt")
plot_hist(hist1["h_trailingPt"], hist2["h_trailingPt"], "h_trailingPt.pdf", "Trailing Pt")
  
# Save histograms to a new .root file
output_file = TFile("LargerTMtchFlTgg.root", "RECREATE")
  
for key, hist in hist1.items():
    hist.Write()
      
for key, hist in hist2.items():
    hist.Write()
  
output_file.Close()
  
# Close .root files
root_file1.Close()
root_file2.Close()
