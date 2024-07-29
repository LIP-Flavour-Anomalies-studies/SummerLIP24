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
data_file1 = "B0ToKstMuMu_JpsiMC_22F_0-1531_miniaod.root"
data_file2 = "B0ToKstMuMu_22F_0-812_miniaod.root"

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

tree1 = root_file1.Get("B0KstMuMu/B0KstMuMuNTuple")
tree2 = root_file2.Get("B0KstMuMu/B0KstMuMuNTuple")

if not tree1:
    print("Error: Could not find tree 'B0KstMuMuNTuple' in the first file.")
    exit()
    
if not tree2:
    print("Error: Could not find tree 'B0KstMuMuNTuple' in the second file.")
    exit()
    
# Create histograms for each parameter
hist1 = {
    "h_bTMass": TH1F("h_bTMass1", "True B Mass", 100, 4.5, 6.0),
    "h_kstTMass": TH1F("h_kstTMass1", "True K* Mass", 100, 0, 3),
    "h_mumuMass": TH1F("h_mumuMass1", "MuMu Mass", 100, 0.8, 11),
    "h_bCosAlphaBS": TH1F("h_bCosAlphaBS1", "B CosAlphaBS", 100, -1.0, 1.0),
    "h_bVtxCL": TH1F("h_bVtxCL1", "B VtxCL", 40, 0, 1.0),
    "h_bLBSs": TH1F("h_bLBSs1", "Significance B LBS", 40, 0, 100),
    "h_bDCABSs": TH1F("h_bDCABSs1", "Significance B DCABS", 100, -20, 20),
    "h_kstTrkpDCABSs": TH1F("h_kstTrkpDCABSs1", "Significance K* TrkpDCABS", 40, -10, 10),
    "h_kstTrkmDCABSs": TH1F("h_kstTrkmDCABSs1", "Significance K* TrkmDCABS", 40, -10, 10),
}

hist2 = {
    "h_bTMass": TH1F("h_bTMass2", "True B Mass", 100, 4.5, 6.0),
    "h_kstTMass": TH1F("h_kstTMass2", "True K* Mass", 100, 0, 3),
    "h_mumuMass": TH1F("h_mumuMass2", "MuMu Mass", 100, 0.8, 11),
    "h_bCosAlphaBS": TH1F("h_bCosAlphaBS2", "B CosAlphaBS", 100, -1.0, 1.0),
    "h_bVtxCL": TH1F("h_bVtxCL2", "B VtxCL", 40, 0, 1.0),
    "h_bLBSs": TH1F("h_bLBSs2", "Significance B LBS", 40, 0, 100),
    "h_bDCABSs": TH1F("h_bDCABSs2", "Significance B DCABS", 100, -20, 20),
    "h_kstTrkpDCABSs": TH1F("h_kstTrkpDCABSs2", "Significance K* TrkpDCABS", 40, -10, 10),
    "h_kstTrkmDCABSs": TH1F("h_kstTrkmDCABSs2", "Significance K* TrkmDCABS", 40, -10, 10),
}

# Function that initializes a Lorentzvector
def TLvector(m, Px, Py, Pz):

    # Create and set the TLorentzVector for the candidate
    lv = TLorentzVector()
    P = np.sqrt(Px**2 + Py**2 + Pz**2)
    E = np.sqrt(P**2 + m**2) 
    lv.SetPxPyPzE(Px, Py, Pz, E)
    
    return lv
    
def Pxyz(particle, tree, i):

    P = []
    tree.GetEntry(i)
    
    # Retrieve the momentum components for the particle candidates
    Px = getattr(tree, particle + "Px")
    Py = getattr(tree, particle + "Py")
    Pz = getattr(tree, particle + "Pz")
   
    P.append(Px)
    P.append(Py)
    P.append(Pz)
    
    return P
    
def flavour_tag(Pm, Pp, j, genSignal):

    # Create TLorentzVectors for both hypotheses
    lvm1 = TLvector(0.140, Pm[0][j], Pm[1][j], Pm[2][j])  # π-
    lvp1 = TLvector(0.494, Pp[0][j], Pp[1][j], Pp[2][j])  # K+

    lvm2 = TLvector(0.494, Pm[0][j], Pm[1][j], Pm[2][j])  # K-
    lvp2 = TLvector(0.140, Pp[0][j], Pp[1][j], Pp[2][j])  # π+
    
    # Calculate invariant masses for each candidate pair
    m1 = (lvm1 + lvp1).M()
    m2 = (lvm2 + lvp2).M()
#    print(f"(m1, m2) -> ({m1}, {m2})")
    
    # Determine the flavour tag based on proximity to the K* mass (0.896 GeV/c²)
    if abs(m1 - 0.896) < abs(m2 - 0.896):
        fl_tag = 1  # K*
#        if genSignal == 2:
 #           print(f"K* mass error -> {abs(m1-0.896)}")
    else:
        fl_tag = 2  # K* Bar
#        if genSignal == 1:
#            print(f"K* Bar mass error -> {abs(m2-0.896)}")

        
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
        genSignal = getattr(tree, "genSignal")
        bMass = getattr(tree, "bMass")
        kstMass = getattr(tree, "kstMass")
        bBarMass = getattr(tree, "bBarMass")
        kstBarMass = getattr(tree, "kstBarMass")
        tMum = getattr(tree, "truthMatchMum")
        tMup = getattr(tree, "truthMatchMup")
        tTrkm = getattr(tree, "truthMatchTrkm")
        tTrkp = getattr(tree, "truthMatchTrkp")
        
        # Retrieve the momentum components for the track candidates
        Pm = Pxyz("kstTrkm", tree, i) 
        Pp = Pxyz("kstTrkp", tree, i)  

        # Loop over each candidate in the event
        for j in range(len(Pm[0])):
        
            fl_tag = flavour_tag(Pm, Pp, j, genSignal)
            tmatch = True
 
#            if not Data:
#                tmatch = (tMum[j] and tMup[j] and tTrkm[j] and tTrkp[j])

            if tmatch:
                                
                if fl_tag == 1:
                    histograms["h_bTMass"].Fill(bMass[j])
                    histograms["h_kstTMass"].Fill(kstMass[j])
#                    if not Data:
#                        if genSignal != 1:
#                            print(f"Event {i}, Candidate {j} -> No gensignal Match!.")
                elif fl_tag == 2:
                    histograms["h_bTMass"].Fill(bBarMass[j])
                    histograms["h_kstTMass"].Fill(kstBarMass[j])
#                    if not Data:
#                        if genSignal != 2:
#                            print(f"Event {i}, Candidate {j} -> No gensignal Match!.")


                histograms["h_mumuMass"].Fill(mumuMass[j])
                histograms["h_bCosAlphaBS"].Fill(bCosAlphaBS[j])
                histograms["h_bVtxCL"].Fill(bVtxCL[j])

                if bLBSE[j] != 0:
                    histograms["h_bLBSs"].Fill(bLBS[j] / bLBSE[j])

                if bDCABSE != 0:
                    histograms["h_bDCABSs"].Fill(bDCABS[j] / bDCABSE[j])

                if kstTrkpDCABSE != 0:          
                    histograms["h_kstTrkpDCABSs"].Fill(kstTrkpDCABS[j] / kstTrkpDCABSE[j])

                if kstTrkmDCABSE != 0:
                    histograms["h_kstTrkmDCABSs"].Fill(kstTrkmDCABS[j] / kstTrkmDCABSE[j])
                
# Fill histograms for both trees
fill_hist(tree1, hist1, False)
fill_hist(tree2, hist2, True)
           
# Normalize the Histograms
for key, hist in hist1.items():
    if hist.GetMaximum() != 0:    
        hist.Scale(1./hist.GetMaximum())
 
for key, hist in hist2.items():
    if hist.GetMaximum() != 0:
        hist.Scale(1./hist.GetMaximum())
           
# Function to plot two histograms on the same canvas and save as .pdf
def plot_hist(hist1, hist2, file_name, title):
    canvas = ROOT.TCanvas("canvas", "Canvas", 800, 600)
    
    # Set line colors for histograms
    hist1.SetLineColor(ROOT.kRed)
    hist2.SetLineColor(ROOT.kBlue)
    
    # Set titles for histograms
    hist1.SetTitle(title)
    hist2.SetTitle(title)
    
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
    stats1 = ROOT.TPaveText(0.6, 0.9, 0.8, 1.0, "NDC")
    stats1.SetBorderSize(1)
    stats1.SetTextColor(ROOT.kRed)
    stats1.AddText(f"Monte Carlo: Entries = {entries1:.0f}")
    stats1.AddText(f"Mean = {mean1:.3f}")
    stats1.AddText(f"Std Dev = {stddev1:.3f}")
    stats1.Draw()
    
    # Create custom statistics box for hist2
    stats2 = ROOT.TPaveText(0.8, 0.9, 1.0, 1.0, "NDC")
    stats2.SetBorderSize(1)
    stats2.SetTextColor(ROOT.kBlue)
    stats2.AddText(f"Real Data: Entries = {entries2:.0f}")
    stats2.AddText(f"Mean = {mean2:.3f}")
    stats2.AddText(f"Std Dev = {stddev2:.3f}")
    stats2.Draw()
    
    # Create a legend and add entries with statistics
    legend = ROOT.TLegend(0.8, 0.2, 1.0, 0.3)  # Adjusted to not overlap with stats boxes
    legend.AddEntry(hist1, "Monte Carlo", "l")
    legend.AddEntry(hist2, "Real Data", "l")
    legend.Draw()
    
    # Save the canvas to a file
    canvas.SaveAs(file_name)
    canvas.Close()

#histograms and save them
plot_hist(hist1["h_bTMass"], hist2["h_bTMass"], "h_bTMass.pdf", "True B Mass")
plot_hist(hist1["h_kstTMass"], hist2["h_kstTMass"], "h_kstTMass.pdf", "True K* Mass")
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
