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
dir = "~/Data/" # Diogo
#dir = "/user/u/u24gmarujo/root_folder/" #Gonçalo
data_file1 = "B0ToKstMuMu_JpsiMC_22F_0-1531_miniaod.root"
#data_file2 = "B0ToKstMuMu_22F_0-812_miniaod.root"

# Open .root files
root_file1 = TFile.Open(dir + data_file1)
#root_file2 = TFile.Open(dir + data_file2)

if not root_file1 or root_file1.IsZombie():
    print(f"Error: Could not open file {dir + data_file1}")
    exit()
    
#if not root_file2 or root_file2.IsZombie():
#    print(f"Error: Could not open file {dir + data_file2}")
#    exit()

tree1 = root_file1.Get("B0KstMuMu/B0KstMuMuNTuple")
#tree2 = root_file2.Get("B0KstMuMu/B0KstMuMuNTuple")

if not tree1:
    print("Error: Could not find tree 'B0KstMuMuNTuple' in the first file.")
    exit()
    
#if not tree2:
#    print("Error: Could not find tree 'B0KstMuMuNTuple' in the second file.")
#    exit()

# Create histograms for each file
hist1 = {
    "h_bTMass": TH1F("h_bTMass1", "True B Mass", 100, 4.5, 6.0),
    "h_kstTMass": TH1F("h_kstTMass1", "True K* Mass", 100, 0, 3),
    "h_mumuMass": TH1F("h_mumuMass1", "MuMu Mass", 100, 0.8, 11),
    "h_bCosAlphaBS": TH1F("h_bCosAlphaBS1", "B CosAlphaBS", 100, -1.0, 1.0),
    "h_bVtxCL": TH1F("h_bVtxCL1", "B VtxCL", 100, 0, 1.0),
    "h_bLBSs": TH1F("h_bLBSs1", "Significance B LBS", 200, -20, 20),
    "h_bDCABSs": TH1F("h_bDCABSs1", "Significance B DCABS", 200, -20, 20),
    "h_kstTrkpDCABSs": TH1F("h_kstTrkpDCABSs1", "Significance K* TrkpDCABS", 100, -10, 10),
    "h_kstTrkmDCABSs": TH1F("h_kstTrkmDCABSs1", "Significance K* TrkmDCABS", 100, -10, 10),
}

#hist2 = {
#    "h_bTMass": TH1F("h_bTMass2", "True B Mass", 100, 4.5, 6.0),
#    "h_kstTMass": TH1F("h_kstTMass2", "True K* Mass", 100, 0, 3),
#    "h_mumuMass": TH1F("h_mumuMass2", "MuMu Mass", 100, 0.8, 11),
#    "h_bCosAlphaBS": TH1F("h_bCosAlphaBS2", "B CosAlphaBS", 100, -1.0, 1.0),
#    "h_bVtxCL": TH1F("h_bVtxCL2", "B VtxCL", 100, 0, 1.0),
#    "h_bLBSs": TH1F("h_bLBSs2", "Significance B LBS", 100, 0, 20),
#    "h_bDCABSs": TH1F("h_bDCABSs2", "Significance B DCABS", 100, 0, 20),
#    "h_kstTrkpDCABSs": TH1F("h_kstTrkpDCABSs2", "Significance K* TrkpDCABS", 100, 0, 10),
#    "h_kstTrkmDCABSs": TH1F("h_kstTrkmDCABSs2", "Significance K* TrkmDCABS", 100, 0, 10),
#}

# Function to fill histograms from a tree
def fill_hist(tree, histograms):

    for i in range(tree.GetEntries()):
        tree.GetEntry(i)

        truthMatchSignal = getattr(tree, "truthMatchSignal")
        
#        print(f"Entry {i}:")
#        for idx, t_match in enumerate(truthMatchSignal):
#            print(f"  truthMatchSignal[{idx}] = {bool(t_match)}")

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

        for idx, t_match in enumerate(truthMatchSignal):

            if t_match or True:

#                print(f"GenSignal in event {i}:{genSignal}.\n")

                if genSignal == 1:
                    histograms["h_bTMass"].Fill(bMass[idx])
                    histograms["h_kstTMass"].Fill(kstMass[idx])
                elif genSignal == 2:
                    histograms["h_bTMass"].Fill(bBarMass[idx])
                    histograms["h_kstTMass"].Fill(kstBarMass[idx])

                histograms["h_mumuMass"].Fill(mumuMass[idx])
#                print(f"Value of mumu Mass that we are filling the Hist with: {mumuMass[idx]}.\n")
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
#fill_hist(tree2, hist2)

# Normalize the Histograms
#for key, hist in hist1.items():
#    hist.Scale(1./hist.GetMaximum())
 
#for key, hist in hist2.items():
#    hist.Scale(1./hist.GetMaximum())
           
# Function to plot two histograms on the same canvas and save as .pdf
def plot_hist(hist1, file_name, title):
    canvas = ROOT.TCanvas("canvas", "Canvas", 800, 600)
    
    # Set line colors for histograms
#    hist1.SetLineColor(ROOT.kRed)
#    hist2.SetLineColor(ROOT.kBlue)
    
    # Set titles for histograms
    hist1.SetTitle(title)
#    hist2.SetTitle(title)
    
    # Draw histograms on the main canvas
    hist1.Draw("HIST")
#    hist2.Draw("HIST SAME")

#    hist1.SetStats(0)
#    hist2.SetStats(0)
    
    # Calculate statistics for hist1
#    entries1 = hist1.GetEntries()
#    mean1 = hist1.GetMean()
#    stddev1 = hist1.GetStdDev()
    
    # Calculate statistics for hist2
#    entries2 = hist2.GetEntries()
#    mean2 = hist2.GetMean()
#    stddev2 = hist2.GetStdDev()
    
    # Create custom statistics box for hist1
#    stats1 = ROOT.TPaveText(0.6, 0.9, 0.8, 1.0, "NDC")
#    stats1.SetBorderSize(1)
#    stats1.SetTextColor(ROOT.kRed)
#    stats1.AddText(f"Monte Carlo: Entries = {entries1:.0f}")
#    stats1.AddText(f"Mean = {mean1:.3f}")
#    stats1.AddText(f"Std Dev = {stddev1:.3f}")
#    stats1.Draw()
    
    # Create custom statistics box for hist2
#    stats2 = ROOT.TPaveText(0.8, 0.9, 1.0, 1.0, "NDC")
#    stats2.SetBorderSize(1)
#    stats2.SetTextColor(ROOT.kBlue)
#    stats2.AddText(f"Data: Entries = {entries2:.0f}")
#    stats2.AddText(f"Mean = {mean2:.3f}")
#    stats2.AddText(f"Std Dev = {stddev2:.3f}")
#    stats2.Draw()
    
    # Create a legend and add entries with statistics
    legend = ROOT.TLegend(0.8, 0.2, 1.0, 0.3)  # Adjusted to not overlap with stats boxes
    legend.AddEntry(hist1, "Monte Carlo", "l")
#    legend.AddEntry(hist2, "Data", "l")
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
output_file = TFile("VariabHist3.root", "RECREATE")
  
for key, hist in hist1.items():
    hist.Write()
      
#for key, hist in hist2.items():
#    hist.Write()
  
output_file.Close()
  
# Close .root files
root_file1.Close()
#root_file2.Close()
