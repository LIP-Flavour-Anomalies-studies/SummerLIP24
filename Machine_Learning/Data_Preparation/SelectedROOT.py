# -*- coding: utf-8 -*-
"""
Created on August 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import ROOT
from ROOT import TFile, TTree, TLorentzVector
import numpy as np

# Input path
dir = "/user/b/boletti/Run3-samples/"
data_file1 = "reco_ntuple_LMNR_1.root" # Monte Carlo
data_file2 = "ntuple_flat_LMNR_PostRefitMomenta_test_2022F_skimSoftMu_1.root" # Data

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
    
# Output files and TTrees
output_file1 = TFile("MC.root", "RECREATE")
output_tree1 = TTree("signal_tree", "Tree with selected variables from Monte Carlo")

output_file2 = TFile("ED.root", "RECREATE")
output_tree2 = TTree("background_tree", "Tree with selected variables from experimental data")

# Variables to be stored in the TTrees
variables = {
    "mumuMass": np.zeros(1, dtype=float),
    "bCosAlphaBS": np.zeros(1, dtype=float),
    "bVtxCL": np.zeros(1, dtype=float),
    "bLBSs": np.zeros(1, dtype=float),
    "bDCABSs": np.zeros(1, dtype=float),
    "kstTrkpDCABSs": np.zeros(1, dtype=float),
    "kstTrkmDCABSs": np.zeros(1, dtype=float),
    "bTMass": np.zeros(1, dtype=float),
    "kstTMass": np.zeros(1, dtype=float),
    "leadingPt": np.zeros(1, dtype=float),
    "trailingPt": np.zeros(1, dtype=float),
}

# Create branches in the TTrees
for var in variables:
    output_tree1.Branch(var, variables[var], f"{var}/D")
    output_tree2.Branch(var, variables[var], f"{var}/D")

# Function that initializes a Lorentzvector
def TLvector(m, pt, eta, phi):
    lv = TLorentzVector()
    lv.SetPtEtaPhiM(pt, eta, phi, m)
    return lv

# Function that returns a vector with the cylindrical coordinates (pt, eta and phi), for each candidate in event i  
def cyl_coord(particle, tree, i):
    C = []
    tree.GetEntry(i)
    pt = getattr(tree, particle + "Pt")
    eta = getattr(tree, particle + "Eta")
    phi = getattr(tree, particle + "Phi")
    C.append(pt)
    C.append(eta)
    C.append(phi)
    return C
    
def flavour_tag(Pm, Pp):
    lvm1 = TLvector(0.140, Pm[0], Pm[1], Pm[2])  # π-
    lvp1 = TLvector(0.494, Pp[0], Pp[1], Pp[2])  # K+
    lvm2 = TLvector(0.494, Pm[0], Pm[1], Pm[2])  # K-
    lvp2 = TLvector(0.140, Pp[0], Pp[1], Pp[2])  # π+
    m1 = (lvm1 + lvp1).M()
    m2 = (lvm2 + lvp2).M()
    if abs(m1 - 0.896) < abs(m2 - 0.896):
        fl_tag = 1  # K*
    else:
        fl_tag = 2  # K* Bar
    return fl_tag

# Function to fill TTrees from a tree
def fill_tree(tree, output_tree, Data):
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
       
        # Retrieve the momentum components in cylindrical coordinates for the track candidates
        Cm = cyl_coord("kstTrkm", tree, i) 
        Cp = cyl_coord("kstTrkp", tree, i)  
        
        # Determine the most likely flavour for this candidate
        fl_tag = flavour_tag(Cm, Cp)
        tmatch = True

        # Classify samples as background or signal
        if fl_tag == 1:
            background = Data and ((5.0 < bMass < 5.133542769) or (5.416657231 < bMass < 5.6))
            signal = not Data and (5.133542769 <= bMass <= 5.416657231)

        if fl_tag == 2:
            background = Data and ((5.0 < bBarMass < 5.133542769) or (5.416657231 < bBarMass < 5.6))
            signal = not Data and (5.133542769 <= bBarMass <= 5.416657231)

        if not Data:
            tMum = getattr(tree, "truthMatchMum")
            tMup = getattr(tree, "truthMatchMup")
            tTrkm = getattr(tree, "truthMatchTrkm")
            tTrkp = getattr(tree, "truthMatchTrkp")
            tmatch = (tMum and tMup and tTrkm and tTrkp)

        # Filter candidates with truthmatching (only for MC) and background/signal purification cut
        if tmatch and (signal or background):
            if fl_tag == 1:
                variables["bTMass"][0] = bMass
                variables["kstTMass"][0] = kstMass
            elif fl_tag == 2:
                variables["bTMass"][0] = bBarMass
                variables["kstTMass"][0] = kstBarMass
                
            variables["mumuMass"][0] = mumuMass
            variables["bCosAlphaBS"][0] = bCosAlphaBS
            variables["bVtxCL"][0] = bVtxCL
            
            if bLBSE != 0:
                variables["bLBSs"][0] = bLBS / bLBSE

            if bDCABSE != 0:
                variables["bDCABSs"][0] = bDCABS / bDCABSE

            if kstTrkpDCABSE != 0:
                variables["kstTrkpDCABSs"][0] = kstTrkpDCABS / kstTrkpDCABSE

            if kstTrkmDCABSE != 0:
                variables["kstTrkmDCABSs"][0] = kstTrkmDCABS / kstTrkmDCABSE
                
            if mumPt > mupPt:
                variables["leadingPt"][0] = mumPt
                variables["trailingPt"][0] = mupPt
            else:
                variables["leadingPt"][0] = mupPt
                variables["trailingPt"][0] = mumPt

            # Fill the TTree with the variables
            output_tree.Fill()

# Fill TTrees for both input trees
fill_tree(tree1, output_tree1, False)
fill_tree(tree2, output_tree2, True)

# Write and close the output files
output_file1.Write()
output_file1.Close()

output_file2.Write()
output_file2.Close()

# Close input .root files
root_file1.Close()
root_file2.Close()
