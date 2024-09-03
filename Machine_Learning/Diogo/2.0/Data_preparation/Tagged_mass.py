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
dir = "/lstore/cms/boletti/Run3-ntuples/"
data_file = "ntuple2_flat_LMNR_PostRefitMomenta_test_2022F_skimSoftMu_1.root" # Data


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
   
   
   
# Function to distinguish Bs from Bbars    
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

        # Filter candidates with truthmatching (only for MC) and background/signal purification cut
        if fl_tag == 1:
            variables["bTMass"][0] = bMass
            variables["kstTMass"][0] = kstMass
        elif fl_tag == 2:
            variables["bTMass"][0] = bBarMass
            variables["kstTMass"][0] = kstBarMass
                
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

'''
------------------------------------- MAIN -----------------------------------------------
'''


# Open .root files
root_file = TFile.Open(dir + data_file)

if not root_file or root_file.IsZombie():
    print(f"Error: Could not open file {dir + data_file}")
    exit()

tree = root_file.Get("ntuple")

if not tree:
    print("Error: Could not find tree 'ntuple' in the file.")
    exit()
 
    
# Output files and TTrees
output_file = TFile("Tagged_mass.root", "RECREATE")
output_tree = TTree("mass_tree", "Tree with selected tagged Bmass.")

# Variables to be stored in the TTrees
variables = {
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
    output_tree.Branch(var, variables[var], f"{var}/D")
    
# Fill TTrees for both input trees
fill_tree(tree, output_tree, True)

# Write and close the output files
output_file.Write()
output_file.Close()

# Close input .root files
root_file.Close()

