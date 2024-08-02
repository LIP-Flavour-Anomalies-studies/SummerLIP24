"""
Created on July 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import ROOT
from ROOT import TFile, TH1F, TCanvas, TLegend, TLorentzVector
import numpy as np
import torch

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

"""
--------------------------------------------------||------------------------------------------------------     
"""     

def classifier(fl_tag, Data, bMass, bBarMass, tree):

     # Check whether events are Signal or Background
     if fl_tag == 1:
            background = (Data == 1) and ((5.0 < bMass < 5.133542769) or (5.416657231 < bMass < 5.6))
            signal = (Data == 0) and (5.133542769 <= bMass <= 5.416657231)

     if fl_tag == 2:
            background = (Data == 1) and ((5.0 < bBarMass < 5.133542769) or (5.416657231 < bBarMass < 5.6))
            signal = (Data == 0) and (5.133542769 <= bBarMass <= 5.416657231)

     # Apply truthmatch filter on Monte Carlo events   
     tmatch = True
     if signal:
         tMum = getattr(tree, "truthMatchMum")
         tMup = getattr(tree, "truthMatchMup")
         tTrkm = getattr(tree, "truthMatchTrkm")
         tTrkp = getattr(tree, "truthMatchTrkp")
         tmatch = (tMum and tMup and tTrkm and tTrkp)
          
     return tmatch, signal, background

"""
--------------------------------------------------||------------------------------------------------------     
"""     

def SignBck_array(trees, branches):

    rows = []
    nback = 0
    nsignal = 0
    
    for Data, tree in enumerate(trees):
    
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
       
            # Retrieve the momentum components in cylindical coordinates for the track candidates
            Cm = cyl_coord("kstTrkm", tree, i) 
            Cp = cyl_coord("kstTrkp", tree, i)  
        
            # Determine the most likely flavour for this candidate
            fl_tag = flavour_tag(Cm, Cp)

            # Classify samples as background or signal
            tmatch, signal, background = classifier(fl_tag, Data, bMass, bBarMass, tree)

            # Filter candidates with truthmatching (only for MC) and background/signal purification cut
            if tmatch and (signal or background):

                current_row = []
                
                if fl_tag == 1:
                    current_row.append(kstMass)

                elif fl_tag == 2:
                    current_row.append(kstBarMass)

                current_row.append(bCosAlphaBS)
                current_row.append(bVtxCL)

                if bLBSE != 0:
                    current_row.append(bLBS / bLBSE)

                if bDCABSE != 0:
                    current_row.append(bDCABS / bDCABSE)

                if kstTrkpDCABSE != 0:          
                    current_row.append(kstTrkpDCABS / kstTrkpDCABSE)

                if kstTrkmDCABSE != 0:
                    current_row.append(kstTrkmDCABS / kstTrkmDCABSE)
                
                if mumPt > mupPt:
                    current_row.append(mumPt)
                    current_row.append(mupPt)
                else:
                    current_row.append(mupPt)
                    current_row.append(mumPt)
                
                if Data == 0:
                    nsignal += 1
                if Data == 1:
                    nback += 1
                        
                rows.append(current_row)
    
    # Define x -> array containing all features for each event
    x = np.array(rows)
    
    # Define y -> array labeling all features of each event as signal correspondent (1) or background (0)            
    y = np.zeros((nsignal + nback, len(branches))) 
    y[:nsignal] = 1
    
    return x, y
       
"""
--------------------------------------------------||------------------------------------------------------     
"""                 

def prepdata(data, data_mc):
                
    treeS = data_mc.Get("ntuple")
    treeB = data.Get("ntuple")
    trees = [treeS, treeB]
    branches = ["kstTMass", "bCosAlphaBS", "bVtxCL", "bLBSs", "bDCABSs", "kstTrkpDCABSs", "kstTrkmDCABSs", "leadingPt", "trailingPt"]
    
    x, y = SignBck_array(trees, branches)
    
    return x, y, branches    
    
"""
--------------------------------------------------||------------------------------------------------------     
"""     

class RegressionDataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):

        """
        data: the dict returned by utils.prepdata
        """
        
        train_X = data
        train_y = labels
        
        self.X = torch.tensor(train_X, dtype=torch.float32)
        self.y = torch.tensor(train_y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

