"""
Created on August 2024

@author: Diogo Pereira
Gonçalo Marujo

LIP Internship Program | Flavour Anomalies
"""

import torch
import ROOT
import numpy as np
import matplotlib.pyplot as plt



def plot_prob_distribution(targets, probabilities, best_thresh, name):
    prob = np.array(probabilities)
    targets = np.array(targets)

    plt.figure(figsize=(8, 6))

    # Background predictions
    background_predict = prob[targets == 0]
    plt.hist(background_predict, bins=40, range=(0,1), density=True, alpha=0.5, label="Background (ED)", color="red", hatch="//", edgecolor="black")
    
    # Signal predictions
    signal_predict = prob[targets == 1]
    plt.hist(signal_predict, bins=40, range=(0,1), density=True, alpha=0.7, label="Signal (MC)", color="blue")
    
    plt.axvline(x=best_thresh, color='grey', lw=2, label=f'Threshold = {best_thresh:.2f}')
    plt.xlabel("Predicted Probability", fontsize=14, labelpad=15)
    plt.ylabel("Normalized Density", fontsize=14, labelpad=15) 
    plt.legend()
    plt.savefig(f"{name}_prob_distribution.pdf")  # Save the plot as a PDF file
    plt.close()





def scatter_plots(test_loader, probabilities, targets, filename):

    # Convert probabilities to a numpy array
    prob = np.array(probabilities)
    targets = np.array(targets)
    
    # Split probabilities based on target labels
    signal_predict = prob[targets == 1]
    background_predict = prob[targets == 0]
    
    # Define feature columns
    columns = ["kstTMass", "bCosAlphaBS", "bVtxCL", "bLBSs", "bDCABSs", "kstTrkpDCABSs", "kstTrkmDCABSs", "leadingPt", "trailingPt"]

    for i in range(len(columns)):
        variables = []
        with torch.no_grad():
            name = columns[i]

            # Collect the feature values for the i-th column
            for inputs, labels in test_loader:
                variables.extend(inputs[:, i].cpu().numpy())
                   
            variables = np.array(variables)
           
        # Split feature values based on target label
        signal_variable = variables[targets == 1]
        background_variable = variables[targets == 0]
           
   
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(signal_variable, signal_predict, marker='o', color='blue', label="Signal (MC)")
        plt.scatter(background_variable, background_predict, marker='o', color='red', alpha = 0.5, label="Background (DATA)")    
        plt.xlabel(name, fontsize=14, labelpad=15)
        plt.ylabel("Predicted Probability", fontsize=14, labelpad=15) 
        plt.legend()
        plt.savefig(f"{filename}_{name}_scatter_plot.pdf")  # Save the plot as a PDF file
        plt.close()


 
 
def plot_hist(data1, data2, name, model_name):

    ROOT.gROOT.SetBatch(True)

    x_min = min(np.min(data1), np.min(data2))
    x_max = max(np.max(data1), np.max(data2))
    n_bins = 10

    # Create the histograms (n_bins, x_min, x_max)
    hist1 = ROOT.TH1F("hist1", f"Signal {name}", n_bins, x_min, x_max)
    hist2 = ROOT.TH1F("hist2", f"Background {name}", n_bins, x_min, x_max)

    # Fill the histograms with data
    for value in data1:
        hist1.Fill(value)

    for value in data2:
        hist2.Fill(value)

    if hist1.Integral() != 0:    
        hist1.Scale(1./hist1.Integral())
 
    if hist2.Integral() != 0:
        hist2.Scale(1./hist2.Integral())

    canvas = ROOT.TCanvas("canvas", "Canvas", 800, 600)
    
    # Set line colors for histograms
    hist1.SetLineColor(ROOT.kRed)
    hist2.SetLineColor(ROOT.kBlue)
    
    # Set titles for histograms
    hist1.SetTitle(name)
    
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
    
    # Create custom statistics box
    stats1 = ROOT.TPaveText(0.6, 0.8, 0.8, 0.9, "NDC")
    stats1.SetBorderSize(1)
    stats1.SetTextColor(ROOT.kRed)
    stats1.AddText(f"Signal: Entries = {entries1:.0f}")
    stats1.AddText(f"Mean = {mean1:.3f}")
    stats1.AddText(f"Std Dev = {stddev1:.3f}")
    stats1.Draw()
    
    stats2 = ROOT.TPaveText(0.8, 0.8, 1.0, 0.9, "NDC")
    stats2.SetBorderSize(1)
    stats2.SetTextColor(ROOT.kBlue)
    stats2.AddText(f"Background: Entries = {entries2:.0f}")
    stats2.AddText(f"Mean = {mean2:.3f}")
    stats2.AddText(f"Std Dev = {stddev2:.3f}")
    stats2.Draw()
    
    # Create a legend and add entries with statistics
    legend = ROOT.TLegend(0.8, 0.2, 1.0, 0.3)  # Adjusted to not overlap with stats boxes
    legend.AddEntry(hist1, "Signal", "l")
    legend.AddEntry(hist2, "Background", "l")
    legend.Draw()
    
    # Save the canvas to a file
    canvas.SaveAs(f"{model_name}_{name}_hist.pdf")
    canvas.Close()





def variable_histograms(test_loader, predictions, x, model_name):

    # Define feature columns
    columns = ["kstTMass", "bCosAlphaBS", "bVtxCL", "bLBSs", "bDCABSs", "kstTrkpDCABSs", "kstTrkmDCABSs", "leadingPt", "trailingPt"]
    predictions = np.array(predictions)

    for i in range(len(columns)):
        variable = []
        with torch.no_grad():
            name = columns[i]

            # Collect the feature values for the i-th column
            for inputs, labels in test_loader:
                variable.extend(inputs[:, i].cpu().numpy())
                   
            variable = np.array(variable)
           
        # Split feature values based on prediction
        signal_variable = variable[predictions == 1]
        background_variable = variable[predictions == 0]
 
        plot_hist(signal_variable, background_variable, name, model_name)
        
    Bmass_bck = x[predictions == 0]
    Bmass_sign = x[predictions == 1]

    plot_hist(Bmass_sign, Bmass_bck, "bTMass", model_name)
  
  
  
