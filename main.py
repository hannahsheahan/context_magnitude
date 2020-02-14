"""
 This is a set of simulations for training a simple MLP or RNN on a relational magnitude problem
 The network will be trained to answer the question: is input 2 > input 1?

 Author: Hannah Sheahan, sheahan.hannah@gmail.com
 Date: 04/12/2019
 Notes:
 - requires ffmpeg for 3D animation generation in generatePlots()
 Issues: N/A
"""
# ---------------------------------------------------------------------------- #
 # my project-specific namespaces
import magnitude_network as mnet
import define_dataset as dset
import MDSplotter as MDSplt

import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import copy
from sklearn.manifold import MDS
from sklearn.utils import shuffle
from importlib import reload
from mpl_toolkits import mplot3d
from matplotlib import animation
import json
import time

# network stuff
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from itertools import product
from datetime import datetime
import argparse

# ---------------------------------------------------------------------------- #

def trainAndSaveANetwork(params, createNewDataset):
    # define the network parameters
    args, device, multiparams = mnet.defineHyperparams() # training hyperparams for network (passed as args when called from command line)
    datasetname, trained_modelname, analysis_name, _ = mnet.getDatasetName(args, *params)
    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState = params

    if createNewDataset:
        trainset, testset = dset.createSeparateInputData(N, fileloc, datasetname, blockTrain, seqTrain, labelContext)
    else:
        trainset, testset, _, _ = dset.loadInputData(fileloc, datasetname)

    # define and train a neural network model, log performance and output trained model
    if networkStyle == 'recurrent':
        model = mnet.trainRecurrentNetwork(args, device, multiparams, trainset, testset, N, params)
    else:
        model = mnet.trainMLPNetwork(args, device, multiparams, trainset, testset, N, params)

    # save the trained weights so we can easily look at them
    print(trained_modelname)
    torch.save(model, trained_modelname)

# ---------------------------------------------------------------------------- #

def analyseNetwork(fileloc, args, params):
    """Perform MDS on:
        - the hidden unit activations (60-dim) for each unique input in each context.
        - the averaged hidden unit activations (60-dim), averaged across the unique judgement values in each context.
        - the recurrent latent states (33-dim), as they evolve across the 12k sequential trials.
    """
    # load the MDS analysis if we already have it and move on
    datasetname, trained_modelname, analysis_name, _ = mnet.getDatasetName(args, *params)

    # load an existing dataset
    try:
        data = np.load(analysis_name+'.npy', allow_pickle=True)
        MDS_dict = data.item()
        preanalysed = True
        print('Loading existing network analysis...')
    except:
        preanalysed = False
        print('Analysing trained network...')

    if not preanalysed:
        # load the trained model and the datasets it was trained/tested on
        trained_model = torch.load(trained_modelname)
        trainset, testset, np_trainset, np_testset = dset.loadInputData(fileloc, datasetname)
        networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState = params

        # pass each input through the model and determine the hidden unit activations
        #if (networkStyle=='recurrent') and retainHiddenState: # pass the whole sequence of trials for the recurrent state
        train_loader = DataLoader(trainset, batch_size=1, shuffle=False)
        activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, time_index, counter, drift = mnet.getActivations(np_trainset, trained_model, networkStyle, retainHiddenState, train_loader)
        dimKeep = 'judgement'                      # representation of the currently presented number, averaging over previous number
        sl_activations, sl_contexts, sl_MDSlabels, sl_refValues, sl_judgeValues, sl_counter = MDSplt.averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, labelContext, counter)

        # How bout if we average activations over the difference values!
        diff_sl_activations, diff_sl_contexts, diff_sl_MDSlabels, diff_sl_refValues, diff_sl_judgeValues, diff_sl_counter, sl_diffValues = MDSplt.diff_averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, labelContext, counter)

        # do MDS on the activations for the training set
        tic = time.time()
        randseed = 3 # so that we get the same MDS each time
        embedding = MDS(n_components=3, random_state=randseed)
        MDS_activations = embedding.fit_transform(activations)
        sl_embedding = MDS(n_components=3, random_state=randseed)
        MDS_slactivations = sl_embedding.fit_transform(sl_activations)

        diff_sl_embedding = MDS(n_components=3, random_state=randseed)
        MDS_diff_slactivations = diff_sl_embedding.fit_transform(diff_sl_activations)


        # now do MDS again but for the latent state activations through time in the training set (***HRS takes ages to do this MDS)
        #print(drift["temporal_activation_drift"].shape)
        #embedding = MDS(n_components=3, random_state=randseed)
        #drift["MDS_latentstate"] = embedding.fit_transform(drift["temporal_activation_drift"])
        #print(drift["MDS_latentstate"].shape)
        toc = time.time()
        print('MDS fitting completed, took (s): ' + str(toc-tic))

        MDS_dict = {"MDS_activations":MDS_activations, "activations":activations, "MDSlabels":MDSlabels, \
                    "labels_refValues":labels_refValues, "labels_judgeValues":labels_judgeValues, "drift":drift,\
                    "labels_contexts":labels_contexts, "MDS_slactivations":MDS_slactivations, "sl_activations":sl_activations,\
                    "sl_contexts":sl_contexts, "sl_MDSlabels":sl_MDSlabels, "sl_refValues":sl_refValues, "sl_judgeValues":sl_judgeValues, "sl_counter":sl_counter,\
                    "MDS_diff_slactivations":MDS_diff_slactivations,"diff_sl_activations":diff_sl_activations, "diff_sl_contexts":diff_sl_contexts, "sl_diffValues":sl_diffValues}

        # save the analysis for next time
        np.save(analysis_name+'.npy', MDS_dict)
        print('Saving network analysis...')

    return MDS_dict

# ---------------------------------------------------------------------------- #

def generatePlots(MDS_dict, args, params):
    # This function just plots stuff and saves the generated figures
    saveFig = True
    plot_diff_code = False    # do we want to plot the difference code or the average A activations
    labelNumerosity = True    # numerosity vs outcome labels
    params.append(saveFig)

    # Label activations by mean number A numerosity
    MDSplt.activationRDMs(MDS_dict, args, params, plot_diff_code)  # activations RSA
    MDSplt.plot3MDSMean(MDS_dict, args, params, labelNumerosity, plot_diff_code) # mean MDS of our hidden activations (averaged across number B)
    MDSplt.plot3MDS(MDS_dict, args, params)      # the full MDS cloud, coloured by different labels

    # Label activations by the difference code numerosity
    #plot_diff_code = True
    #MDSplt.activationRDMs(MDS_dict, args, params, plot_diff_code)  # activations RSA
    #MDSplt.plot3MDSMean(MDS_dict, args, params, labelNumerosity, plot_diff_code)

    # Plot checks on the training data sequencing
    #n = plt.hist(activations)   # They are quite sparse activations (but we dont really care that much)
    #MDSplt.viewTrainingSequence(MDS_dict, args, params)  # Plot the context sequencing in the training set through time
    #MDSplt.instanceCounter(MDS_dict, args, params)  # Check how many samples we have of each unique input (should be context-ordered)

    # MDS with output labels (true/false labels)
    #labelNumerosity = False
    #MDSplt.plot3MDS(MDS_dict, args, params, labelNumerosity, plot_diff_code)
    #MDSplt.plot3MDSContexts(MDS_dict, labelNumerosity, args, params)  # plot the MDS with context labels. ***HRS obsolete?

    # 3D Animations
    #MDSplt.animate3DMDS(MDS_dict, args, params, plot_diff_code)  # plot a 3D version of the MDS constructions
    #MDSplt.animate3DdriftMDS(MDS_dict, args, params)             # plot a 3D version of the latent state MDS

# ---------------------------------------------------------------------------- #

if __name__ == '__main__':

    # dataset parameters
    createNewDataset = False          # re-generate the random train/test dataset each time?
    fileloc = 'datasets/'
    N = 15                            # total max numerosity for the greatest range we deal with
    blockTrain = True                 # whether to block the training by context
    seqTrain = True                   # whether there is sequential structure linking inputs A and B i.e. if at trial t+1 input B (ref) == input A from trial t
    labelContext = 'true'          # 'true', 'random', 'constant', does the input contain true markers of context (1-3) or random ones (still 1-3)?
    retainHiddenState = False          # initialise the hidden state for each pair as the hidden state of the previous pair
    if not blockTrain:
        seqTrain = False              # cant have sequential AB training structure if contexts are intermingled

    # which model / trained dataset we want to look at
    networkStyle = 'recurrent' #'recurrent'  # 'mlp'
    #noiselevels = np.linspace(0, 2.5, 25)
    noiselevels = [0.0]

    for noise_std in noiselevels:
        params = [networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState]

        # Train the network from scratch
        trainAndSaveANetwork(params, createNewDataset)

        # Analyse the trained network
        args, _, _ = mnet.defineHyperparams() # network training hyperparams
        MDS_dict = analyseNetwork(fileloc, args, params)

        #np.save("constantcontextlabel_activations.npy", MDS_dict["sl_activations"])
        generatePlots(MDS_dict, args, params)

# ---------------------------------------------------------------------------- #
