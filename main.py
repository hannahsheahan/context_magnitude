"""
 This is a first pass simulation for training a simple MLP on a relational magnitude problem
 i.e. the network will be trained to answer the question: is input 2 > input 1?

 Author: Hannah Sheahan, sheahan.hannah@gmail.com
 Date: 04/12/2019
 Notes:
 - requires ffmpeg for 3D animation generation
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
    args, device, multiparams = mnet.defineHyperparams() # training hyperparames for network (passed as args when called from command line)
    datasetname, trained_modelname = mnet.getDatasetName(*params)
    if createNewDataset:
        trainset, testset = dset.createSeparateInputData(N, fileloc, datasetname, blockTrain, seqTrain, labelContext)
    else:
        trainset, testset, _, _ = dset.loadInputData(fileloc, datasetname)

    # define and train a neural network model, log performance and output trained model
    if networkStyle == 'recurrent':
        model = mnet.trainRecurrentNetwork(args, device, multiparams, trainset, testset, N, noise_std)
    else:
        model = mnet.trainMLPNetwork(args, device, multiparams, trainset, testset, N)

    # save the trained weights so we can easily look at them
    print(trained_modelname)
    torch.save(model, trained_modelname)

# ---------------------------------------------------------------------------- #

def analyseNetwork(fileloc, params):
    # load the trained model and the datasets it was trained/tested on
    datasetname, trained_modelname = mnet.getDatasetName(*params)
    trained_model = torch.load(trained_modelname)
    trainset, testset, np_trainset, np_testset = dset.loadInputData(fileloc, datasetname)

    # pass each input through the model and determine the hidden unit activations
    activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts = mnet.getActivations(np_trainset,trained_model, networkStyle)
    dimKeep = 'judgement'                      # representation of the currently presented number, averaging over previous number
    sl_activations, sl_contexts, sl_MDSlabels, sl_refValues, sl_judgeValues = MDSplt.averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, labelContext)

    # do MDS on the activations for the training set
    randseed = 3 # so that we get the same MDS each time
    embedding = MDS(n_components=3, random_state=randseed)
    MDS_activations = embedding.fit_transform(activations)
    sl_embedding = MDS(n_components=3, random_state=randseed)
    MDS_slactivations = sl_embedding.fit_transform(sl_activations)

    MDS_dict = {"MDS_activations":MDS_activations, "activations":activations, "MDSlabels":MDSlabels,\
                "labels_refValues":labels_refValues, "labels_judgeValues":labels_judgeValues,\
                "labels_contexts":labels_contexts, "MDS_slactivations":MDS_slactivations, "sl_activations":sl_activations,\
                "sl_contexts":sl_contexts, "sl_MDSlabels":sl_MDSlabels, "sl_refValues":sl_refValues, "sl_judgeValues":sl_judgeValues}

    return MDS_dict

# ---------------------------------------------------------------------------- #

def generatePlots(MDS_dict, params):
    # This function just plots stuff and saves the generated figures
    saveFig = True
    params.append(saveFig)

    # they are quite sparse activations? (but we dont really care that much)
    #n = plt.hist(activations)

    # Take a look at the activations RSA
    MDSplt.activationRDMs(MDS_dict, params)

    # # plot the MDS of our hidden activations, with number labels but flatten across the other factor
    labelNumerosity = True
    MDSplt.plot3MDSMean(MDS_dict, labelNumerosity, params)

    # plot the MDS with number labels
    labelNumerosity = True
    MDSplt.plot3MDS(MDS_dict, labelNumerosity, params)

    # plot the MDS with output labels (true/false labels)
    labelNumerosity = False
    MDSplt.plot3MDS(MDS_dict, labelNumerosity, params)

    # plot the MDS with context labels
    MDSplt.plot3MDSContexts(MDS_dict, labelNumerosity, params)

    # plot a 3D version of the MDS constructions
    MDSplt.animate3DMDS(MDS_dict, params)

# ---------------------------------------------------------------------------- #

if __name__ == '__main__':

    # dataset parameters
    createNewDataset = True
    fileloc = 'datasets/'
    N = 15                            # total max numerosity for the greatest range we deal with
    blockTrain = True                 # whether to block the training by context
    seqTrain = True                   # whether there is sequential structure linking inputs A and B i.e. if at trial t+1 input B (ref) == input A from trial t
    labelContext = False
    if not blockTrain:
        seqTrain = False              # cant have sequential AB training structure if contexts are intermingled

    # which model / trained dataset we want to look at
    networkStyle = 'recurrent' #'recurrent'  # 'mlp'
    #noiselevels = np.linspace(0, 2.5, 25)
    noiselevels = [0.0]
    num_repeats = range(10)

    for repeat in num_repeats:
        for noise_std in noiselevels:
            params = [networkStyle, noise_std, blockTrain, seqTrain, labelContext]

            # Train the network from scratch
            trainAndSaveANetwork(params, createNewDataset)

            # Analyse the trained network
            #MDS_dict = analyseNetwork(fileloc, params)
            #generatePlots(MDS_dict, params)

# ---------------------------------------------------------------------------- #
