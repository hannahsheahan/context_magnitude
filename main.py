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

def trainAndSaveANetwork():

    # Define the training hyperparameters for our network (passed as args when calling main.py from command line)
    args, device, multiparams = mnet.defineHyperparams()
    N = 15                         # total max numerosity for the greatest range we deal with

    # a dataset for us to work with
    createNewDataset = False
    fileloc = 'datasets/'

    networkStyle = 'recurrent' #'recurrent'  # 'mlp'
    noise_std = 1.2
    blockTrain = True            # whether to block the training by context
    seqTrain = True        # whether there is sequential structure linking inputs A and B i.e. if at trial t+1 input B (ref) == input A from trial t
    labelContext = True
    if not blockTrain:
        seqTrain = False   # we cant have sequential AB training structure if contexts are intermingled

    params = [networkStyle, noise_std, blockTrain, seqTrain, labelContext]

    datasetname, trained_modelname = mnet.getDatasetName(*params)

    if createNewDataset:
        trainset, testset = dset.createSeparateInputData(N, fileloc, datasetname, blockTrain, seqTrain, labelContext)
    else:
        trainset, testset, _, _ = dset.loadInputData(fileloc, datasetname)

    # define and train a neural network model, log performance and output trained model
    if networkStyle == 'recurrent':
        args.epochs = args.epochs * 2  # the recurrent network needs more training time
        model = mnet.trainRecurrentNetwork(args, device, multiparams, trainset, testset, N, noise_std)
    else:
        model = mnet.trainMLPNetwork(args, device, multiparams, trainset, testset, N)

    # save the trained weights so we can easily look at them
    print(trained_modelname)
    torch.save(model, trained_modelname)

# ---------------------------------------------------------------------------- #

# Some interactive mode plotting code...
reload(mnet)
reload(MDSplt)

# which model / trained dataset we want to look at
networkStyle = 'recurrent' #'recurrent'  #'mlp'
noise_std = 1.2
blockTrain = True
seqTrain = True
labelContext = True
params = [networkStyle, noise_std, blockTrain, seqTrain, labelContext]

# load the trained model and the datasets it was trained/tested on
datasetname, trained_modelname = mnet.getDatasetName(*params)
trained_model = torch.load(trained_modelname)
fileloc = 'datasets/'
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

# ---------------------------------------------------------------------------- #
# Plot stuff
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
"""

if __name__ == '__main__':
    trainAndSaveANetwork()
"""
