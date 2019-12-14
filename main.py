"""
 This is a first pass simulation for training a simple MLP on a relational magnitude problem
 i.e. the network will be trained to answer the question: is input 2 > input 1?

 Author: Hannah Sheahan, sheahan.hannah@gmail.com
 Date: 04/12/2019
 Notes: N/A
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

def main():

    # Define the training hyperparameters for our network (passed as args when calling main.py from command line)
    args, device, multiparams = mnet.defineHyperparams()

    # a dataset for us to work with
    createNewDataset = True
    fileloc = 'datasets/'

    blockTrain = True            # whether to block the training by context
    seqTrain = True        # whether there is sequential structure linking inputs A and B i.e. if at trial t+1 input B (ref) == input A from trial t
    if not blockTrain:
        seqTrain = False   # we cant have sequential AB training structure if contexts are intermingled

    datasetname, trained_modelname = mnet.setDatasetName(blockTrain, seqTrain)

    if createNewDataset:
        N = 15                         # total max numerosity for the greatest range we deal with
        trainset, testset = dset.createSeparateInputData(N, fileloc, datasetname, blockTrain, seqTrain)
    else:
        trainset, testset, _, _ = dset.loadInputData(fileloc, datasetname)

    # define and train a neural network model, log performance and output trained model
    model = mnet.trainNetwork(args, device, multiparams, trainset, testset, N)

    # save the trained weights so we can easily look at them
    torch.save(model, trained_modelname)

# ---------------------------------------------------------------------------- #

# Some interactive mode plotting code...
reload(MDSplt)
#reload(mnet)

# load the trained model and the datasets it was trained/tested on
blockTrain = True
seqTrain = True
datasetname, trained_model = mnet.getDatasetName(blockTrain, seqTrain)
fileloc = 'datasets/'
trainset, testset, np_trainset, np_testset = dset.loadInputData(fileloc, datasetname)

# pass each input through the model and determine the hidden unit activations
# ***HRS remember that this looks for the unique inputs in 'input' so when context stops being an actual input it will lose this unless careful
activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts = mnet.getActivations(np_trainset,trained_model)
dimKeep = 'judgement'                      # representation of the currently presented number, averaging over previous number
sl_activations, sl_contexts, sl_MDSlabels, sl_refValues, sl_judgeValues = MDSplt.averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels)

# do MDS on the activations for the training set
embedding = MDS(n_components=3)
MDS_activations = embedding.fit_transform(activations)
sl_embedding = MDS(n_components=3)
MDS_slactivations = sl_embedding.fit_transform(sl_activations)

# plot the MDS of our hidden activations
saveFig = True
labelNumerosity = True

# they are both quite sparse activations
n = plt.hist(activations)

# Take a look at the activations RSA
MDSplt.activationRDMs(activations, sl_activations)

# plot the MDS with number labels but flatten across the other factor
reload(MDSplt)
MDSplt.plot3MDSMean(MDS_slactivations, sl_MDSlabels, sl_refValues, sl_judgeValues, sl_contexts, labelNumerosity, blockTrain, seqTrain, saveFig)

# plot the MDS with number labels
MDSplt.plot3MDS(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, labelNumerosity, blockTrain, seqTrain, saveFig)

# plot the MDS with output labels (true/false labels)
labelNumerosity = False
MDSplt.plot3MDS(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, labelNumerosity, blockTrain, seqTrain, saveFig)

# plot the MDS with context labels
MDSplt.plot3MDSContexts(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, labelNumerosity, blockTrain, seqTrain, saveFig)

"""

# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    main()
