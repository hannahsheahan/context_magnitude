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

def averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels):
    """This function will average the hidden unit activations over one of the two numbers involved in the representation:
    either the reference or the judgement number. This is so that we can then compare to Fabrice's plots
     which are averaged over the previously presented number (input B).
     - dimKeep = 'reference' or 'judgement'
    """
    # prior to performing the MDS we want to know whether to flatten over a particular value i.e. if plotting for reference value, flatten over the judgement value and vice versa
    uniqueValues = [int(np.unique(labels_judgeValues)[i]) for i in range(len(np.unique(labels_judgeValues)))]
    Ncontexts = 3
    flat_activations = np.zeros((Ncontexts,len(uniqueValues),activations.shape[1]))
    flat_values = np.zeros((Ncontexts,len(uniqueValues),1))
    flat_outcomes = np.empty((Ncontexts,len(uniqueValues),1))
    flat_contexts = np.empty((Ncontexts,len(uniqueValues),1))
    divisor = np.zeros((Ncontexts,len(uniqueValues)))

    # which label to flatten over (we keep whichever dimension is dimKeep, and average over the other)
    if dimKeep == 'reference':
        flattenValues = labels_refValues
    else:
        flattenValues = labels_judgeValues

    # pick out all the activations that meet this condition for each context and then average over them
    for context in range(Ncontexts):
        for value in uniqueValues:
            for i in range(labels_judgeValues.shape[0]):
                if labels_contexts[i] == context+1:  # remember to preserve the context structure
                    if flattenValues[i] == value:
                        flat_activations[context, value-1,:] += activations[i]
                        flat_contexts[context,value-1] = context
                        flat_values[context,value-1] = value
                        flat_outcomes[context,value-1] = MDSlabels[i]
                        divisor[context,value-1] +=1

            # take the mean i.e. normalise by the number of instances that met that condition
            if int(divisor[context,value-1]) == 0:
                flat_activations[context, value-1] = np.full_like(flat_activations[context, value-1], np.nan)
            else:
                flat_activations[context, value-1] = np.divide(flat_activations[context, value-1, :], divisor[context,value-1])

    # now cast out all the null instances e.g 1-5, 10-15 in certain contexts
    flat_activations = dset.flattenFirstDim(flat_activations)
    flat_contexts = dset.flattenFirstDim(flat_contexts)
    flat_values = dset.flattenFirstDim(flat_values)
    flat_outcomes = dset.flattenFirstDim(flat_outcomes)
    singlelabel_activations = []
    singlelabel_refValues = []
    singlelabel_judgeValues = []
    singlelabel_contexts = []
    singlelabel_MDSlabels = []
    for i in range(flat_activations.shape[0]):
        checknan = np.asarray([ np.isnan(flat_activations[i][j]) for j in range(len(flat_activations[i]))])
        if (checknan).all():
            pass
        else:
            singlelabel_activations.append(flat_activations[i])
            singlelabel_contexts.append(flat_contexts[i])
            singlelabel_MDSlabels.append(flat_outcomes[i])

            if dimKeep == 'reference':
                singlelabel_refValues.append(flat_values[i])
                singlelabel_judgeValues.append(0)
            else:
                singlelabel_refValues.append(0)
                singlelabel_judgeValues.append(flat_values[i])

    # finally, reshape the outputs so that they match our inputs nicely
    singlelabel_activations = np.asarray(singlelabel_activations)
    singlelabel_refValues = np.asarray(singlelabel_refValues)
    singlelabel_judgeValues = np.asarray(singlelabel_judgeValues)
    singlelabel_contexts = np.asarray(singlelabel_contexts)
    singlelabel_MDSlabels = np.asarray(singlelabel_MDSlabels)

    if dimKeep == 'reference':
        singlelabel_judgeValues = np.expand_dims(singlelabel_judgeValues, axis=1)
    else:
        singlelabel_refValues = np.expand_dims(singlelabel_refValues, axis=1)

    return singlelabel_activations, singlelabel_contexts, singlelabel_MDSlabels, singlelabel_refValues, singlelabel_judgeValues

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
reload(mnet)

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
sl_activations, sl_contexts, sl_MDSlabels, sl_refValues, sl_judgeValues = averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels)

# do MDS on the activations for the training set
embedding = MDS(n_components=3)
MDS_activations = embedding.fit_transform(activations)
sl_embedding = MDS(n_components=3)
MDS_slactivations = sl_embedding.fit_transform(sl_activations)

# plot the MDS of our hidden activations
saveFig = True
labelNumerosity = True

# they are both quite sparse activations
plt.hist(activations)

# Take a look at the activations RSA
MDSplt.activationRDMs(activations, sl_activations)

# plot the MDS with number labels but flatten across the other factor
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
