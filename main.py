"""
 This is a first pass simulation for training a simple MLP on a relational magnitude problem
 i.e. the network will be trained to answer the question: is input 2 > input 1?

 Author: Hannah Sheahan, sheahan.hannah@gmail.com
 Date: 04/12/2019
 Notes: N/A
 Issues: N/A
"""
# ---------------------------------------------------------------------------- #
import mag_network as mnet           # functions/classes for our network training
import define_dataset as dset
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import random
from sklearn.manifold import MDS
from sklearn.utils import shuffle
import copy

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from itertools import product  # makes testing and comparing different hyperparams in tensorboard easy
import argparse                # makes defining the hyperparams and tools for running our network easier from the command line

#--------------------------------------------------#

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

#--------------------------------------------------#

def autoSaveFigure(basetitle, blockedTraining, sequentialABTraining, labelNumerosity, saveFig):
    """This function will save the currently open figure with a base title and some details pertaining to how the activations were generated."""
    # automatic save file title details
    if blockedTraining:
        blockedtext = '_blocked'
    else:
        blockedtext = ''

    if sequentialABTraining:
        seqtext = '_sequential'
    else:
        seqtext = ''
    if labelNumerosity:
        cbar.ax.set_yticklabels(['1','15'])
        labeltext = '_numerosity'
    else:
        labeltext = '_contexts'

    if saveFig:
        plt.savefig(basetitle+blockedtext+seqtext+labeltext+'.pdf',bbox_inches='tight')

#--------------------------------------------------#

def plot3MDSMean(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, contextcolours, labelNumerosity, blockedTraining, sequentialABTraining, saveFig):
    """This function is just like plot3MDS and plot3MDSContexts but for the formatting of the data which has been averaged across one of the two numerosity values.
    """
    fig,ax = plt.subplots(1,3, figsize=(14,5))
    colours = get_cmap(10, 'magma')
    diffcolours = get_cmap(20, 'magma')
    for j in range(3):  # 3 MDS dimensions
        if j==0:
            dimA = 0
            dimB = 1
            ax[j].set_xlabel('dim 1')
            ax[j].set_ylabel('dim 2')
        elif j==1:
            dimA = 0
            dimB = 2
            ax[j].set_xlabel('dim 1')
            ax[j].set_ylabel('dim 3')
        elif j==2:
            dimA = 1
            dimB = 2
            ax[j].set_xlabel('dim 2')
            ax[j].set_ylabel('dim 3')

        ax[j].set_title('context')
        for i in range((MDS_activations.shape[0])):
            # colour by context
            ax[j].scatter(MDS_activations[i, dimA], MDS_activations[i, dimB], color=contextcolours[int(labels_contexts[i])-1])

        ax[j].axis('equal')
        ax[j].set(xlim=(-1, 1), ylim=(-1, 1))

#--------------------------------------------------#

def plot3MDSContexts(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, contextcolours, labelNumerosity, blockedTraining, sequentialABTraining, saveFig):
    """This is a just function to plot the MDS of activations and label the dots with the colour of the context."""

    fig,ax = plt.subplots(1,3, figsize=(14,5))
    colours = get_cmap(10, 'magma')
    diffcolours = get_cmap(20, 'magma')
    for j in range(3):  # 3 MDS dimensions

        if j==0:
            dimA = 0
            dimB = 1
            ax[j].set_xlabel('dim 1')
            ax[j].set_ylabel('dim 2')
        elif j==1:
            dimA = 0
            dimB = 2
            ax[j].set_xlabel('dim 1')
            ax[j].set_ylabel('dim 3')
        elif j==2:
            dimA = 1
            dimB = 2
            ax[j].set_xlabel('dim 2')
            ax[j].set_ylabel('dim 3')

        ax[j].set_title('context')
        for i in range((MDS_activations.shape[0])):
            # colour by context
            ax[j].scatter(MDS_activations[i, dimA], MDS_activations[i, dimB], color=contextcolours[int(labels_contexts[i])-1])

        ax[j].axis('equal')
        ax[j].set(xlim=(-3, 3), ylim=(-3, 3))

    autoSaveFigure('figures/3MDS_60hiddenactivations_contexts', blockedTraining, sequentialABTraining, labelNumerosity, saveFig)

#--------------------------------------------------#

def plot3MDS(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, contextcolours, labelNumerosity, blockedTraining, sequentialABTraining, saveFig):
    """This is a function to plot the MDS of activations and label according to numerosity and context"""

    # Plot the hidden activations for the 3 MDS dimensions
    fig,ax = plt.subplots(3,3, figsize=(14,15))
    colours = get_cmap(10, 'viridis')
    diffcolours = get_cmap(20, 'viridis')

    for k in range(3):
        for j in range(3):  # 3 MDS dimensions
            if j==0:
                dimA = 0
                dimB = 1
                ax[k,j].set_xlabel('dim 1')
                ax[k,j].set_ylabel('dim 2')
            elif j==1:
                dimA = 0
                dimB = 2
                ax[k,j].set_xlabel('dim 1')
                ax[k,j].set_ylabel('dim 3')
            elif j==2:
                dimA = 1
                dimB = 2
                ax[k,j].set_xlabel('dim 2')
                ax[k,j].set_ylabel('dim 3')

            for i in range((MDS_activations.shape[0])):
                if labelNumerosity:

                    # colour by numerosity
                    if k==0:
                        ax[k,j].scatter(MDS_activations[i, dimA], MDS_activations[i, dimB], color=diffcolours(int(10+labels_judgeValues[i]-labels_refValues[i])), edgecolors=contextcolours[int(labels_contexts[i])-1])
                    elif k==1:
                        ax[k,j].scatter(MDS_activations[i, dimA], MDS_activations[i, dimB], color=colours(int(labels_refValues[i])-1), edgecolors=contextcolours[int(labels_contexts[i])-1])
                    else:
                        im = ax[k,j].scatter(MDS_activations[i, dimA], MDS_activations[i, dimB], color=colours(int(labels_judgeValues[i])-1), edgecolors=contextcolours[int(labels_contexts[i])-1])
                        if j==2:
                            if i == (MDS_activations.shape[0])-1:
                                cbar = fig.colorbar(im, ticks=[0,1])
                else:
                    # colour by true/false label
                    if MDSlabels[i]==0:
                        colour = 'red'
                    else:
                        colour = 'green'
                    ax[k,j].scatter(MDS_activations[i, dimA], MDS_activations[i, dimB], color=colour)
                    #if k==0:
                    #    ax[k,j].text(MDS_activations[i, dimA], MDS_activations[i, dimB]+0.05, str(labels_judgeValues[i][0]-labels_refValues[i][0]), color=colour)
                    #elif k==1:
                    #    ax[k,j].text(MDS_activations[i, dimA], MDS_activations[i, dimB]+0.05, str(labels_refValues[i][0]), color=colour)
                    #else:
                    #    ax[k,j].text(MDS_activations[i, dimA], MDS_activations[i, dimB]+0.05, str(labels_judgeValues[i][0]), color=colour)

                # some titles
                if k==0:
                    ax[k,j].set_title('value difference')
                    ax[k,j].axis('equal')
                elif k==1:
                    ax[k,j].set_title('reference')
                else:
                    ax[k,j].set_title('judgement')
                ax[k,j].set(xlim=(-3, 3), ylim=(-3, 3))  # set axes equal and the same for comparison

    autoSaveFigure('figures/3MDS_60hiddenactivations', blockedTraining, sequentialABTraining, labelNumerosity, saveFig)

#--------------------------------------------------#

def getActivations(trainset,trained_model):
    """ This will determine the hidden unit activations for each *unique* input in the training set
     there are many repeats of inputs in the training set so just doing it over the unique ones will help speed up our MDS by loads."""

    # determine the unique inputs for the training set (there are repeats)
    unique_inputs, uniqueind = np.unique(trainset["input"], axis=0, return_index=True)
    unique_labels = trainset["label"][uniqueind]
    unique_context = trainset["context"][uniqueind]
    unique_refValue = trainset["refValue"][uniqueind]
    unique_judgementValue = trainset["judgementValue"][uniqueind]

    # preallocate some space...
    labels_refValues = np.empty((len(uniqueind),1))
    labels_judgeValues = np.empty((len(uniqueind),1))
    contexts = np.empty((len(uniqueind),1))
    MDSlabels = np.empty((len(uniqueind),1))
    hdim = len(list(trained_model.fc1.parameters())[0])
    activations = np.empty(( len(uniqueind), hdim ))

    #  pass each input through the netwrk and see what happens to the hidden layer activations
    for sample in range(len(uniqueind)):
        sample_input = unique_inputs[sample]
        sample_label = unique_labels[sample]
        labels_refValues[sample] = dset.turnOneHotToInteger(unique_refValue[sample])
        labels_judgeValues[sample] = dset.turnOneHotToInteger(unique_judgementValue[sample])
        MDSlabels[sample] = sample_label
        contexts[sample] = dset.turnOneHotToInteger(unique_context[sample])
        # get the activations for that input
        h1activations,_,_ = trained_model.get_activations(batchToTorch(torch.from_numpy(sample_input)))
        activations[sample] = h1activations.detach()

    return activations, MDSlabels, labels_refValues, labels_judgeValues, contexts

#--------------------------------------------------#

def averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels):
    """This function will average the hidden unit activations over one of the two numbers involved in the representation:
    either the reference or the judgement number. This is so that we can then compare to Fabrice's plots
     which are averaged over the previously presented number (input B).
     - dimKeep = 'reference' or 'judgement'
    """
    # prior to performing the MDS we want to know whether to flatten over a particular value i.e. if plotting for reference value, flatten over the judgement value and vice versa
    uniqueValues = [int(np.unique(labels_judgeValues)[i]) for i in range(len(np.unique(labels_judgeValues)))]
    flat_activations = np.zeros((3,len(uniqueValues),activations.shape[1]))
    flat_values = np.zeros((3,len(uniqueValues),1))
    flat_outcomes = np.empty((3,len(uniqueValues),1))
    flat_contexts = np.empty((3,len(uniqueValues),1))
    divisor = np.zeros((3,len(uniqueValues)))

    # which label to flatten over (we keep whichever dimension is dimKeep, and average over the other)
    if dimKeep == 'reference':
        flattenValues = labels_refValues
    else:
        flattenValues = labels_judgeValues

    # pick out all the activations that meet this condition for each context and then average over them
    for context in range(3):
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

#--------------------------------------------------#

def main():

    # Define the training hyperparameters for our network (passed as args when calling main.py from command line)
    args, device, multiparams = mnet.defineHyperparams()

    # a dataset for us to work with
    createNewDataset = True
    fileloc = 'datasets/'

    blockedTraining = True            # whether to block the training by context
    sequentialABTraining = True        # whether there is sequential structure linking inputs A and B i.e. if at trial t+1 input B (ref) == input A from trial t
    if not blockedTraining:
        sequentialABTraining = False   # we cant have sequential AB training structure if contexts are intermingled

    datasetname, trained_modelname = mnet.setDatasetName(blockedTraining, sequentialABTraining)

    if createNewDataset:
        N = 15                         # total max numerosity for the greatest range we deal with
        trainset, testset = dset.createSeparateInputData(N, fileloc, datasetname, blockedTraining, sequentialABTraining)
    else:
        trainset, testset, _, _ = dset.loadInputData(fileloc, datasetname)

    # define and train a neural network model, log performance and output trained model
    model = mnet.trainNetwork(args, device, multiparams, trainset, testset, N)

    # save the trained weights so we can easily look at them
    torch.save(model, trained_modelname)

#--------------------------------------------------#
"""
# Some interactive mode plotting code...

# Now lets take a look at our weights and the responses to the inputs in the training set we trained on
blockedTraining = True
sequentialABTraining = True
datasetname, trained_model = mnet.getDatasetName(blockedTraining, sequentialABTraining)

# lets load the dataset we used for training the model
fileloc = 'datasets/'
trainset, testset, np_trainset, np_testset = dset.loadInputData(fileloc, datasetname)

# pass each input through the model and determine the hidden unit activations ***HRS remember that this looks for the unique inputs in 'input' so when context stops being an actual input it will lose this unless careful
activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts = getActivations(np_trainset,trained_model)

dimKeep = 'judgement'  # this is what Fabrice's plots show (representation of the currently presented number)
sl_activations, sl_contexts, sl_MDSlabels, sl_refValues, sl_judgeValues = averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels)

# do MDS on the activations for the training set
embedding = MDS(n_components=3)
MDS_activations = embedding.fit_transform(activations)

# plot the MDS of our hidden activations
saveFig = True
labelNumerosity = True
contextcolours = ['gold','dodgerblue', 'orangered']  #1-15, 1-10, 5-15 like fabrices colours

# plot the MDS with number labels but flatten across the other factor
# ***HRS the following plot looks like flat out wrong because only in +ve dimensions. Is the plotting wrong or the MDS averaging wrong?? I think it must be the MDS averaging.


plot3MDSMean(sl_activations, sl_MDSlabels, sl_refValues, sl_judgeValues, sl_contexts, contextcolours, labelNumerosity, blockedTraining, sequentialABTraining, saveFig)

# plot the MDS with number labels
plot3MDS(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, contextcolours, labelNumerosity, blockedTraining, sequentialABTraining, saveFig)



# plot the MDS with output labels (true/false labels)
labelNumerosity = False
plot3MDS(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, contextcolours, labelNumerosity, blockedTraining, sequentialABTraining, saveFig)

# plot the MDS with context labels
plot3MDSContexts(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, contextcolours, labelNumerosity, blockedTraining, sequentialABTraining, saveFig)

"""

#--------------------------------------------------#

# to run from the command line
if __name__ == '__main__':
    main()
