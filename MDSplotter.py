"""
This is a selection of functions for plotting MDS projections of relative magnitude-trained networks.

Author: Hannah Sheahan, sheahan.hannah@gmail.com
Date: 13/12/2019
Notes: N/A
Issues: N/A
"""
# ---------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import random

from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.utils import shuffle

# generic plotting settings
contextcolours = ['gold', 'dodgerblue', 'orangered']   # 1-15, 1-10, 5-15 like fabrices colours

# ---------------------------------------------------------------------------- #

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

# ---------------------------------------------------------------------------- #

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
        labeltext = '_numerosity'
    else:
        labeltext = '_contexts'

    if saveFig:
        plt.savefig(basetitle+blockedtext+seqtext+labeltext+'.pdf',bbox_inches='tight')

# ---------------------------------------------------------------------------- #

def activationRDMs(activations, sl_activations):
    """Plot the representational disimilarity structure of the hidden unit activations, sorted by context, and within that magnitude.
    Context order:  1-15, 1-10, 5-15
    """
    fig, ax = plt.subplots(1,2)
    D = pairwise_distances(activations)  # note that activations are structured by: context (1-15,1-10,5-15) and judgement value magnitude within that.
    im = ax[0].imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title('All activations')

    # this looks like absolute magnitude to me (note the position of the light diagonals on the between context magnitudes - they are not centred)
    D = pairwise_distances(sl_activations)
    im = ax[1].imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title('Averaged activations')

# ---------------------------------------------------------------------------- #

def plot3MDS(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, labelNumerosity, blockedTraining, sequentialABTraining, saveFig):
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
                                if labelNumerosity:
                                    cbar.ax.set_yticklabels(['1','15'])

                else:
                    # colour by true/false label
                    if MDSlabels[i]==0:
                        colour = 'red'
                    else:
                        colour = 'green'
                    ax[k,j].scatter(MDS_activations[i, dimA], MDS_activations[i, dimB], color=colour)

                # some titles
                if k==0:
                    ax[k,j].set_title('value difference')
                    ax[k,j].axis('equal')
                elif k==1:
                    ax[k,j].set_title('reference')
                else:
                    ax[k,j].set_title('judgement')
                ax[k,j].set(xlim=(-3, 3), ylim=(-3, 3))  # set axes equal and the same for comparison

    autoSaveFigure('figures/3MDS60_', blockedTraining, sequentialABTraining, labelNumerosity, saveFig)

# ---------------------------------------------------------------------------- #

def plot3MDSContexts(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, labelNumerosity, blockedTraining, sequentialABTraining, saveFig):
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

    autoSaveFigure('figures/3MDS60_contexts_', blockedTraining, sequentialABTraining, labelNumerosity, saveFig)

# ---------------------------------------------------------------------------- #

def plot3MDSMean(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, labelNumerosity, blockedTraining, sequentialABTraining, saveFig):
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
        ax[j].set(xlim=(-3, 3), ylim=(-3, 3))

    autoSaveFigure('figures/3MDS60_meanJudgement_', blockedTraining, sequentialABTraining, labelNumerosity, saveFig)

# ---------------------------------------------------------------------------- #
