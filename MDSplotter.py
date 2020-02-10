"""
This is a selection of functions for plotting MDS projections of relative magnitude-trained networks.

Author: Hannah Sheahan, sheahan.hannah@gmail.com
Date: 14/12/2019
Notes: N/A
Issues: N/A
"""
# ---------------------------------------------------------------------------- #
import define_dataset as dset
from mpl_toolkits import mplot3d
import numpy as np
import copy
import sys
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation

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

def autoSaveFigure(basetitle, networkStyle, blockTrain, seqTrain, labelNumerosity, givenContext, labelContexts, noise_std, retainHiddenState, saveFig):
    """This function will save the currently open figure with a base title and some details pertaining to how the activations were generated."""
    # automatic save file title details
    retainstatetext = '_retainstate' if retainHiddenState else ''
    blockedtext = '_blocked' if blockTrain else ''
    seqtext = '_sequential' if seqTrain else ''
    labeltext = '_numerosity' if labelNumerosity else '_outcomes'
    contexts = '_contexts' if labelContexts else ''
    if givenContext=='true':
        contextlabelledtext = '_truecontextlabelled'
    elif givenContext=='random':
        contextlabelledtext =  '_randomcontextlabel'
    elif givenContext=='constant':
        contextlabelledtext =  '_constantcontextlabel'

    if saveFig:
        plt.savefig(basetitle+networkStyle+blockedtext+seqtext+contextlabelledtext+labeltext+contexts+retainstatetext+str(noise_std)+'.pdf',bbox_inches='tight')

    return basetitle+networkStyle+blockedtext+seqtext+contextlabelledtext+labeltext+contexts+retainstatetext+str(noise_std)

# ---------------------------------------------------------------------------- #

def activationRDMs(MDS_dict, params):
    """Plot the representational disimilarity structure of the hidden unit activations, sorted by context, and within that magnitude.
    Context order:  1-15, 1-10, 5-15
    """
    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, saveFig = params
    fig, ax = plt.subplots(1,2, figsize=(10,3))
    D = pairwise_distances(MDS_dict["activations"])  # note that activations are structured by: context (1-15,1-10,5-15) and judgement value magnitude within that.
    im = ax[0].imshow(D, zorder=2, cmap='Blues', interpolation='nearest', vmin=0, vmax=6)

    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('disimilarity')
    ax[0].set_title('All activations')
    ax[0].set_xticks([0,210,300])
    ax[0].set_xticklabels(['1-15', '1-10', '5-15'])
    ax[0].set_yticks([0,210,300])
    ax[0].set_yticklabels(['1-15', '1-10', '5-15'])

    # this looks like absolute magnitude to me (note the position of the light diagonals on the between context magnitudes - they are not centred)
    D = pairwise_distances(MDS_dict["sl_activations"])
    im = ax[1].imshow(D, zorder=2, cmap='Blues', interpolation='nearest', vmin=0, vmax=6)

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('disimilarity')
    ax[1].set_title('Averaged activations')
    ax[1].set_xticks([0,15,25])
    ax[1].set_xticklabels(['1-15', '1-10', '5-15'])
    ax[1].set_yticks([0,15,25])
    ax[1].set_yticklabels(['1-15', '1-10', '5-15'])

    n = autoSaveFigure('figures/RDM_', networkStyle, blockTrain, seqTrain, False, labelContext, False, noise_std,  retainHiddenState, saveFig)

# ---------------------------------------------------------------------------- #

def diff_activationRDMs(MDS_dict, params):
    """Plot the representational disimilarity structure of the hidden unit activations, sorted by context, and within that magnitude.
    Context order:  1-15, 1-10, 6-15
    """
    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, saveFig = params
    fig, ax = plt.subplots(1,2, figsize=(10,3))

    # this looks like absolute magnitude to me (note the position of the light diagonals on the between context magnitudes - they are not centred)
    D = pairwise_distances(MDS_dict["diff_sl_activations"])
    im = ax[1].imshow(D, zorder=2, cmap='Blues', interpolation='nearest', vmin=0, vmax=6)

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('disimilarity')
    ax[1].set_title('Averaged activations for a difference code')
    ax[1].set_xticks([0,27,45])
    ax[1].set_xticklabels(['-14:+14', '-9:+9', '-9:+9'])
    ax[1].set_yticks([0,27,45])
    ax[1].set_yticklabels(['-14:+14', '-9:+9', '-9:+9'])

    n = autoSaveFigure('figures/RDM_differencecode_', networkStyle, blockTrain, seqTrain, False, labelContext, False, noise_std,  retainHiddenState, saveFig)

# ---------------------------------------------------------------------------- #

def plot3MDS(MDS_dict, labelNumerosity, params):
    """This is a function to plot the MDS of activations and label according to numerosity and context"""

    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, saveFig = params

    # Plot the hidden activations for the 3 MDS dimensions
    colours = plt.cm.get_cmap('viridis')
    diffcolours = plt.cm.get_cmap('viridis')
    outcomecolours = ['red', 'green']

    norm = mplcol.Normalize(vmin=1, vmax=15)
    dnorm = mplcol.Normalize(vmin=-14, vmax=14)

    if not labelContext:
        labels_contexts = np.full_like(MDS_dict["labels_contexts"], 1)
    else:
        labels_contexts = MDS_dict["labels_contexts"]
    MDS_act = MDS_dict["MDS_activations"]

    for k in range(5):
        fig,ax = plt.subplots(1,3, figsize=(10,3.3))
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

            for i in range((MDS_act.shape[0])):

                # colour by numerosity
                if k==0:   # difference labels

                    #ax[k,j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=diffcolours(int(10+MDS_dict["labels_judgeValues"][i]-MDS_dict["labels_refValues"][i])), edgecolors=contextcolours[int(MDS_dict["labels_contexts"][i])-1])
                    im = ax[j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=diffcolours(dnorm(int(MDS_dict["labels_judgeValues"][i]-MDS_dict["labels_refValues"][i]))), s=20)
                    #ax[j].text(MDS_act[i, dimA], MDS_act[i, dimB], str(int(MDS_dict["labels_judgeValues"][i]-MDS_dict["labels_refValues"][i])), color='black', size=4, horizontalalignment='center', verticalalignment='center')
                    if j==2:
                        if i == (MDS_act.shape[0])-1:
                            cbar = fig.colorbar(im, ticks=[0,1])
                            if labelNumerosity:
                                cbar.ax.set_yticklabels(['-14','14'])
                elif k==1:  # B values
                    #ax[k,j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=colours(int(MDS_dict["labels_refValues"][i])-1), edgecolors=contextcolours[int(MDS_dict["labels_contexts"][i])-1])
                    im = ax[j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=colours(norm(int(MDS_dict["labels_refValues"][i])-1)), s=20)
                    if j==2:
                        if i == (MDS_act.shape[0])-1:
                            cbar = fig.colorbar(im, ticks=[0,1])
                            if labelNumerosity:
                                cbar.ax.set_yticklabels(['1','15'])
                elif k==2:  # A values
                    #im = ax[k,j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=colours(int(MDS_dict["labels_judgeValues"][i])-1), edgecolors=contextcolours[int(MDS_dict["labels_contexts"][i])-1])
                    im = ax[j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=colours(norm(int(MDS_dict["labels_judgeValues"][i])-1)), s=20)
                    if j==2:
                        if i == (MDS_act.shape[0])-1:
                            cbar = fig.colorbar(im, ticks=[0,1])
                            if labelNumerosity:
                                cbar.ax.set_yticklabels(['1','15'])
                elif k==3:  # context labels
                    im = ax[j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=contextcolours[int(MDS_dict["labels_contexts"][i])-1], s=20)

                elif k==4:
                    im = ax[j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=outcomecolours[int(MDS_dict["MDSlabels"][i])], s=20)


                # some titles
                if k==0:
                    ax[j].set_title('A - B labels')
                    #ax[j].axis('equal')
                    tx = 'AminusBlabel_'
                elif k==1:
                    ax[j].set_title('B labels')
                    tx = 'Blabel_'
                elif k==2:
                    ax[j].set_title('A labels')
                    tx = 'Alabel_'
                elif k==3:
                    ax[j].set_title('context labels')
                    tx = 'contlabel_'
                elif k==4:
                    ax[j].set_title('outcome labels')
                    tx = 'outcomelabel_'

                ax[j].set(xlim=(-4, 4), ylim=(-4, 4))  # set axes equal and the same for comparison

        n = autoSaveFigure('figures/3MDS60_'+tx, networkStyle, blockTrain, seqTrain, labelNumerosity, labelContext, False, noise_std, retainHiddenState, saveFig)

# ---------------------------------------------------------------------------- #

def plot3MDSContexts(MDS_dict, labelNumerosity, params):
    """This is a just function to plot the MDS of activations and label the dots with the colour of the context."""

    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, saveFig = params
    labels_contexts = MDS_dict["labels_contexts"] if labelContext else  np.full_like(MDS_dict["labels_contexts"], 1)

    MDS_act = MDS_dict["MDS_activations"]
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
        for i in range((MDS_act.shape[0])):
            # colour by context
            ax[j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=contextcolours[int(labels_contexts[i])-1])

        ax[j].axis('equal')
        ax[j].set(xlim=(-3, 3), ylim=(-3, 3))

    n = autoSaveFigure('figures/3MDS60_', networkStyle, blockTrain, seqTrain, labelNumerosity, labelContext, True, noise_std,  retainHiddenState,saveFig)

# ---------------------------------------------------------------------------- #

def plot3MDSMean(MDS_dict, labelNumerosity, params, plot_diff_code=False):
    """This function is just like plot3MDS and plot3MDSContexts but for the formatting of the data which has been averaged across one of the two numerosity values.
    Because there are fewer datapoints I also label the numerosity inside each context, like Fabrice does.
     - use the flag 'plot_diff_code' to plot the difference signal (A-B) rather than the A activations
    """
    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, saveFig = params
    fig,ax = plt.subplots(1,3, figsize=(18,5))
    colours = get_cmap(10, 'magma')
    diffcolours = get_cmap(20, 'magma')

    if plot_diff_code:
        MDS_act = MDS_dict["MDS_diff_slactivations"]
        contextlabel = MDS_dict["diff_sl_contexts"]
        numberlabel = MDS_dict["sl_diffValues"]
        differenceCodeText = 'differencecode_'
    else:
        MDS_act = MDS_dict["MDS_slactivations"]
        contextlabel = MDS_dict["sl_contexts"]
        numberlabel = MDS_dict["sl_judgeValues"]
        differenceCodeText = ''

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

        if plot_diff_code:
            contextA = range(27)
            contextB = range(27,45)
            contextC = range(45, 63)
        else:
            contextA = range(15)
            contextB = range(15,25)
            contextC = range(25, 35)
            # only plot lines between the MDS dots when plotting the average A activations, not A-B difference code (A-B structured differently)
            ax[j].plot(MDS_act[contextA, dimA], MDS_act[contextA, dimB], color=contextcolours[0])
            ax[j].plot(MDS_act[contextB, dimA], MDS_act[contextB, dimB], color=contextcolours[1])
            ax[j].plot(MDS_act[contextC, dimA], MDS_act[contextC, dimB], color=contextcolours[2])

        for i in range((MDS_act.shape[0])):
            # colour by context
            ax[j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=contextcolours[int(contextlabel[i])], s=80)

            # label numerosity in white inside the marker
            ax[j].text(MDS_act[i, dimA], MDS_act[i, dimB], str(int(numberlabel[i])), color='black', size=8, horizontalalignment='center', verticalalignment='center')

        ax[j].axis('equal')
        if networkStyle=='mlp':
            ax[j].set(xlim=(-2, 2), ylim=(-2, 2))
        else:
            #ax[j].set(xlim=(-1, 1), ylim=(-1, 1))
            ax[j].set(xlim=(-4, 4), ylim=(-4, 4))

    n = autoSaveFigure('figures/3MDS60_'+differenceCodeText+'meanJudgement_', networkStyle, blockTrain, seqTrain, labelNumerosity, labelContext, True, noise_std,  retainHiddenState,saveFig)

# ---------------------------------------------------------------------------- #

def averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, givenContext, counter):
    """This function will average the hidden unit activations over one of the two numbers involved in the representation:
    either the reference or the judgement number. This is so that we can then compare to Fabrice's plots
     which are averaged over the previously presented number (input B).
    Prior to performing the MDS we want to know whether to flatten over a particular value
    i.e. if plotting for reference value, flatten over the judgement value and vice versa.
     - dimKeep = 'reference' or 'judgement'
    """

    # initializing
    uniqueValues = [int(np.unique(labels_judgeValues)[i]) for i in range(len(np.unique(labels_judgeValues)))]
    Ncontexts = 3
    flat_activations = np.zeros((Ncontexts,len(uniqueValues),activations.shape[1]))
    flat_values = np.zeros((Ncontexts,len(uniqueValues),1))
    flat_outcomes = np.empty((Ncontexts,len(uniqueValues),1))
    flat_contexts = np.empty((Ncontexts,len(uniqueValues),1))
    flat_counter = np.zeros((Ncontexts,len(uniqueValues),1))
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
                        flat_counter[context,value-1] += counter[i]
                        divisor[context,value-1] +=1

            # take the mean i.e. normalise by the number of instances that met that condition
            if int(divisor[context,value-1]) == 0:
                flat_activations[context, value-1] = np.full_like(flat_activations[context, value-1], np.nan)
            else:
                flat_activations[context, value-1] = np.divide(flat_activations[context, value-1, :], divisor[context,value-1])

    # now cast out all the null instances e.g 1-5, 10-15 in certain contexts
    flat_activations, flat_contexts, flat_values, flat_outcomes, flat_counter = [dset.flattenFirstDim(i) for i in [flat_activations, flat_contexts, flat_values, flat_outcomes, flat_counter]]
    sl_activations, sl_refValues, sl_judgeValues, sl_contexts, sl_MDSlabels, sl_counter = [[] for i in range(6)]

    for i in range(flat_activations.shape[0]):
        checknan = np.asarray([ np.isnan(flat_activations[i][j]) for j in range(len(flat_activations[i]))])
        if (checknan).all():
            pass
        else:
            sl_activations.append(flat_activations[i])
            sl_contexts.append(flat_contexts[i])
            sl_MDSlabels.append(flat_outcomes[i])
            sl_counter.append(flat_counter[i])

            if dimKeep == 'reference':
                sl_refValues.append(flat_values[i])
                sl_judgeValues.append(0)
            else:
                sl_refValues.append(0)
                sl_judgeValues.append(flat_values[i])

    # finally, reshape the outputs so that they match our inputs nicely
    sl_activations, sl_refValues, sl_judgeValues, sl_contexts, sl_MDSlabels, sl_counter = [np.asarray(i) for i in [sl_activations, sl_refValues, sl_judgeValues, sl_contexts, sl_MDSlabels, sl_counter]]
    if dimKeep == 'reference':
        sl_judgeValues = np.expand_dims(sl_judgeValues, axis=1)
    else:
        sl_refValues = np.expand_dims(sl_refValues, axis=1)

    return sl_activations, sl_contexts, sl_MDSlabels, sl_refValues, sl_judgeValues, sl_counter

# ---------------------------------------------------------------------------- #

def diff_averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, givenContext, counter):
    """This function will average the hidden unit activations over one of the two numbers involved in the representation:
    either the reference or the judgement number. This is so that we can then compare to Fabrice's plots
     which are averaged over the previously presented number (input B).
    Prior to performing the MDS we want to know whether to flatten over a particular value
    i.e. if plotting for reference value, flatten over the judgement value and vice versa.
     - dimKeep = 'reference' or 'judgement'
     This is a variant which plots the difference code.
    """

    # initializing
    uniqueValues = [i for i in range(-14,14)] # hacked for now
    #uniqueValues = [int(np.unique(labels_judgeValues)[i]) for i in range(len(np.unique(labels_judgeValues)))]
    Ncontexts = 3
    flat_activations = np.zeros((Ncontexts,len(uniqueValues),activations.shape[1]))
    flat_values = np.zeros((Ncontexts,len(uniqueValues),1))
    flat_outcomes = np.empty((Ncontexts,len(uniqueValues),1))
    flat_contexts = np.empty((Ncontexts,len(uniqueValues),1))
    flat_counter = np.zeros((Ncontexts,len(uniqueValues),1))
    divisor = np.zeros((Ncontexts,len(uniqueValues)))


    # which label to flatten over (we keep whichever dimension is dimKeep, and average over the other)

    flattenValues = [labels_judgeValues[i] - labels_refValues[i] for i in range(len(labels_refValues))]

    # pick out all the activations that meet this condition for each context and then average over them
    for context in range(Ncontexts):
        for value in uniqueValues:
            for i in range(len(flattenValues)):
                if labels_contexts[i] == context+1:  # remember to preserve the context structure
                    if flattenValues[i] == value:
                        flat_activations[context, value-1,:] += activations[i]
                        flat_contexts[context,value-1] = context
                        flat_values[context,value-1] = value
                        flat_outcomes[context,value-1] = MDSlabels[i]
                        flat_counter[context,value-1] += counter[i]
                        divisor[context,value-1] +=1

            # take the mean i.e. normalise by the number of instances that met that condition
            if int(divisor[context,value-1]) == 0:
                flat_activations[context, value-1] = np.full_like(flat_activations[context, value-1], np.nan)
            else:
                flat_activations[context, value-1] = np.divide(flat_activations[context, value-1, :], divisor[context,value-1])

    # now cast out all the null instances e.g 1-5, 10-15 in certain contexts
    flat_activations, flat_contexts, flat_values, flat_outcomes, flat_counter = [dset.flattenFirstDim(i) for i in [flat_activations, flat_contexts, flat_values, flat_outcomes, flat_counter]]
    sl_activations, sl_refValues, sl_judgeValues, sl_contexts, sl_MDSlabels, sl_counter, sl_diffValues = [[] for i in range(7)]

    for i in range(flat_activations.shape[0]):
        checknan = np.asarray([ np.isnan(flat_activations[i][j]) for j in range(len(flat_activations[i]))])
        if (checknan).all():
            pass
        else:
            sl_activations.append(flat_activations[i])
            sl_contexts.append(flat_contexts[i])
            sl_MDSlabels.append(flat_outcomes[i])
            sl_counter.append(flat_counter[i])

            # hack for now
            sl_refValues.append(0)
            sl_diffValues.append(flat_values[i])
            sl_judgeValues.append(0)


    # finally, reshape the outputs so that they match our inputs nicely
    sl_activations, sl_refValues, sl_judgeValues, sl_contexts, sl_MDSlabels, sl_counter, sl_diffValues = [np.asarray(i) for i in [sl_activations, sl_refValues, sl_judgeValues, sl_contexts, sl_MDSlabels, sl_counter, sl_diffValues]]

    sl_judgeValues = np.expand_dims(sl_judgeValues, axis=1)
    sl_refValues = np.expand_dims(sl_refValues, axis=1)

    return sl_activations, sl_contexts, sl_MDSlabels, sl_refValues, sl_judgeValues, sl_counter, sl_diffValues

# ---------------------------------------------------------------------------- #

def animate3DMDS(MDS_dict, params, plot_diff_code):
    """ This function will plot the numerosity labeled, context-marked MDS projections
     of the hidden unit activations on a 3D plot, animate/rotate that plot to view it
     from different angles and optionally save it as a mp4 file.
     - use the flag 'plot_diff_code' to plot the difference signal (A-B) rather than the A activations
    """
    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, saveFig = params
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    if plot_diff_code:
        slMDS = MDS_dict["MDS_diff_slactivations"]
        labels = MDS_dict["sl_diffValues"]
        differenceCodeText = 'differencecode_'
        # which MDS points correspond to which contexts
        contextA = range(27)
        contextB = range(27,45)
        contextC = range(45, 63)
    else:
        slMDS = MDS_dict["MDS_slactivations"]
        labels = MDS_dict["sl_judgeValues"]
        differenceCodeText = ''
        contextA = range(15)
        contextB = range(15,25)
        contextC = range(25, 35)

    def init():

        points = [contextA, contextB, contextC] #if labelContext else [contextA]

        for i in range(len(points)):
            ax.scatter(slMDS[points[i], 0], slMDS[points[i], 1], slMDS[points[i], 2], color=contextcolours[i])

            if not plot_diff_code:  # the difference code is arranged differently
                ax.plot(slMDS[points[i], 0], slMDS[points[i], 1], slMDS[points[i], 2], color=contextcolours[i])
            for j in range(len(points[i])):
                label = str(int(labels[points[i][j]]))
                ax.text(slMDS[points[i][j], 0], slMDS[points[i][j], 1], slMDS[points[i][j], 2], label, color='black', size=8, horizontalalignment='center', verticalalignment='center')
        ax.set_xlabel('MDS dim 1')
        ax.set_ylabel('MDS dim 2')
        ax.set_zlabel('MDS dim 3')
        return fig,

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,

    # Animate.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)

    # save the animation as an mp4.
    if saveFig:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        strng = autoSaveFigure('animations/MDS_3Danimation_'+ differenceCodeText, networkStyle, blockTrain, seqTrain, True, labelContext, True, noise_std,  retainHiddenState,False)
        anim.save(strng+'.mp4', writer=writer)

# ---------------------------------------------------------------------------- #

def instanceCounter(MDS_dict, params):
    """ Plot a histogram showing the number of times each unique input (reference averaged) and context was in the generated training set."""

    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, saveFig = params
    plt.figure()
    rangeA = np.arange(15)
    rangeB = np.arange(15,25)
    rangeC = np.arange(25, 35)
    y = MDS_dict["sl_counter"].flatten()

    plt.bar(rangeA, y[rangeA], color='gold', edgecolor = 'gold')
    plt.bar(rangeB, y[rangeB], color='dodgerblue', edgecolor = 'dodgerblue')
    plt.bar(rangeC, y[rangeC], color='orangered', edgecolor = 'orangered')
    plt.xlabel('Numbers and contexts')
    plt.ylabel('Instances in training set')

    n = autoSaveFigure('figures/InstanceCounter_meanJudgement', networkStyle, blockTrain, seqTrain, True, labelContext, True, noise_std,  retainHiddenState, saveFig)

# ---------------------------------------------------------------------------- #

def viewTrainingSequence(MDS_dict, params):
    """Take the data loader and view how the contexts and latent states evolved in time in the training set."""

    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, saveFig = params
    MDS_latentstate = MDS_dict["drift"]["MDS_latentstate"]
    temporal_context = MDS_dict["drift"]["temporal_context"]

    # context in time/trials in training set
    plt.figure()
    plt.plot(temporal_context.flatten())
    plt.xlabel('Trials in training set')
    plt.ylabel('Context (0: 1-15; 1: 1-10; 2: 6-15)')
    n = autoSaveFigure('figures/temporalcontext_', networkStyle, blockTrain, seqTrain, True, labelContext, True, noise_std,  retainHiddenState, saveFig)

    # latent state drift in time/trials in training set
    fig,ax = plt.subplots(1,3, figsize=(18,5))

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

        ax[j].set_title('latent state drift')

        # perhaps draw a coloured line between adjacent numbers
        # ax[j].plot(MDS_latentstate[:, dimA], MDS_latentstate[:, dimB], color='grey')

        #for i in range((MDS_latentstate.shape[0])):
        for i in range(2000,3500): # a subset of trials
            # colour by context
            ax[j].scatter(MDS_latentstate[i, dimA], MDS_latentstate[i, dimB], color=contextcolours[int(temporal_context[i])-1], s=20)
            ax[j].plot([MDS_latentstate[i, dimA], MDS_latentstate[i+1, dimA]], [MDS_latentstate[i, dimB],MDS_latentstate[i+1, dimB]], color=contextcolours[int(temporal_context[i])-1])

        ax[j].axis('equal')
        #ax[j].set(xlim=(-4, 4), ylim=(-4, 4))

    n = autoSaveFigure('figures/latentstatedrift_', networkStyle, blockTrain, seqTrain, True, labelContext, True, noise_std,  retainHiddenState, saveFig)

# ---------------------------------------------------------------------------- #

def animate3DdriftMDS(MDS_dict, params):
    """ This function will plot the latent state drift MDS projections
     on a 3D plot, animate/rotate that plot to view it
     from different angles and optionally save it as a mp4 file.
    """
    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, saveFig = params
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    MDS_latentstate = MDS_dict["drift"]["MDS_latentstate"]
    temporal_context = MDS_dict["drift"]["temporal_context"]

    def init():

        #points = [contextA, contextB, contextC] #if labelContext else [contextA]

        for i in range(2000,3500):
            ax.scatter(MDS_latentstate[i, 0], MDS_latentstate[i, 1], MDS_latentstate[i, 2], color=contextcolours[int(temporal_context[i])-1])
            #ax.plot(slMDS[points[i], 0], slMDS[points[i], 1], slMDS[points[i], 2], color=contextcolours[i])

        ax.set_xlabel('MDS dim 1')
        ax.set_ylabel('MDS dim 2')
        ax.set_zlabel('MDS dim 3')
        return fig,

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,

    # Animate.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)

    # save the animation as an mp4.
    if saveFig:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        strng = autoSaveFigure('animations/latentdrift_MDS_3Danimation_', networkStyle, blockTrain, seqTrain, True, labelContext, True, noise_std,  retainHiddenState,False)
        anim.save(strng+'.mp4', writer=writer)

# ---------------------------------------------------------------------------- #
