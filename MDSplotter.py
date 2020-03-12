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

def autoSaveFigure(basetitle, args, networkStyle, blockTrain, seqTrain, labelNumerosity, givenContext, labelContexts, noise_std, retainHiddenState, plot_diff_code, whichTrialType, allFullRange, whichContext, saveFig):
    """This function will save the currently open figure with a base title and some details pertaining to how the activations were generated."""

    # conver the hyperparameter settings into a string ID
    str_args = '_bs'+ str(args.batch_size_multi[0]) + '_lr' + str(args.lr_multi[0]) + '_ep' + str(args.epochs) + '_r' + str(args.recurrent_size) + '_h' + str(args.hidden_size) + '_bpl' + str(args.BPTT_len)

    # automatic save file title details
    if whichContext==0:
        whichcontexttext = ''
    elif whichContext==1:
        whichcontexttext = '_fullrange_1-15_only'
    elif whichContext==2:
        whichcontexttext = '_lowrange_1-10_only'
    elif whichContext==3:
        whichcontexttext = '_highrange_6-15_only'
    diffcodetext = '_diffcode' if plot_diff_code else ''
    retainstatetext = '_retainstate' if retainHiddenState else '_resetstate'
    blockedtext = '_blck' if blockTrain else ''
    seqtext = '_seq' if seqTrain else ''
    labeltext = '_number' if labelNumerosity else '_outcomes'
    contexts = '_contexts' if labelContexts else ''
    networkTxt = 'RNN' if networkStyle == 'recurrent' else 'MLP'
    trialtypetxt = '_compare' if whichTrialType == 'compare' else '_filler'
    numberrangetxt = '_numrangeblocked' if allFullRange==False else '_numrangeintermingled'
    if givenContext=='true':
        contextlabelledtext = '_truecontextlabel'
    elif givenContext=='random':
        contextlabelledtext =  '_randcontextlabel'
    elif givenContext=='constant':
        contextlabelledtext =  '_constcontextlabel'

    if saveFig:
        plt.savefig(basetitle+networkTxt+whichcontexttext+numberrangetxt+diffcodetext+trialtypetxt+contextlabelledtext+blockedtext+seqtext+labeltext+contexts+retainstatetext+'_n'+str(noise_std)+str_args+'.pdf',bbox_inches='tight')

    return basetitle+networkTxt+whichcontexttext+numberrangetxt+diffcodetext+trialtypetxt+contextlabelledtext+blockedtext+seqtext+labeltext+contexts+retainstatetext+'_n'+str(noise_std)+str_args

# ---------------------------------------------------------------------------- #

def activationRDMs(MDS_dict, args, params, plot_diff_code, whichTrialType='compare'):
    """Plot the representational disimilarity structure of the hidden unit activations, sorted by context, and within that magnitude.
    Context order:  1-15, 1-10, 5-15
     - use the flag 'plot_diff_code' to plot the difference signal (A-B) rather than the A activations
    """
    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange, whichContext, saveFig = params

    if whichTrialType=='filler':
        MDS_dict = MDS_dict["filler_dict"]

    fig = plt.figure(figsize=(5,3))
    ax = plt.gca()
    if plot_diff_code:
        D = pairwise_distances(MDS_dict["diff_sl_activations"])
        labelticks = ['-14:+14', '-9:+9', '-9:+9']
        ticks = [0,27,45]
        differenceCodeText = 'differencecode_'
    else:
        D = pairwise_distances(MDS_dict["sl_activations"])  # note that activations are structured by: context (1-15,1-10,5-15) and judgement value magnitude within that.
        if whichTrialType == 'filler':
            labelticks = ['1-15', '1-15', '1-15']
            ticks = [0,15,30]
        else:
            labelticks = ['1-15', '1-10', '6-15']
            ticks = [0,15,25]
        differenceCodeText = ''

    im = plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest', vmin=0, vmax=5)

    #    divider = make_axes_locatable(ax[1])
    #    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im)
    cbar.set_label('disimilarity')
    ax.set_title('Averaged activations')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labelticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labelticks)

    n = autoSaveFigure('figures/RDM_'+differenceCodeText, args, networkStyle, blockTrain, seqTrain, False, labelContext, False, noise_std,  retainHiddenState, plot_diff_code, whichTrialType, allFullRange, whichContext, saveFig)

# ---------------------------------------------------------------------------- #

def plot3MDS(MDS_dict, args, params, labelNumerosity=True, whichTrialType='compare'):
    """This is a function to plot the MDS of activations and label according to numerosity and context"""

    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange, whichContext, saveFig = params

    if whichTrialType=='filler':
        MDS_dict = MDS_dict["filler_dict"]

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

        n = autoSaveFigure('figures/3MDS60_'+tx, args, networkStyle, blockTrain, seqTrain, labelNumerosity, labelContext, False, noise_std, retainHiddenState, False, whichTrialType, allFullRange, whichContext, saveFig)

# ---------------------------------------------------------------------------- #

def plot3MDSContexts(MDS_dict, args, labelNumerosity, params, whichTrialType='compare'):
    """This is a just function to plot the MDS of activations and label the dots with the colour of the context."""

    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange, whichContext, saveFig = params
    if whichTrialType=='filler':
        MDS_dict = MDS_dict["filler_dict"]

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

    n = autoSaveFigure('figures/3MDS60_', args, networkStyle, blockTrain, seqTrain, labelNumerosity, labelContext, True, noise_std,  retainHiddenState, False, whichTrialType, allFullRange, whichContext, saveFig)

# ---------------------------------------------------------------------------- #

def plot3MDSMean(MDS_dict, args, params, labelNumerosity=True, plot_diff_code=False, whichTrialType='compare'):
    """This function is just like plot3MDS and plot3MDSContexts but for the formatting of the data which has been averaged across one of the two numerosity values.
    Because there are fewer datapoints I also label the numerosity inside each context, like Fabrice does.
     - use the flag 'plot_diff_code' to plot the difference signal (A-B) rather than the A activations
    """
    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange, whichContext, saveFig = params

    if whichTrialType=='filler':
        MDS_dict = MDS_dict["filler_dict"]

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
            if whichTrialType=='filler':
                contextA = range(15)
                contextB = range(15,30)
                contextC = range(30, 45)
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

    n = autoSaveFigure('figures/3MDS60_'+differenceCodeText+'meanJudgement_', args, networkStyle, blockTrain, seqTrain, labelNumerosity, labelContext, True, noise_std,  retainHiddenState, plot_diff_code, whichTrialType, allFullRange, whichContext, saveFig)

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
    """
     This is a hacky variant of averageReferenceNumerosity(), which averages over numbers which have the same difference (A-B).
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

def animate3DMDS(MDS_dict, args, params, plot_diff_code=False, whichTrialType='compare'):
    """ This function will plot the numerosity labeled, context-marked MDS projections
     of the hidden unit activations on a 3D plot, animate/rotate that plot to view it
     from different angles and optionally save it as a mp4 file.
     - use the flag 'plot_diff_code' to plot the difference signal (A-B) rather than the A activations
    """
    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange, whichContext, saveFig = params

    if whichTrialType=='filler':
        MDS_dict = MDS_dict["filler_dict"]

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
        if whichTrialType=='filler':
            contextA = range(15)
            contextB = range(15,30)
            contextC = range(30, 45)

        else:
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
        strng = autoSaveFigure('animations/MDS_3Danimation_'+ differenceCodeText, args, networkStyle, blockTrain, seqTrain, True, labelContext, True, noise_std,  retainHiddenState, plot_diff_code, whichTrialType, allFullRange, whichContext, False)
        anim.save(strng+'.mp4', writer=writer)

# ---------------------------------------------------------------------------- #

def instanceCounter(MDS_dict, args, params, whichTrialType='compare'):
    """ Plot a histogram showing the number of times each unique input (reference averaged) and context was in the generated training set."""

    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, whichContext, saveFig = params
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

    n = autoSaveFigure('figures/InstanceCounter_meanJudgement', args, networkStyle, blockTrain, seqTrain, True, labelContext, True, noise_std,  retainHiddenState, False, whichTrialType, allFullRange, whichContext, saveFig)

# ---------------------------------------------------------------------------- #

def viewTrainingSequence(MDS_dict, args, params, whichTrialType='compare'):
    """Take the data loader and view how the contexts and latent states evolved in time in the training set.
    Also plots the sequence of compare vs filler trials.
    """

    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange, whichContext, saveFig = params
    MDS_latentstate = MDS_dict["drift"]["MDS_latentstate"]
    temporal_context = MDS_dict["drift"]["temporal_context"]
    temporal_trialtypes = MDS_dict["temporal_trialtypes"]

    # context in time/trials in training set
    plt.figure()
    plt.plot(temporal_context.flatten())
    plt.xlabel('Trials in training set')
    plt.ylabel('Context (0: 1-15; 1: 1-10; 2: 6-15)')
    n = autoSaveFigure('figures/temporalcontext_', args, networkStyle, blockTrain, seqTrain, True, labelContext, True, noise_std,  retainHiddenState, False, whichTrialType, allFullRange, whichContext, saveFig)

    # trial types changing with time in training set
    plt.figure()
    plt.plot(temporal_trialtypes.flatten())
    plt.xlabel('Trials in training set')
    plt.ylabel('Trial type: 0-filler; 1-compare')
    n = autoSaveFigure('figures/temporaltrialtype_', args, networkStyle, blockTrain, seqTrain, True, labelContext, True, noise_std,  retainHiddenState, False, whichTrialType, allFullRange, whichContext, saveFig)

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

    n = autoSaveFigure('figures/latentstatedrift_', args, networkStyle, blockTrain, seqTrain, True, labelContext, True, noise_std,  retainHiddenState, False, whichTrialType, allFullRange, whichContext, saveFig)

# ---------------------------------------------------------------------------- #

def animate3DdriftMDS(MDS_dict, args, params, whichTrialType='compare'):
    """ This function will plot the latent state drift MDS projections
     on a 3D plot, animate/rotate that plot to view it
     from different angles and optionally save it as a mp4 file.
    """
    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange, whichContext, saveFig = params
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
        strng = autoSaveFigure('animations/latentdrift_MDS_3Danimation_', args, networkStyle, blockTrain, seqTrain, True, labelContext, True, noise_std,  retainHiddenState, False, whichTrialType, allFullRange, whichContext, False)
        anim.save(strng+'.mp4', writer=writer)

# ---------------------------------------------------------------------------- #


def performanceMean(number_differences, performance):
    """
    This function calculates the mean network performance as a function of the distance between the current number and some mean context signal
    - the absolute difference |(current - mean)| signal is already in number_differences
    """
    unique_diffs = np.unique(number_differences)
    tally = np.zeros((len(unique_diffs),))    # a counter for computing mean
    aggregate_perf = np.zeros((len(unique_diffs),))
    for i in range(len(unique_diffs)):
        num = unique_diffs[i]
        ind = np.argwhere([number_differences[i]==num for i in range(len(number_differences))])
        tally[i] = len(ind)
        for k in range(len(ind)):
            aggregate_perf[i] += performance[ind[k][0]]
    mean_performance = np.divide(aggregate_perf, tally)
    return mean_performance, unique_diffs

# ---------------------------------------------------------------------------- #

def perfVdistContextMean(params, testParams):
    """
    Plot performance after just a single lesion vs the (absolute) distance of the seen number to the context mean
    Also plot vs distance for the (absolute) distance to the global mean. We are hoping for some linear trends with
     a stronger trend for the former (local context).
    """
    # set up lesioned network parameters
    args, trained_model, device, testloader, criterion, retainHiddenState, printOutput = testParams
    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange, whichContext  = params
    whichLesion = 'number'
    lesionFrequency = 0.0  # just lesion the second to last compare trial and assess on the final compare trial in each sequence

    # load analysis of network at test with lesions
    blcktxt = '_interleaved' if allFullRange else '_temporalblocked'
    contexttxt = '_contextcued' if labelContext=='true' else '_nocontextcued'
    darkcolours = ['darkgoldenrod','blue','darkred']
    if whichContext==0:  # proceed as normal
        nmodels = 1
    else:
        nmodels = 3      # load in models trained on just a single context and compare them
        whichContexttxt = '_contextmodelstrainedseparately'
        context_handles = []
        allglobal_meanperf = []
        allglobal_uniquediffs = []

    context_perf = [[] for i in range(3)]
    context_numberdiffs = [[] for i in range(3)]
    context_globalnumberdiff = [[] for i in range(3)]
    plt.figure()

    if whichContext==0:
        data = (np.load('network_analysis/lesion_tests/lesiontests'+blcktxt+contexttxt+str(lesionFrequency)+'.npy', allow_pickle=True)).item()
        data = data["bigdict_lesionperf"]

        # evaluate the context mean for each network assessment
        contextmean = np.zeros((data.shape[0],data.shape[1]))
        numberdiffs = np.zeros((data.shape[0],data.shape[1]))
        globalnumberdiffs = np.zeros((data.shape[0],data.shape[1]))
        perf = np.zeros((data.shape[0],data.shape[1]))
        globalmean = 8

        for seq in range(data.shape[0]):
            for compare_idx in range(data.shape[1]):
                context = data[seq][compare_idx]["underlying_context"]
                if context==1:
                    contextmean[seq][compare_idx] = 8
                elif context==2:
                    contextmean[seq][compare_idx] = 5.5
                elif context==3:
                    contextmean[seq][compare_idx] = 10.5

                # calculate difference between current number and context or global mean
                numberdiffs[seq][compare_idx] = np.abs(np.asarray(data[seq][compare_idx]["assess_number"]-contextmean[seq][compare_idx]))
                globalnumberdiffs[seq][compare_idx] = np.abs(np.asarray(data[seq][compare_idx]["assess_number"]-globalmean))
                perf[seq][compare_idx] = data[seq][compare_idx]["lesion_perf"]

                # context-specific
                context_perf[context-1].append(perf[seq][compare_idx])
                context_numberdiffs[context-1].append(numberdiffs[seq][compare_idx])
                context_globalnumberdiff[context-1].append(globalnumberdiffs[seq][compare_idx])


        # flatten across sequences and the trials in those sequences
        globalnumberdiffs = dset.flattenFirstDim(globalnumberdiffs)
        numberdiffs = dset.flattenFirstDim(numberdiffs)
        perf = dset.flattenFirstDim(perf)
        meanperf, uniquediffs = performanceMean(numberdiffs, perf)
        global_meanperf, global_uniquediffs = performanceMean(globalnumberdiffs, perf)

        global_contextmean, = plt.plot(global_uniquediffs, global_meanperf, color='black') # this is captured already by the separation into contexts and looks jiggly because of different context values on x-axis

    else:
        for whichmodel in range(nmodels):
            colourindex = 3 if  whichContext==0 else whichmodel
            whichContext = whichmodel+1  # update to the context-specific model we want
            if whichContext==0:
                range_txt = ''
            elif whichContext==1:
                range_txt = '_fullrangeonly'
            elif whichContext==2:
                range_txt = '_lowrangeonly'
            elif whichContext==3:
                range_txt = '_highrangeonly'
            data = (np.load('network_analysis/lesion_tests/lesiontests'+blcktxt+contexttxt+range_txt+str(lesionFrequency)+'.npy', allow_pickle=True)).item()
            data = data["bigdict_lesionperf"]

            # evaluate the context mean for each network assessment
            contextmean = np.zeros((data.shape[0],data.shape[1]))
            numberdiffs = np.zeros((data.shape[0],data.shape[1]))
            globalnumberdiffs = np.zeros((data.shape[0],data.shape[1]))
            perf = np.zeros((data.shape[0],data.shape[1]))
            globalmean = 8

            for seq in range(data.shape[0]):
                for compare_idx in range(data.shape[1]):
                    context = data[seq][compare_idx]["underlying_context"]
                    if context==1:
                        contextmean[seq][compare_idx] = 8
                    elif context==2:
                        contextmean[seq][compare_idx] = 5.5
                    elif context==3:
                        contextmean[seq][compare_idx] = 10.5

                    # calculate difference between current number and context or global mean
                    numberdiffs[seq][compare_idx] = np.abs(np.asarray(data[seq][compare_idx]["assess_number"]-contextmean[seq][compare_idx]))
                    globalnumberdiffs[seq][compare_idx] = np.abs(np.asarray(data[seq][compare_idx]["assess_number"]-globalmean))
                    perf[seq][compare_idx] = data[seq][compare_idx]["lesion_perf"]

                    # context-specific
                    context_perf[whichmodel].append(perf[seq][compare_idx])
                    context_numberdiffs[whichmodel].append(numberdiffs[seq][compare_idx])
                    context_globalnumberdiff[whichmodel].append(globalnumberdiffs[seq][compare_idx])


            # flatten across sequences and the trials in those sequences
            globalnumberdiffs = dset.flattenFirstDim(globalnumberdiffs)
            numberdiffs = dset.flattenFirstDim(numberdiffs)
            perf = dset.flattenFirstDim(perf)
            meanperf, uniquediffs = performanceMean(numberdiffs, perf)
            global_meanperf, global_uniquediffs = performanceMean(globalnumberdiffs, perf)

            global_contextmean, = plt.plot(global_uniquediffs, global_meanperf, color=darkcolours[whichmodel]) # this is captured already by the separation into contexts and looks jiggly because of different context values on x-axis
            context_handles.append(global_contextmean)
            allglobal_uniquediffs.append(global_uniquediffs)
            allglobal_meanperf.append(global_meanperf)

    # assess mean performance under each context (HRS hacky for now)
    context1_meanperf, context1_uniquediffs = performanceMean(context_numberdiffs[0], context_perf[0])
    context2_meanperf, context2_uniquediffs = performanceMean(context_numberdiffs[1], context_perf[1])
    context3_meanperf, context3_uniquediffs = performanceMean(context_numberdiffs[2], context_perf[2])

    # plot and save the figure
    ref7, = plt.plot([0,7],[.5,1],linestyle=':',color='grey')
    ref4, = plt.plot([0,4.5],[.5,1],linestyle=':',color='grey')

    # context-specific performance i.e. how did performance change with dist. to mean in each context
    local_contextmean_context1, = plt.plot(context1_uniquediffs, context1_meanperf, color='gold')
    local_contextmean_context2, = plt.plot(context2_uniquediffs, context2_meanperf, color='dodgerblue')
    local_contextmean_context3, = plt.plot(context3_uniquediffs, context3_meanperf, color='orangered')

    if whichContext==0:
        plt.legend((ref7, global_contextmean, local_contextmean_context1, local_contextmean_context2, local_contextmean_context3), ('unity refs, max|\u0394|={4.5,7}', '\u03BC = global median', '\u03BC = local median | context A, 1:15','\u03BC = local median | context B, 1:10','\u03BC = local median | context C, 6-15'))
    else:
        allglobal_uniquediffs = dset.flattenFirstDim(np.asarray(allglobal_uniquediffs))
        allglobal_meanperf = dset.flattenFirstDim(np.asarray(allglobal_meanperf))
        meanglobal_meanperf, meanglobal_uniquediffs = performanceMean(allglobal_uniquediffs, allglobal_meanperf)
        totalglobal, = plt.plot(meanglobal_uniquediffs, meanglobal_meanperf, 'black')
        plt.legend((ref7, totalglobal, context_handles[0], context_handles[1], context_handles[2], local_contextmean_context1, local_contextmean_context2, local_contextmean_context3), ('unity refs, max|\u0394|={4.5,7}', 'global \u03BC | average across all 3 nets', 'global \u03BC | context A: 1-15','global \u03BC | context B: 1-10','global \u03BC | context C: 6-15', 'local \u03BC | context A: 1-15','local \u03BC | context B: 1-10','local \u03BC | context C: 6-15'))

    plt.ylabel('RNN perf. immediately post-lesion (just 1 lesion)')
    plt.xlabel('|current# - \u03BC|')
    ax = plt.gca()
    ax.set_ylim(0.45,1.05)
    plt.title('RNN ('+blcktxt[1:]+', '+contexttxt[1:]+whichContexttxt+')')
    whichTrialType = 'compare'
    autoSaveFigure('figures/perf_v_distToContextMean_postlesion_'+whichContexttxt, args, networkStyle, blockTrain, seqTrain, True, labelContext, False, noise_std, retainHiddenState, False, whichTrialType, allFullRange, whichContext, True)

# ---------------------------------------------------------------------------- #
