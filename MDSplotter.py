"""
This is a selection of functions for plotting MDS projections of relative magnitude-trained networks.

Author: Hannah Sheahan, sheahan.hannah@gmail.com
Date: 14/12/2019
Notes: N/A
Issues: N/A
"""
# ---------------------------------------------------------------------------- #
import define_dataset as dset
import constants as const
import magnitude_network as mnet
import analysis_helpers as anh

from mpl_toolkits import mplot3d
import numpy as np
import scipy as sp
import copy
import sys
import random
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation

from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.utils import shuffle

# generic plotting settings
contextcolours = ['gold', 'dodgerblue', 'orangered', 'black']   # 1-16, 1-11, 6-16 like fabrices colours

# ---------------------------------------------------------------------------- #

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

# ---------------------------------------------------------------------------- #

def autoSaveFigure(basetitle, args, labelNumerosity, plot_diff_code, whichTrialType, saveFig):
    """This function will save the currently open figure with a base title and some details pertaining to how the activations were generated."""

    # conver the hyperparameter settings into a string ID
    str_args = '_bs'+ str(args.batch_size_multi[0]) + '_lr' + str(args.lr_multi[0]) + '_ep' + str(args.epochs) + '_r' + str(args.recurrent_size) + '_h' + str(args.hidden_size) + '_bpl' + str(args.BPTT_len) + '_trlf' + str(args.train_lesion_freq) + '_id' + str(args.model_id)

    # automatic save file title details
    if args.which_context==0:
        whichcontexttext = ''
    elif args.which_context==1:
        whichcontexttext = '_fullrange_1-16_only'
    elif args.which_context==2:
        whichcontexttext = '_lowrange_1-11_only'
    elif args.which_context==3:
        whichcontexttext = '_highrange_6-16_only'
    diffcodetext = '_diffcode' if plot_diff_code else ''
    retainstatetext = '_retainstate' if args.retain_hidden_state else '_resetstate'
    labeltext = '_number' if labelNumerosity else '_outcomes'
    networkTxt = 'RNN' if args.network_style == 'recurrent' else 'MLP'
    trialtypetxt = '_compare' if whichTrialType == 'compare' else '_filler'
    numberrangetxt = '_numrangeblocked' if args.all_fullrange==False else '_numrangeintermingled'
    if args.label_context=='true':
        contextlabelledtext = '_truecontextlabel'
    elif args.label_context=='random':
        contextlabelledtext =  '_randcontextlabel'
    elif args.label_context=='constant':
        contextlabelledtext =  '_constcontextlabel'

    if saveFig:
        plt.savefig(basetitle+networkTxt+whichcontexttext+numberrangetxt+diffcodetext+trialtypetxt+contextlabelledtext+labeltext+retainstatetext+'_n'+str(args.noise_std)+str_args+'.pdf',bbox_inches='tight')

    plt.close()
    return basetitle+networkTxt+whichcontexttext+numberrangetxt+diffcodetext+trialtypetxt+contextlabelledtext+labeltext+retainstatetext+'_n'+str(args.noise_std)+str_args

# ---------------------------------------------------------------------------- #

def activationRDMs(MDS_dict, args, plot_diff_code, whichTrialType='compare', saveFig=True):
    """Plot the representational disimilarity structure of the hidden unit activations, sorted by context, and within that magnitude.
    Context order:  1-16, 1-11, 5-16
     - use the flag 'plot_diff_code' to plot the difference signal (A-B) rather than the A activations
    """
    if whichTrialType=='filler':
        MDS_dict = MDS_dict["filler_dict"]

    fig = plt.figure(figsize=(5,3))
    ax = plt.gca()
    if plot_diff_code:
        D = pairwise_distances(MDS_dict["diff_sl_activations"])
        labelticks = ['-15:+15', '-10:+10', '-10:+10']
        ticks = [0,(const.FULLR_SPAN-1)*2, (const.FULLR_SPAN-1)*2 + (const.LOWR_SPAN-1)*2]
        differenceCodeText = 'differencecode_'
    else:
        D = pairwise_distances(MDS_dict["sl_activations"])  # note that activations are structured by: context (1-15,1-10,5-15) and judgement value magnitude within that.
        if whichTrialType == 'filler':
            labelticks = ['1-16', '1-16', '1-16']
            ticks = [0,const.FULLR_SPAN,const.FULLR_SPAN*2]
        else:
            labelticks = ['1-16', '1-11', '6-16']
            ticks = [0, const.FULLR_SPAN, const.FULLR_SPAN+const.LOWR_SPAN]
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

    n = autoSaveFigure('figures/RDM_'+differenceCodeText, args, False, plot_diff_code, whichTrialType, saveFig)

# ---------------------------------------------------------------------------- #

def plot3MDS(MDS_dict, args, labelNumerosity=True, whichTrialType='compare', saveFig=True):
    """This is a function to plot the MDS of activations and label according to numerosity and context"""

    if whichTrialType=='filler':
        MDS_dict = MDS_dict["filler_dict"]

    # Plot the hidden activations for the 3 MDS dimensions
    colours = plt.cm.get_cmap('viridis')
    diffcolours = plt.cm.get_cmap('viridis')
    outcomecolours = ['red', 'green']

    norm = mplcol.Normalize(vmin=const.FULLR_LLIM, vmax=const.FULLR_ULIM)
    dnorm = mplcol.Normalize(vmin=-const.FULLR_ULIM+1, vmax=const.FULLR_ULIM-1)

    if not args.label_context:
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
                                cbar.ax.set_yticklabels(['1','16'])
                elif k==2:  # A values
                    #im = ax[k,j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=colours(int(MDS_dict["labels_judgeValues"][i])-1), edgecolors=contextcolours[int(MDS_dict["labels_contexts"][i])-1])
                    im = ax[j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=colours(norm(int(MDS_dict["labels_judgeValues"][i])-1)), s=20)
                    if j==2:
                        if i == (MDS_act.shape[0])-1:
                            cbar = fig.colorbar(im, ticks=[0,1])
                            if labelNumerosity:
                                cbar.ax.set_yticklabels(['1','16'])
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

        n = autoSaveFigure('figures/3MDS60_'+tx, args, labelNumerosity, False, whichTrialType, saveFig)

# ---------------------------------------------------------------------------- #

def plot3MDSContexts(MDS_dict, args, labelNumerosity, whichTrialType='compare', saveFig=True):
    """This is a just function to plot the MDS of activations and label the dots with the colour of the context."""

    if whichTrialType=='filler':
        MDS_dict = MDS_dict["filler_dict"]

    labels_contexts = MDS_dict["labels_contexts"]

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

    n = autoSaveFigure('figures/3MDS60_', args, labelNumerosity, False, whichTrialType, saveFig)

# ---------------------------------------------------------------------------- #

def plot3MDSMean(MDS_dict, args, labelNumerosity=True, plot_diff_code=False, whichTrialType='compare', saveFig=True):
    """This function is just like plot3MDS and plot3MDSContexts but for the formatting of the data which has been averaged across one of the two numerosity values.
    Because there are fewer datapoints I also label the numerosity inside each context, like Fabrice does.
     - use the flag 'plot_diff_code' to plot the difference signal (A-B) rather than the A activations
    """

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
            contextA = range((const.FULLR_SPAN-1)*2)
            contextB = range((const.FULLR_SPAN-1)*2, (const.FULLR_SPAN-1)*2 + (const.LOWR_SPAN-1)*2)
            contextC = range((const.FULLR_SPAN-1)*2 + (const.LOWR_SPAN-1)*2, (const.FULLR_SPAN-1)*2 + (const.LOWR_SPAN-1)*2 + (const.HIGHR_SPAN-1)*2)
        else:
            if whichTrialType=='filler':
                contextA = range(const.FULLR_SPAN)
                contextB = range(const.FULLR_SPAN,const.FULLR_SPAN*2)
                contextC = range(const.FULLR_SPAN*2, const.FULLR_SPAN*3)
            else:
                contextA = range(const.FULLR_SPAN)
                contextB = range(const.FULLR_SPAN,const.FULLR_SPAN+const.LOWR_SPAN)
                contextC = range(const.FULLR_SPAN+const.LOWR_SPAN, const.FULLR_SPAN+const.LOWR_SPAN+const.HIGHR_SPAN)

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
        if args.network_style=='mlp':
            ax[j].set(xlim=(-2, 2), ylim=(-2, 2))
        else:
            #ax[j].set(xlim=(-1, 1), ylim=(-1, 1))
            ax[j].set(xlim=(-4, 4), ylim=(-4, 4))

    n = autoSaveFigure('figures/3MDS60_'+differenceCodeText+'meanJudgement_', args, labelNumerosity, plot_diff_code, whichTrialType, saveFig)

# ---------------------------------------------------------------------------- #

def animate3DMDS(MDS_dict, args, plot_diff_code=False, whichTrialType='compare', saveFig=True):
    """ This function will plot the numerosity labeled, context-marked MDS projections
     of the hidden unit activations on a 3D plot, animate/rotate that plot to view it
     from different angles and optionally save it as a mp4 file.
     - use the flag 'plot_diff_code' to plot the difference signal (A-B) rather than the A activations
    """

    if whichTrialType=='filler':
        MDS_dict = MDS_dict["filler_dict"]

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    if plot_diff_code:
        slMDS = MDS_dict["MDS_diff_slactivations"]
        labels = MDS_dict["sl_diffValues"]
        differenceCodeText = 'differencecode_'
        # which MDS points correspond to which contexts
        contextA = range((const.FULLR_SPAN-1)*2)
        contextB = range((const.FULLR_SPAN-1)*2,(const.FULLR_SPAN-1)*2+(const.LOWR_SPAN-1)*2)
        contextC = range((const.FULLR_SPAN-1)*2+(const.LOWR_SPAN-1)*2, (const.FULLR_SPAN-1)*2+(const.LOWR_SPAN-1)*2+(const.HIGHR_SPAN-1)*2)
    else:
        slMDS = MDS_dict["MDS_slactivations"]
        labels = MDS_dict["sl_judgeValues"]
        differenceCodeText = ''
        if whichTrialType=='filler':
            contextA = range(const.FULLR_SPAN)
            contextB = range(const.FULLR_SPAN,const.FULLR_SPAN*2)
            contextC = range(const.FULLR_SPAN*2, const.FULLR_SPAN*3)

        else:
            contextA = range(const.FULLR_SPAN)
            contextB = range(const.FULLR_SPAN,const.FULLR_SPAN+const.LOWR_SPAN)
            contextC = range(const.FULLR_SPAN+const.LOWR_SPAN, const.FULLR_SPAN+const.LOWR_SPAN+const.HIGHR_SPAN)


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
        strng = autoSaveFigure('animations/MDS_3Danimation_'+ differenceCodeText, args, True,  plot_diff_code, whichTrialType, False)
        anim.save(strng+'.mp4', writer=writer)

# ---------------------------------------------------------------------------- #

def instanceCounter(MDS_dict, args, whichTrialType='compare'):
    """ Plot a histogram showing the number of times each unique input (reference averaged) and context was in the generated training set."""

    plt.figure()
    rangeA = np.arange(const.FULLR_SPAN)
    rangeB = np.arange(const.FULLR_SPAN,const.FULLR_SPAN+const.LOWR_SPAN)
    rangeC = np.arange(const.FULLR_SPAN+const.LOWR_SPAN, const.FULLR_SPAN+const.LOWR_SPAN+const.HIGHR_SPAN)
    y = MDS_dict["sl_counter"].flatten()

    plt.bar(rangeA, y[rangeA], color='gold', edgecolor = 'gold')
    plt.bar(rangeB, y[rangeB], color='dodgerblue', edgecolor = 'dodgerblue')
    plt.bar(rangeC, y[rangeC], color='orangered', edgecolor = 'orangered')
    plt.xlabel('Numbers and contexts')
    plt.ylabel('Instances in training set')

    n = autoSaveFigure('figures/InstanceCounter_meanJudgement', args, True, False, whichTrialType, saveFig)

# ---------------------------------------------------------------------------- #

def viewTrainingSequence(MDS_dict, args, whichTrialType='compare', saveFig=True):
    """Take the data loader and view how the contexts and latent states evolved in time in the training set.
    Also plots the sequence of compare vs filler trials.
    """

    MDS_latentstate = MDS_dict["drift"]["MDS_latentstate"]
    temporal_context = MDS_dict["drift"]["temporal_context"]
    temporal_trialtypes = MDS_dict["temporal_trialtypes"]

    # context in time/trials in training set
    plt.figure()
    plt.plot(temporal_context.flatten())
    plt.xlabel('Trials in training set')
    plt.ylabel('Context (0: 1-16; 1: 1-11; 2: 6-16)')
    n = autoSaveFigure('figures/temporalcontext_', args, True, False, whichTrialType, saveFig)

    # trial types changing with time in training set
    plt.figure()
    plt.plot(temporal_trialtypes.flatten())
    plt.xlabel('Trials in training set')
    plt.ylabel('Trial type: 0-filler; 1-compare')
    n = autoSaveFigure('figures/temporaltrialtype_', True, False, whichTrialType, saveFig)

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

    n = autoSaveFigure('figures/latentstatedrift_', True, False, whichTrialType, saveFig)

# ---------------------------------------------------------------------------- #

def animate3DdriftMDS(MDS_dict, args, whichTrialType='compare', saveFig=True):
    """ This function will plot the latent state drift MDS projections
     on a 3D plot, animate/rotate that plot to view it
     from different angles and optionally save it as a mp4 file.
    """
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
        strng = autoSaveFigure('animations/latentdrift_MDS_3Danimation_', True, False, whichTrialType, False)
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

def plotOptimalReferencePerformance(ax, args):
    # ***HRS these numbers need changing for the new longer number ranges
    if args.which_context==0:
        localpolicy_optimal = ax.axhline(y=77.07, linestyle=':', color='lightpink')
        globalpolicy_optimal = ax.axhline(y=73.43, linestyle=':', color='lightblue')
        #globaldistpolicy_optimal = ax.axhline(y=76.5, linestyle=':', color='lightgreen')
        #handles = [localpolicy_optimal, globalpolicy_optimal, globaldistpolicy_optimal]
        handles = [localpolicy_optimal, globalpolicy_optimal]
    else:
        #oldbenchmark1 = ax.axhline(y=77.41, linestyle=':', color='grey')
        #oldbenchmark2 = ax.axhline(y=72.58, linestyle=':', color='grey')
        #oldbenchmark3 = ax.axhline(y=76.5, linestyle=':', color='grey')
        #contextA_localpolicy = ax.axhline(y=76.67, color='gold')
        #contextBC_localpolicy = ax.axhline(y=77.78, color='orangered')
        #handles = [oldbenchmark1, oldbenchmark2, oldbenchmark3, contextA_localpolicy, contextBC_localpolicy]
        handles = []

    return handles

# ---------------------------------------------------------------------------- #

def compareLesionTests(args, device):
    """
    This function compareLesionTests() compares the post-lesion test set performance of networks
     which were trained with different frequencies of lesions in the training set.
     - this will now search for the lesion assessments for all the model instances that match the args
     - this should now plot a dot +- SEM over model instances at each dot to see how variable it is.
       Note that there will be more variability in the lesioned cases just because when we lesion during
       training we do so randomly with a frequency, which will be different every time.
    """
    plt.figure()
    ax = plt.gca()
    #localpolicy_optimal, globalpolicy_optimal, globaldistpolicy_optimal = plotOptimalReferencePerformance(ax, args)
    localpolicy_optimal, globalpolicy_optimal = plotOptimalReferencePerformance(ax, args)
    frequencylist = [0.0, 0.1, 0.2, 0.3, 0.4]  # training frequencies of different networks to consider
    offsets = [0.01,0.02,0.03]
    overall_lesioned_tests = []

    # file naming
    blcktxt = '_interleaved' if args.all_fullrange else '_temporalblocked'
    contexttxt = '_contextcued' if args.label_context=='true' else '_nocontextcued'
    range_txt = ''
    if args.which_context==0:
        range_txt = ''
    elif args.which_context==1:
        range_txt = '_fullrangeonly'
    elif args.which_context==2:
        range_txt = '_lowrangeonly'
    elif args.which_context==3:
        range_txt = '_highrangeonly'

    for train_lesion_frequency in frequencylist:

        args.train_lesion_freq = train_lesion_frequency
        allmodels = anh.getModelNames(args)
        data = [[] for i in range(len(allmodels))]
        context_tests = np.zeros((const.NCONTEXTS, len(allmodels)))
        perf = np.zeros((const.NCONTEXTS, len(allmodels)))
        counts = np.zeros((const.NCONTEXTS, len(allmodels)))
        unlesioned_test = np.zeros((len(allmodels),))
        lesioned_test = np.zeros((len(allmodels),))

        # find all model ids that fit our requirements
        for ind, m in enumerate(allmodels):
            args.model_id = anh.getIdfromName(m)
            print('modelid: ' + str(args.model_id))
            testParams = anh.setupTestParameters(args, device)
            basefilename = 'network_analysis/lesion_tests/lesiontests'+m[:-4]
            filename = basefilename+'.npy'

            # perform or load the lesion tests
            lesiondata, regulartestdata = anh.performLesionTests(args, testParams, basefilename)
            data[ind] = lesiondata["bigdict_lesionperf"]
            lesioned_test[ind] = lesiondata["lesioned_testaccuracy"]
            unlesioned_test[ind] = regulartestdata["normal_testaccuracy"]

            # evaluate performance on the different contexts
            for seq in range(data[ind].shape[0]):
                for compare_idx in range(data[ind][seq].shape[0]):
                    context = data[ind][seq][compare_idx]["underlying_context"]-1
                    perf[context, ind] += data[ind][seq][compare_idx]["lesion_perf"]
                    counts[context, ind] += 1
            meanperf = 100 * np.divide(perf[:, ind], counts[:, ind])
            for context in range(const.NCONTEXTS):
                print('context {} performance: {}/{} ({:.2f}%)'.format(context+1, perf[context, ind], counts[context, ind], meanperf[context]))
                context_tests[context, ind] = meanperf[context]

        # now determine mean +-sem over models of that lesion frequency
        mean_lesioned_test = np.nanmean(lesioned_test)
        sem_lesioned_test = sp.stats.sem(lesioned_test)

        mean_unlesioned_test = np.nanmean(unlesioned_test)
        sem_unlesioned_test = sp.stats.sem(unlesioned_test)

        mean_contextlesion_test = np.nanmean(context_tests,axis=1)
        sem_contextlesion_test = sp.stats.sem(context_tests,axis=1)

        # plot unlesioned performance
        hnolesion = plt.errorbar(train_lesion_frequency, mean_unlesioned_test, sem_unlesioned_test, color='black', markersize=5, fmt='o')

        # plot lesioned network performance
        #plt.plot(train_lesion_frequency, overall_lesioned_tests, '.', color='black')
        #htotal, = plt.plot(freq, overall_lesioned_tests, color='black')
        hlesion = plt.errorbar(train_lesion_frequency, mean_lesioned_test, sem_lesioned_test, color='grey', markersize=5, fmt='o')
        #plt.text(train_lesion_frequency,lesioned_testaccuracy+1, '{:.2f}%'.format(lesioned_testaccuracy), color='blue')

        # plot post-lesion performance divided up by context
        context_handles = []
        for context in range(const.NCONTEXTS):
            tmp = plt.errorbar(train_lesion_frequency+offsets[context], mean_contextlesion_test[context], sem_contextlesion_test[context], color=contextcolours[context], markersize=4, fmt='o')
            context_handles.append(tmp)
        print('\n')

    plt.xlabel('Lesion frequency during training')
    plt.ylabel('Test perf. post-single lesion')
    ax.set_ylim((50,103))
    plt.legend((localpolicy_optimal, globalpolicy_optimal, hnolesion, hlesion, context_handles[0], context_handles[1], context_handles[2]),('Optimal | local \u03C0, local #distr.','Optimal | global \u03C0, local #distr.','Unlesioned perf. across whole sequence', 'Perf. immediately post-lesion','Perf. immediately post-lesion, context A: 1-16','Perf. immediately post-lesion, context B: 1-11','Perf. immediately post-lesion, context C: 6-16'))
    plt.title('RNN ('+blcktxt[1:]+', '+contexttxt[1:]+', trained with lesions)')
    whichTrialType = 'compare'
    autoSaveFigure('figures/lesionfreq_trainedlesions_', args, True, False, whichTrialType, True)

# ---------------------------------------------------------------------------- #

def perfVdistContextMean(args, device):
    frequencylist = [0.0, 0.1]  # training frequencies of different networks to consider
    overall_lesioned_tests = []

    # file naming
    blcktxt = '_interleaved' if args.all_fullrange else '_temporalblocked'
    contexttxt = '_contextcued' if args.label_context=='true' else '_nocontextcued'
    range_txt = ''
    if args.which_context==0:
        range_txt = ''
    elif args.which_context==1:
        range_txt = '_fullrangeonly'
    elif args.which_context==2:
        range_txt = '_lowrangeonly'
    elif args.which_context==3:
        range_txt = '_highrangeonly'

    for j,train_lesion_frequency in enumerate(frequencylist):
        plt.subplot(1,2,j+1)
        if j==0:
            plt.ylabel('Performance post-lesion')
            plt.title('Trained without lesions')
        if j==1:
            plt.title('Trained with 10% lesions')

        ax = plt.gca()

        args.train_lesion_freq = train_lesion_frequency
        allmodels = anh.getModelNames(args)
        data = [[] for i in range(len(allmodels))]
        global_meanperf = []
        context1_perf, context2_perf, context3_perf = [[] for i in range(3)]
        global_uniquediffs = []
        context1_numberdiffs, context2_numberdiffs, context3_numberdiffs = [[] for i in range(3)]

        # find all model ids that fit our requirements
        for ind, m in enumerate(allmodels):
            args.model_id = anh.getIdfromName(m)
            print('modelid: ' + str(args.model_id))
            testParams = anh.setupTestParameters(args, device)
            basefilename = 'network_analysis/lesion_tests/lesiontests'+m[:-4]
            filename = basefilename+'.npy'

            # perform or load the lesion tests
            lesiondata, regulartestdata = anh.performLesionTests(args, testParams, basefilename)
            data[ind] = lesiondata["bigdict_lesionperf"]
            gp, cp, gd, cd = anh.lesionperfbyNumerosity(data[ind])
            global_meanperf.append(gp)
            global_uniquediffs.append(gd)
            context1_perf.append(cp[0])
            context2_perf.append(cp[1])
            context3_perf.append(cp[2])
            context1_numberdiffs.append(cd[0])
            context2_numberdiffs.append(cd[1])
            context3_numberdiffs.append(cd[2])

        # mean over models
        global_meanperf = np.array(global_meanperf)
        context1_perf = np.array(context1_perf)
        context2_perf = np.array(context2_perf)
        context3_perf = np.array(context3_perf)
        global_uniquediffs = np.array(global_uniquediffs)
        context1_numberdiffs = np.array(context1_numberdiffs)
        context2_numberdiffs = np.array(context2_numberdiffs)
        context3_numberdiffs = np.array(context3_numberdiffs)

        global_meanperf_mean = np.mean(global_meanperf, axis=0)
        global_meanperf_sem = np.std(global_meanperf, axis=0) / np.sqrt(global_meanperf.shape[0])
        global_uniquediffs = np.mean(global_uniquediffs, axis=0)

        context1_perf_mean = np.mean(context1_perf, axis=0)
        context1_perf_sem = np.std(context1_perf, axis=0) / np.sqrt(context1_perf.shape[0])
        context2_perf_mean = np.mean(context2_perf, axis=0)
        context2_perf_sem = np.std(context2_perf, axis=0) / np.sqrt(context2_perf.shape[0])
        context3_perf_mean = np.mean(context3_perf, axis=0)
        context3_perf_sem = np.std(context3_perf, axis=0) / np.sqrt(context3_perf.shape[0])

        context1_numberdiffs = np.mean(context1_numberdiffs, axis=0)
        context2_numberdiffs = np.mean(context2_numberdiffs, axis=0)
        context3_numberdiffs = np.mean(context3_numberdiffs, axis=0)

        # plotting
        global_contextmean = plt.errorbar(global_uniquediffs, global_meanperf_mean, global_meanperf_sem, color='black', fmt='-o',markersize=2) # this is captured already by the separation into contexts and looks jiggly because of different context values on x-axis
        plt.xlabel('|current# - \u03BC|')
        #ax.set_xticks([0, 7.5])
        #ax.set_xticklabels([0, 7.5])

        print(context3_perf_sem)

        # plot and save the figure
        ref7, = plt.plot([0,7.5],[.5,1],linestyle=':',color='grey')
        ref4, = plt.plot([0,5],[.5,1],linestyle=':',color='grey')

        # context-specific performance i.e. how did performance change with dist. to mean in each context
        local_contextmean_context1 = plt.errorbar(context1_numberdiffs, context1_perf_mean, context1_perf_sem, color='gold', fmt='-o',markersize=2)
        local_contextmean_context2 = plt.errorbar(context2_numberdiffs, context2_perf_mean, context2_perf_sem, color='dodgerblue', fmt='-o',markersize=2)
        local_contextmean_context3 = plt.errorbar(context3_numberdiffs, context3_perf_mean, context3_perf_sem, color='orangered', fmt='-o',markersize=2)
    plt.legend((ref7, global_contextmean, local_contextmean_context1, local_contextmean_context2, local_contextmean_context3), ('unity refs, max|\u0394|={5,7.5}', '\u03BC = global median', '\u03BC = local median | context A, 1:16','\u03BC = local median | context B, 1:11','\u03BC = local median | context C, 6-16'))
    whichTrialType = 'compare'
    autoSaveFigure('figures/perf_v_distToContextMean_postlesion_', args, True, False, whichTrialType, True)

# ---------------------------------------------------------------------------- #
