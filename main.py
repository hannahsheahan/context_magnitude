"""
 This is a set of simulations for training a simple RNN on a relative magnitude problem.
 The network will be trained to answer the question: is input N > input N-t?
 Where t is between 3-5, i.e. the inputs to be compared in each sequence are separated by several 'filler' inputs.

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
import plotter as mplt
import analysis_helpers as anh
import constants as const

import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import random
import copy
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.utils import shuffle
from importlib import reload
from mpl_toolkits import mplot3d
from matplotlib import animation
import json
import time
import os

# network stuff
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
#from torch.utils.tensorboard import SummaryWriter
from itertools import product
from datetime import datetime
import argparse

# ---------------------------------------------------------------------------- #

def trainAndSaveANetwork(args):
    """This function will:
    - create a new train/test dataset,
    - train a new RNN according to the hyperparameters in args on that dataset,
    - save the model (and training record) with an auto-generated name based on those args.
    """

    # define the network parameters
    datasetname, trained_modelname, analysis_name, _ = mnet.getDatasetName(args)

    if args.create_new_dataset:
        trainset, testset = dset.createSeparateInputData(datasetname, args)
    else:
        trainset, testset, _, _, _, _ = dset.loadInputData(const.DATASET_DIRECTORY, datasetname)

    # define and train a neural network model, log performance and output trained model
    if args.network_style == 'recurrent':
        model = mnet.trainRecurrentNetwork(args, device, multiparams, trainset, testset)
    else:
        model = mnet.trainMLPNetwork(args, device, multiparams, trainset, testset)

    # save the trained weights so we can easily look at them
    print('Saving trained model...')
    print(trained_modelname)
    torch.save(model, trained_modelname)

# ---------------------------------------------------------------------------- #

def analyseNetwork(args):
    """Perform MDS on:
        - the hidden unit activations for each unique input in each context.
        - the averaged hidden unit activations, averaged across the unique judgement values in each context.
        - the above for both a regular test set and the cross validation set (in case we need it later)
    """
    # load the MDS analysis if we already have it and move on
    datasetname, trained_modelname, analysis_name, _ = mnet.getDatasetName(args) # HRS this needs modifying to choose the right training iter.

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
        trainset, testset, crossvalset, np_trainset, np_testset, np_crossvalset = dset.loadInputData(const.DATASET_DIRECTORY, datasetname)

        # pass each input through the model and determine the hidden unit activations
        setnames = ['test', 'crossval']
        for set in setnames:

            # Assess the network activations on either the regular test set or the cross-validation set
            if set=='test':
                test_loader = DataLoader(testset, batch_size=1, shuffle=False)
            elif set =='crossval':
                test_loader = DataLoader(crossvalset, batch_size=1, shuffle=False)

            for whichTrialType in ['compare', 'filler']:
                activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, time_index, counter, drift, temporal_trialtypes = mnet.getActivations(args, np_testset, trained_model, test_loader, whichTrialType)

                dimKeep = 'judgement'                      # representation of the currently presented number, averaging over previous number
                sl_activations, sl_contexts, sl_MDSlabels, sl_refValues, sl_judgeValues, sl_counter = anh.averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, args.label_context, counter)
                diff_sl_activations, diff_sl_contexts, diff_sl_MDSlabels, diff_sl_refValues, diff_sl_judgeValues, diff_sl_counter, sl_diffValues = anh.diff_averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, args.label_context, counter)

                # do MDS on the activations for the test set
                print('Performing MDS on trials of type: {} in {} set...'.format(whichTrialType, set))
                tic = time.time()

                D = pairwise_distances(activations, metric='correlation') # using correlation distance
                np.fill_diagonal(np.asarray(D), 0)
                MDS_activations, _ = anh.cmdscale(D)

                D = pairwise_distances(sl_activations, metric='correlation') # using correlation distance
                np.fill_diagonal(np.asarray(D), 0)
                MDS_slactivations, _ = anh.cmdscale(D)

                D = pairwise_distances(diff_sl_activations, metric='correlation') # using correlation distance
                np.fill_diagonal(np.asarray(D), 0)
                MDS_diff_slactivations, _ = anh.cmdscale(D)

                toc = time.time()
                print('MDS fitting on trial types {} completed, took (s): {:.2f}'.format(whichTrialType, toc-tic))

                dict = {"MDS_activations":MDS_activations, "activations":activations, "MDSlabels":MDSlabels, "temporal_trialtypes":temporal_trialtypes,\
                            "labels_refValues":labels_refValues, "labels_judgeValues":labels_judgeValues, "drift":drift,\
                            "labels_contexts":labels_contexts, "MDS_slactivations":MDS_slactivations, "sl_activations":sl_activations,\
                            "sl_contexts":sl_contexts, "sl_MDSlabels":sl_MDSlabels, "sl_refValues":sl_refValues, "sl_judgeValues":sl_judgeValues, "sl_counter":sl_counter,\
                            "MDS_diff_slactivations":MDS_diff_slactivations,"diff_sl_activations":diff_sl_activations, "diff_sl_contexts":diff_sl_contexts, "sl_diffValues":sl_diffValues}

                if whichTrialType=='compare':
                    MDS_dict = dict
                else:
                    MDS_dict["filler_dict"] = dict

            # save our activation RDMs for easy access
            np.save(const.RDM_DIRECTORY + 'RDM_'+set+'_compare_'+analysis_name[29:]+'.npy', MDS_dict["sl_activations"])  # the RDM matrix only
            np.save(const.RDM_DIRECTORY + 'RDM_'+set+'_fillers_'+analysis_name[29:]+'.npy', MDS_dict["filler_dict"]["sl_activations"])  # the RDM matrix only
            if set=='test':
                MDS_dict['testset_assessment'] = MDS_dict
            elif set=='crossval':
                MDS_dict['crossval_assessment'] = MDS_dict

        # save the analysis for next time
        print('Saving network analysis...')
        np.save(analysis_name+'.npy', MDS_dict)                    # the full MDS analysis

    return MDS_dict

# ---------------------------------------------------------------------------- #

def generatePlots(MDS_dict, args):
    """ This function just plots stuff and saves the generated figures."""
    saveFig = True
    plot_diff_code = False    # do we want to plot the difference code or the average A activations
    labelNumerosity = True    # numerosity vs outcome labels
    trialTypes = ['compare']  # ['compare', 'filler'] if you want to also see the activations for filler numbers

    for whichTrialType in trialTypes:

        # Label activations by mean number A numerosity
        mplt.activationRDMs(MDS_dict, args, plot_diff_code, whichTrialType)  # activations RSA
        axislimits = (-0.8, 0.8)
        mplt.plot3MDSMean(MDS_dict, args, labelNumerosity, plot_diff_code, whichTrialType, saveFig, 80, axislimits) # mean MDS of our hidden activations (averaged across number B)

        #mplt.plot3MDS(MDS_dict, args, whichTrialType)      # the full MDS cloud, coloured by different labels

        # Label activations by the difference code numerosity
        #plot_diff_code = True
        #mplt.activationRDMs(MDS_dict, args, plot_diff_code, whichTrialType)  # activations RSA
        #mplt.plot3MDSMean(MDS_dict, args, labelNumerosity, plot_diff_code, whichTrialType)

        # Plot checks on the training data sequencing
        #n = plt.hist(activations)   # They are quite sparse activations (but we dont really care that much)
        #mplt.viewTrainingSequence(MDS_dict, args)  # Plot the context sequencing in the training set through time
        #mplt.instanceCounter(MDS_dict, args)  # Check how many samples we have of each unique input (should be context-ordered)

        # MDS with output labels (true/false labels)
        #labelNumerosity = False
        #mplt.plot3MDS(MDS_dict, args, labelNumerosity, plot_diff_code)

        # 3D Animations
        #mplt.animate3DMDS(MDS_dict, args, plot_diff_code)  # plot a 3D version of the MDS constructions

# ---------------------------------------------------------------------------- #
def averageActivationsAcrossModels(args):
    """ This function takes all models trained under the conditions in args, and averages
    the resulting test activations before MDS is performed, and then do MDS on the average activations.
     - Note:  messy but functional.
    """

    allmodels = anh.getModelNames(args)
    MDS_meandict = {}
    MDS_meandict["filler_dict"] = {}
    sl_activations = [[] for i in range(len(allmodels))]
    contextlabel = [[] for i in range(len(allmodels))]
    numberlabel = [[] for i in range(len(allmodels))]
    filler_sl_activations = [[] for i in range(len(allmodels))]
    filler_contextlabel = [[] for i in range(len(allmodels))]
    filler_numberlabel = [[] for i in range(len(allmodels))]

    for ind, m in enumerate(allmodels):
        args.model_id = anh.getIdfromName(m)
        print('Loading model: {}'.format(args.model_id))
        # Analyse the trained network (extract and save network activations)
        mdict = analyseNetwork(args)
        sl_activations[ind] = mdict["sl_activations"]
        contextlabel[ind] = mdict["sl_contexts"]
        numberlabel[ind] = mdict["sl_judgeValues"]
        filler_sl_activations[ind] = mdict["filler_dict"]["sl_activations"]
        filler_contextlabel[ind] = mdict["filler_dict"]["sl_contexts"]
        filler_numberlabel[ind] = mdict["filler_dict"]["sl_judgeValues"]

    MDS_meandict["sl_activations"] = np.mean(sl_activations, axis=0)
    MDS_meandict["sl_contexts"] = np.mean(contextlabel, axis=0)
    MDS_meandict["sl_judgeValues"] = np.mean(numberlabel, axis=0)
    MDS_meandict["filler_dict"]["sl_activations"] = np.mean(filler_sl_activations, axis=0)
    MDS_meandict["filler_dict"]["sl_contexts"] = np.mean(filler_contextlabel, axis=0)
    MDS_meandict["filler_dict"]["sl_judgeValues"] = np.mean(filler_numberlabel, axis=0)

    # Perform MDS on averaged activations for the compare trial data
    pairwise_data = pairwise_distances(MDS_meandict["sl_activations"], metric='correlation') # using correlation distance
    np.fill_diagonal(np.asarray(pairwise_data), 0)
    MDS_act, evals = anh.cmdscale(pairwise_data)

    # Perform MDS on averaged activations for the filler trial data
    pairwise_data = pairwise_distances(MDS_meandict["filler_dict"]["sl_activations"], metric='correlation') # using correlation distance
    np.fill_diagonal(np.asarray(pairwise_data), 0)
    MDS_act_filler, evals = anh.cmdscale(pairwise_data)

    MDS_meandict["MDS_slactivations"] = MDS_act
    MDS_meandict["filler_dict"]["MDS_slactivations"] = MDS_act_filler
    args.model_id = 0

    return MDS_meandict, args

# ---------------------------------------------------------------------------- #

if __name__ == '__main__':

    # set up dataset and network hyperparams via command line
    args, device, multiparams = mnet.defineHyperparams()
    args.label_context = 'true'   # 'true' = context cued explicitly in input; 'constant' = context not cued explicity
    args.all_fullrange = False    # False = blocked; True = interleaved
    args.train_lesion_freq = 0.1  # 0.0 or 0.1  (also 0.2, 0.3, 0.4 for blocked & true context case)

    #args.model_id = 7388  # an example single model case

    # Train a network from scratch and save it
    #trainAndSaveANetwork(args)

    # Analyse the trained network (extract and save network activations)
    #MDS_dict = analyseNetwork(args)

    # Check the average final performance for trained models matching args
    #anh.averagePerformanceAcrossModels(args)

    # Visualise the resultant network activations (RDMs and MDS)
    #MDS_dict, args = averageActivationsAcrossModels(args)
    generatePlots(MDS_dict, args)  # (Figure 3 + extras)

    # Plot the lesion test performance
    #mplt.perfVContextDistance(args, device)     # Assess performance after a lesion vs context distance (Figure 2 and S1)
    #mplt.compareLesionTests(args, device)      # compare the performance across the different lesion frequencies during training (Figure 2)

    # Statistical tests: is network behaviour better fit by an agent using the local-context or global-context policy
    #anh.getSSEForContextModels(args, device)

# ---------------------------------------------------------------------------- #
