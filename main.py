"""
 This is a set of simulations for training a simple MLP or RNN on a relational magnitude problem
 The network will be trained to answer the question: is input 2 > input 1?

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
import MDSplotter as MDSplt
import constants as const

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
import time
import os

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

def trainAndSaveANetwork(args):
    # define the network parameters
    datasetname, trained_modelname, analysis_name, _ = mnet.getDatasetName(args)

    if args.create_new_dataset:
        trainset, testset = dset.createSeparateInputData(datasetname, args)
    else:
        trainset, testset, _, _, _, _ = dset.loadInputData(args.fileloc, datasetname)

    # define and train a neural network model, log performance and output trained model
    if args.network_style == 'recurrent':
        model = mnet.trainRecurrentNetwork(args, device, multiparams, trainset, testset)
    else:
        model = mnet.trainMLPNetwork(args, device, multiparams, trainset, testset)

    # save the trained weights so we can easily look at them
    print(trained_modelname)
    torch.save(model, trained_modelname)

# ---------------------------------------------------------------------------- #

def analyseNetwork(args):
    """Perform MDS on:
        - the hidden unit activations for each unique input in each context.
        - the averaged hidden unit activations, averaged across the unique judgement values in each context.
        - the above for both the regular training set and the cross validation set
    """
    # load the MDS analysis if we already have it and move on
    datasetname, trained_modelname, analysis_name, _ = mnet.getDatasetName(args)

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
        trainset, testset, crossvalset, np_trainset, np_testset, np_crossvalset = dset.loadInputData(args.fileloc, datasetname)

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
                sl_activations, sl_contexts, sl_MDSlabels, sl_refValues, sl_judgeValues, sl_counter = MDSplt.averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, args.label_context, counter)
                diff_sl_activations, diff_sl_contexts, diff_sl_MDSlabels, diff_sl_refValues, diff_sl_judgeValues, diff_sl_counter, sl_diffValues = MDSplt.diff_averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, args.label_context, counter)

                # do MDS on the activations for the training set
                print('Performing MDS on trials of type: {} in {} set...'.format(whichTrialType, set))
                tic = time.time()
                randseed = 3 # so that we get the same MDS each time
                embedding = MDS(n_components=3, random_state=randseed)
                MDS_activations = embedding.fit_transform(activations)
                sl_embedding = MDS(n_components=3, random_state=randseed)
                MDS_slactivations = sl_embedding.fit_transform(sl_activations)
                diff_sl_embedding = MDS(n_components=3, random_state=randseed)
                MDS_diff_slactivations = diff_sl_embedding.fit_transform(diff_sl_activations)

                # now do MDS again but for the latent state activations through time in the training set (***HRS takes ages to do this MDS)
                #print(drift["temporal_activation_drift"].shape)
                #embedding = MDS(n_components=3, random_state=randseed)
                #drift["MDS_latentstate"] = embedding.fit_transform(drift["temporal_activation_drift"])
                #print(drift["MDS_latentstate"].shape)

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
            np.save('network_analysis/RDMs/RDM_'+set+'_compare_'+analysis_name[29:]+'.npy', MDS_dict["sl_activations"])  # the RDM matrix only
            np.save('network_analysis/RDMs/RDM_'+set+'_fillers_'+analysis_name[29:]+'.npy', MDS_dict["filler_dict"]["sl_activations"])  # the RDM matrix only
            if set=='test':
                MDS_dict['testset_assessment'] = MDS_dict
            elif set=='crossval':
                MDS_dict['crossval_assessment'] = MDS_dict

        # save the analysis for next time
        print('Saving network analysis...')
        np.save(analysis_name+'.npy', MDS_dict)                                          # the full MDS analysis

    return MDS_dict

# ---------------------------------------------------------------------------- #

def generatePlots(MDS_dict, args):
    # This function just plots stuff and saves the generated figures
    saveFig = True
    plot_diff_code = False    # do we want to plot the difference code or the average A activations
    labelNumerosity = True    # numerosity vs outcome labels
    trialTypes = ['compare', 'filler']

    for whichTrialType in trialTypes:
        # Label activations by mean number A numerosity
        MDSplt.activationRDMs(MDS_dict, args, plot_diff_code, whichTrialType)  # activations RSA
        MDSplt.plot3MDSMean(MDS_dict, args, labelNumerosity, plot_diff_code, whichTrialType) # mean MDS of our hidden activations (averaged across number B)
        #MDSplt.plot3MDS(MDS_dict, args, whichTrialType)      # the full MDS cloud, coloured by different labels

        # Label activations by the difference code numerosity
        #plot_diff_code = True
        #MDSplt.activationRDMs(MDS_dict, args, plot_diff_code, whichTrialType)  # activations RSA
        #MDSplt.plot3MDSMean(MDS_dict, args, labelNumerosity, plot_diff_code, whichTrialType)

        # Plot checks on the training data sequencing
        #n = plt.hist(activations)   # They are quite sparse activations (but we dont really care that much)
        #MDSplt.viewTrainingSequence(MDS_dict, args)  # Plot the context sequencing in the training set through time
        #MDSplt.instanceCounter(MDS_dict, args)  # Check how many samples we have of each unique input (should be context-ordered)

        # MDS with output labels (true/false labels)
        #labelNumerosity = False
        #MDSplt.plot3MDS(MDS_dict, args, labelNumerosity, plot_diff_code)
        #MDSplt.plot3MDSContexts(MDS_dict, labelNumerosity, args)  # plot the MDS with context labels. ***HRS obsolete?

        # 3D Animations
        #MDSplt.animate3DMDS(MDS_dict, args, plot_diff_code)  # plot a 3D version of the MDS constructions
        #MDSplt.animate3DdriftMDS(MDS_dict, args)             # plot a 3D version of the latent state MDS

# ---------------------------------------------------------------------------- #

if __name__ == '__main__':

    # set up dataset and network hyperparams via command line
    args, device, multiparams = mnet.defineHyperparams()
    """
    niters = 1
    lesionfreqs = [0.1, 0.2, 0.3, 0.4]
    for freq in lesionfreqs:
        for i in range(niters):
            args.train_lesion_freq = freq
            args.model_id = random.randint(1,10000)

            # Train the network from scratch
            trainAndSaveANetwork(args)

            # Analyse the trained network
            MDS_dict = analyseNetwork(args)

            # Perform lesion tests on the network
            #blcktxt = '_interleaved' if args.all_fullrange else '_temporalblocked'
            #contexttxt = '_contextcued' if args.label_context=='true' else '_nocontextcued'
            #range_txt = ''
            #testParams = mnet.setupTestParameters(args, device)
            #MDSplt.performLesionTests(args, testParams, 'network_analysis/lesion_tests/lesiontests'+blcktxt+contexttxt+range_txt+'_trainlf'+str(args.train_lesion_freq))

            # Visualise the resultant network activations (RDMs and MDS)
            #generatePlots(MDS_dict, args)
    """

    # test of dataset...bugger
    #trainAndSaveANetwork(args)
    _, _, _, numpy_trainset, numpy_testset, _ = dset.loadInputData('datasets/', 'dataset_truecontextlabel_numrangeblocked_bpl120_id0')
    seq = 5
    trialtype = list(numpy_testset['trialtypeinputs'][seq])
    nums = [dset.turnOneHotToInteger(numpy_testset['judgementValue'][seq][i])[0] for i in range(len(dset.turnOneHotToInteger(numpy_testset['judgementValue'][seq])))]
    prefillers =  [nums[i] for i in range(len(nums)-1) if trialtype[i]==0.0 and trialtype[i+1]==1.0]
    postfillers =  [nums[i] for i in range(2,len(nums)) if trialtype[i]==0.0 and trialtype[i-1]==1.0]  # start at ind 2 to ignore first post-compare trial

    print((trialtype))
    print((nums))
    print(prefillers)
    print(postfillers)
    for i in range(len(postfillers)):
        if postfillers[i]==prefillers[i]:
            print(i)
            print('oh no! this shouldnt happen')

    # Plot the lesion test performance
    #testParams = mnet.setupTestParameters(args, device)
    #MDSplt.perfVdistContextMean(testParams)  # Assess performance after a lesion as a function of the 'seen' number
    #MDSplt.compareLesionTests(args, device)      # compare the performance across the different lesion frequencies during training


# ---------------------------------------------------------------------------- #
