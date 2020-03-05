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

def trainAndSaveANetwork(params, createNewDataset, include_fillers):
    # define the network parameters
    args, device, multiparams = mnet.defineHyperparams() # training hyperparams for network (passed as args when called from command line)
    datasetname, trained_modelname, analysis_name, _ = mnet.getDatasetName(args, *params)
    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange = params

    if createNewDataset:
        trainset, testset = dset.createSeparateInputData(N, fileloc, datasetname, args.BPTT_len, blockTrain, seqTrain, include_fillers, labelContext, allFullRange)
        #trainset, testset = dset.createSeparateInputData(N, fileloc, datasetname, blockTrain, seqTrain, labelContext)
    else:
        trainset, testset, _, _ = dset.loadInputData(fileloc, datasetname)

    # define and train a neural network model, log performance and output trained model
    if networkStyle == 'recurrent':
        model = mnet.trainRecurrentNetwork(args, device, multiparams, trainset, testset, N, params)
    else:
        model = mnet.trainMLPNetwork(args, device, multiparams, trainset, testset, N, params)

    # save the trained weights so we can easily look at them
    print(trained_modelname)
    torch.save(model, trained_modelname)

# ---------------------------------------------------------------------------- #

def analyseNetwork(fileloc, args, params):
    """Perform MDS on:
        - the hidden unit activations (60-dim) for each unique input in each context.
        - the averaged hidden unit activations (60-dim), averaged across the unique judgement values in each context.
        - the recurrent latent states (33-dim), as they evolve across the 12k sequential trials.
    """
    # load the MDS analysis if we already have it and move on
    datasetname, trained_modelname, analysis_name, _ = mnet.getDatasetName(args, *params)

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
        trainset, testset, np_trainset, np_testset = dset.loadInputData(fileloc, datasetname)
        networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange = params

        # pass each input through the model and determine the hidden unit activations
        #if (networkStyle=='recurrent') and retainHiddenState: # pass the whole sequence of trials for the recurrent state
        train_loader = DataLoader(trainset, batch_size=1, shuffle=False)
        for whichTrialType in ['compare', 'filler']:

            activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, time_index, counter, drift, temporal_trialtypes = mnet.getActivations(np_trainset, trained_model, networkStyle, retainHiddenState, train_loader, whichTrialType)
            dimKeep = 'judgement'                      # representation of the currently presented number, averaging over previous number
            sl_activations, sl_contexts, sl_MDSlabels, sl_refValues, sl_judgeValues, sl_counter = MDSplt.averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, labelContext, counter)

            # How bout if we average activations over the difference values!
            diff_sl_activations, diff_sl_contexts, diff_sl_MDSlabels, diff_sl_refValues, diff_sl_judgeValues, diff_sl_counter, sl_diffValues = MDSplt.diff_averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, labelContext, counter)

            # do MDS on the activations for the training set
            print('Performing MDS on trials of type: {}...'.format(whichTrialType))
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

        # save the analysis for next time
        print('Saving network analysis...')
        np.save(analysis_name+'.npy', MDS_dict)                                          # the full MDS analysis
        np.save('network_analysis/RDMs/RDM_compare_'+analysis_name[29:]+'.npy', MDS_dict["sl_activations"])  # the RDM matrix only
        np.save('network_analysis/RDMs/RDM_fillers_'+analysis_name[29:]+'.npy', MDS_dict["filler_dict"]["sl_activations"])  # the RDM matrix only


    return MDS_dict

# ---------------------------------------------------------------------------- #

def generatePlots(MDS_dict, args, params):
    # This function just plots stuff and saves the generated figures
    saveFig = True
    plot_diff_code = False    # do we want to plot the difference code or the average A activations
    labelNumerosity = True    # numerosity vs outcome labels
    params.append(saveFig)
    trialTypes = ['compare', 'filler']

    for whichTrialType in trialTypes:
        # Label activations by mean number A numerosity
        MDSplt.activationRDMs(MDS_dict, args, params, plot_diff_code, whichTrialType)  # activations RSA
        MDSplt.plot3MDSMean(MDS_dict, args, params, labelNumerosity, plot_diff_code, whichTrialType) # mean MDS of our hidden activations (averaged across number B)
        MDSplt.plot3MDS(MDS_dict, args, params, whichTrialType)      # the full MDS cloud, coloured by different labels

        # Label activations by the difference code numerosity
        #plot_diff_code = True
        #MDSplt.activationRDMs(MDS_dict, args, params, plot_diff_code, whichTrialType)  # activations RSA
        #MDSplt.plot3MDSMean(MDS_dict, args, params, labelNumerosity, plot_diff_code, whichTrialType)

        # Plot checks on the training data sequencing
        #n = plt.hist(activations)   # They are quite sparse activations (but we dont really care that much)
        #MDSplt.viewTrainingSequence(MDS_dict, args, params)  # Plot the context sequencing in the training set through time
        #MDSplt.instanceCounter(MDS_dict, args, params)  # Check how many samples we have of each unique input (should be context-ordered)

        # MDS with output labels (true/false labels)
        #labelNumerosity = False
        #MDSplt.plot3MDS(MDS_dict, args, params, labelNumerosity, plot_diff_code)
        #MDSplt.plot3MDSContexts(MDS_dict, labelNumerosity, args, params)  # plot the MDS with context labels. ***HRS obsolete?

        # 3D Animations
        #MDSplt.animate3DMDS(MDS_dict, args, params, plot_diff_code)  # plot a 3D version of the MDS constructions
        #MDSplt.animate3DdriftMDS(MDS_dict, args, params)             # plot a 3D version of the latent state MDS

# ---------------------------------------------------------------------------- #

def testTrainedNetwork(args, trained_model, device, testloader, criterion, retainHiddenState, printOutput):
    """
    Test the standard trained network with no lesions.
    """
    # evalate lesioned and regular test performance
    normal_testloss, normal_testaccuracy = mnet.recurrent_test(args, trained_model, device, testloader, criterion, retainHiddenState, printOutput)
    print('Regular network, test performance: {:.2f}%'.format(normal_testaccuracy))

# ---------------------------------------------------------------------------- #

def setupTestParameters(params):
    """
    Evaluate a network on the test set with either the context or the numerical input stream lesioned (set to zero).
    Compare to performance when the network is evaluated as normal on the test set.
    """
    args, device, multiparams = mnet.defineHyperparams() # training hyperparams for network (passed as args when called from command line)
    datasetname, trained_modelname, analysis_name, _ = mnet.getDatasetName(args, *params)
    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange = params

    # load the test set appropriate for the dataset our model was trained on
    trainset, testset, _, _ = dset.loadInputData(fileloc, datasetname)
    testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

    # load our trained model
    trained_model = torch.load(trained_modelname)
    criterion = nn.BCELoss() #nn.CrossEntropyLoss()   # binary cross entropy loss
    printOutput = True

    testParams = [args, trained_model, device, testloader, criterion, retainHiddenState, printOutput]

    return testParams

# ---------------------------------------------------------------------------- #

def performLesionTests(params, lesionBins):
    """
    Lesion the network test inputs with different frequencies, and assess performance.
    """

    X = lesionBins
    freq = np.linspace(0,1,X)
    lesioned_tests = []
    testParams = setupTestParameters(params)
    whichLesion = 'number'

    for i in range(X):
        if i==0:
            # evaluate regular test performance
            _, normal_testaccuracy = mnet.recurrent_test(*testParams)
            print('Regular network, test performance: {:.2f}%'.format(normal_testaccuracy))
            lesioned_tests.append(normal_testaccuracy)
        else:
            # evaluate network at test with lesions
            lesionFrequency =  freq[i] # fraction of compare trials to lesion (0-1)
            _, lesioned_testaccuracy = mnet.recurrent_lesion_test(*testParams, whichLesion, lesionFrequency)
            print('{}-lesioned network, test performance: {:.2f}%'.format(whichLesion, lesioned_testaccuracy))
            lesioned_tests.append(lesioned_testaccuracy)

    plt.figure()
    plt.plot(freq, lesioned_tests, '.', color='blue' )
    plt.plot(freq, lesioned_tests, color='blue')
    plt.xlabel('Lesion frequency (0-1)')
    plt.ylabel('Perf. post-lesion trial')

    #plt.savefig('lesionFrequencyTest_numberlesion_constantcontext.pdf',bbox_inches='tight')
    plt.title('RNN w/ no context label, BPTT120: lesion tests')
    MDSplt.autoSaveFigure('figures/lesionfreq_vs_testperf_', args, networkStyle, blockTrain, seqTrain, labelNumerosity, givenContext, labelContexts, noise_std, retainHiddenState, plot_diff_code, whichTrialType, allFullRange, saveFig)

# ---------------------------------------------------------------------------- #

if __name__ == '__main__':

    # dataset parameters
    createNewDataset = True          # re-generate the random train/test dataset each time?
    include_fillers = True           # True: task is like Fabrice's with filler trials; False: solely compare trials
    fileloc = 'datasets/'
    N = 15                           # global: max numerosity for creating one-hot vectors. HRS to turn local, this wont be changed.
    allFullRange = True             # default: False. True: to randomise the context range on each trial (but preserve things like that current compare trial != prev compare trial, and filler spacing)
    blockTrain = True                # whether to block the training by context
    seqTrain = True                  # whether there is sequential structure linking inputs A and B i.e. if at trial t+1 input B (ref) == input A from trial t
    labelContext = 'true'            # 'true', 'random', 'constant', does the input contain true markers of context (1-3), random ones (still 1-3), or constant (1)?
    retainHiddenState = True         # initialise the hidden state for each pair as the hidden state of the previous pair
    if not blockTrain:
        seqTrain = False              # cant have sequential AB training structure if contexts are intermingled. HRS to deprecate seqTrain, this will always be true.

    # which model we want to look at
    networkStyle = 'recurrent'       # 'recurrent' or 'mlp'. MLP now  unused, hasnt been tested for several updates.
    noise_std = 0.0                  # default: 0.0. Can be manipulated to inject iid noise into the recurrent hiden state between numerical inputs.
    params = [networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange]

    # Train the network from scratch
    trainAndSaveANetwork(params, createNewDataset, include_fillers)

    # Perform lesion tests on the network
    #lesionBins = 10
    #performLesionTests(params, lesionBins)

    # Analyse the trained network
    args, _, _ = mnet.defineHyperparams() # network training hyperparams
    MDS_dict = analyseNetwork(fileloc, args, params)

    # Visualise the resultant network activations (RDMs and MDS)
    generatePlots(MDS_dict, args, params)

# ---------------------------------------------------------------------------- #
