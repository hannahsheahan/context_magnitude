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
import analysis_helpers as anh
import constants as const
import lines_model

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
    print('Saving trained model...')
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
                sl_activations, sl_contexts, sl_MDSlabels, sl_refValues, sl_judgeValues, sl_counter = anh.averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, args.label_context, counter)
                diff_sl_activations, diff_sl_contexts, diff_sl_MDSlabels, diff_sl_refValues, diff_sl_judgeValues, diff_sl_counter, sl_diffValues = anh.diff_averageReferenceNumerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, args.label_context, counter)

                # do MDS on the activations for the test set
                print('Performing MDS on trials of type: {} in {} set...'.format(whichTrialType, set))
                tic = time.time()
                randseed = 3 # so that we get the same MDS each time

                embedding = MDS(n_components=3, random_state=randseed, dissimilarity='precomputed')
                D = pairwise_distances(activations, metric='correlation') # using correlation distance
                MDS_activations = embedding.fit_transform(D)
                sl_embedding = MDS(n_components=3, random_state=randseed, dissimilarity='precomputed')
                D = pairwise_distances(sl_activations, metric='correlation') # using correlation distance
                MDS_slactivations = sl_embedding.fit_transform(D)
                diff_sl_embedding = MDS(n_components=3, random_state=randseed, dissimilarity='precomputed')
                D = pairwise_distances(diff_sl_activations, metric='correlation') # using correlation distance
                MDS_diff_slactivations = diff_sl_embedding.fit_transform(D)

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
        axislimits = (-0.8, 0.8)
        MDSplt.plot3MDSMean(MDS_dict, args, labelNumerosity, plot_diff_code, whichTrialType, saveFig, 80, axislimits) # mean MDS of our hidden activations (averaged across number B)

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
def averageActivationsAcrossModels(args):
    # Take all models trained under the conditions in args, and average the resulting test activations before we plot
    # This will average the activations across all model BEFORE MDS is performed, and then do MDS on the average activations
    # *HRS This function is ugly AF so can tidy later

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

    # Now do MDS on the mean activations across all networks
    #randseed = 2#3   # so that we get the same MDS each time
    #sl_embedding = MDS(n_components=3, random_state=randseed, dissimilarity='precomputed')
    #D = pairwise_distances(MDS_meandict["sl_activations"], metric='correlation') # using correlation distance
    #MDS_slactivations = sl_embedding.fit_transform(D)

    #filler_sl_embedding = MDS(n_components=3, random_state=randseed, dissimilarity='precomputed')
    #D = pairwise_distances(MDS_meandict["filler_dict"]["sl_activations"], metric='correlation') # using correlation distance
    #filler_MDS_slactivations = filler_sl_embedding.fit_transform(D)

    # using CMDscale so that it works well with the EEG too
    # fo the compare trial data
    pairwise_data = pairwise_distances(MDS_meandict["sl_activations"], metric='correlation') # using correlation distance
    np.fill_diagonal(np.asarray(pairwise_data), 0)
    MDS_act, evals = lines_model.cmdscale(pairwise_data)

    # for the filler trial data
    pairwise_data = pairwise_distances(MDS_meandict["filler_dict"]["sl_activations"], metric='correlation') # using correlation distance
    np.fill_diagonal(np.asarray(pairwise_data), 0)
    MDS_act_filler, evals = lines_model.cmdscale(pairwise_data)


    MDS_meandict["MDS_slactivations"] = MDS_act
    MDS_meandict["filler_dict"]["MDS_slactivations"] = MDS_act_filler
    args.model_id = 0

    return MDS_meandict, args

# ---------------------------------------------------------------------------- #

def averagePerformanceAcrossModels(args):

    matched_models = anh.getModelNames(args)
    print(matched_models)
    all_training_records = os.listdir(const.TRAININGRECORDS_DIRECTORY)

    train_performance = []
    test_performance = []
    for ind, m in enumerate(matched_models):
        args.model_id = anh.getIdfromName(m)

        for training_record in all_training_records:
            if ('_id'+str(args.model_id)+'.' in training_record):
                if  ('trlf'+str(args.train_lesion_freq) in training_record) and (args.label_context in training_record):
                    print('Found matching model: id{}'.format(args.model_id))
                    # we've found the training record for a model we care about
                    with open(os.path.join(const.TRAININGRECORDS_DIRECTORY, training_record)) as record_file:
                        record = json.load(record_file)
                        train_performance.append(record["trainingPerformance"])
                        test_performance.append(record["testPerformance"])

    train_performance = np.asarray(train_performance)
    test_performance = np.asarray(test_performance)
    n_models = train_performance.shape[0]

    mean_train_performance = np.mean(train_performance, axis=0)
    std_train_performance = np.std(train_performance, axis=0) / np.sqrt(n_models)

    mean_test_performance = np.mean(test_performance, axis=0)
    std_test_performance = np.std(test_performance, axis=0) / np.sqrt(n_models)

    print('Final training performance across {} models: {:.3f} +- {:.3f}'.format(n_models, mean_train_performance[-1], std_train_performance[-1]))  # mean +- std
    print('Final test performance across {} models: {:.3f} +- {:.3f}'.format(n_models, mean_test_performance[-1], std_test_performance[-1]))  # mean +- std
    plt.figure()
    h1 = plt.errorbar(range(11), mean_train_performance, std_train_performance, color='dodgerblue')
    h2 = plt.errorbar(range(11), mean_test_performance, std_test_performance, color='green')
    plt.legend((h1,h2), ['train','test'])

    MDSplt.autoSaveFigure(os.path.join(const.FIGURE_DIRECTORY, '_trainingrecord_'), args, True, False, 'compare', True)

# ---------------------------------------------------------------------------- #

if __name__ == '__main__':

    # set up dataset and network hyperparams via command line
    args, device, multiparams = mnet.defineHyperparams()
    args.label_context = 'true'
    args.all_fullrange = False
    args.train_lesion_freq=0.1

    #args.model_id = 7388  # an example case

    # Train the network from scratch
    #trainAndSaveANetwork(args)

    # Analyse the trained network (extract and save network activations)
    #MDS_dict = analyseNetwork(args)

    # Check the average final performance for trained models matching args
    averagePerformanceAcrossModels(args)

    # Visualise the resultant network activations (RDMs and MDS)
    #MDS_dict, args = averageActivationsAcrossModels(args)
    #generatePlots(MDS_dict, args)

    # Plot the lesion test performance
    #MDSplt.perfVdistContextMean(args, device)     # Assess performance after a lesion vs context distance
    #MDSplt.compareLesionTests(args, device)      # compare the performance across the different lesion frequencies during training

    # Assess whether this class of trained networks use local-context or global-context policy
    #args.train_lesion_freq = 0.1
    #anh.getSSEForContextModels(args, device)

# ---------------------------------------------------------------------------- #
