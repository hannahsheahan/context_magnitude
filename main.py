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
    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange, whichContext = params

    if createNewDataset:
        trainset, testset = dset.createSeparateInputData(N, fileloc, datasetname, args.BPTT_len, blockTrain, seqTrain, include_fillers, labelContext, allFullRange, whichContext)
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

def performLesionTests(params, nlesionBins):
    """
    Lesion the network test inputs with different frequencies, and assess performance.
    HRS this is long and ugly, tidy it up.
    """

    X = nlesionBins
    freq = np.linspace(0,1,X)
    context_tests = np.zeros((X,3))
    networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange, whichContext = params

    if whichContext==0:  # proceed as normal
        nmodels = 1
    else:
        nmodels = 3      # load in models trained on just a single context and compare them
        whichContexttxt = '_contextmodelstrainedseparately'
        context_handles = []

    # file naming
    blcktxt = '_interleaved' if allFullRange else '_temporalblocked'
    contexttxt = '_contextcued' if labelContext=='true' else '_nocontextcued'

    plt.figure(figsize=(5,6))
    colours = ['gold', 'dodgerblue', 'orangered', 'black']
    offsets = [0.17,0.14,0.11]

    # plot baseline metrics
    ax = plt.gca()
    if whichContext==0:
        dslope, = plt.plot([0,1],[100,50],'--',color='grey')  # the theoretical performance line if all that mattered was the number of nonlesioned trials
        localpolicy_optimal = ax.axhline(y=77.41, linestyle=':', color='lightpink')
        globalpolicy_optimal = ax.axhline(y=72.58, linestyle=':', color='lightblue')
        globaldistpolicy_optimal = ax.axhline(y=76.5, linestyle=':', color='lightgreen')
    else:
        oldbenchmark1 = ax.axhline(y=77.41, linestyle=':', color='grey')
        oldbenchmark2 = ax.axhline(y=72.58, linestyle=':', color='grey')
        oldbenchmark3 = ax.axhline(y=76.5, linestyle=':', color='grey')

        contextA_localpolicy = ax.axhline(y=76.67, color='gold')
        contextBC_localpolicy = ax.axhline(y=77.78, color='orangered')


    for whichmodel in range(nmodels):
        lesioned_tests = []
        overall_lesioned_tests = []
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
        basefilename = 'network_analysis/lesion_tests/lesiontests'+blcktxt+contexttxt+range_txt
        regularfilename = basefilename + '_regular.npy'

        params = [networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange, whichContext]
        testParams = mnet.setupTestParameters(fileloc, params)
        args, trained_model, device, testloader, criterion, retainHiddenState, printOutput = testParams
        whichLesion = 'number'

        for i in range(X):

            lesionFrequency =  freq[i]
            filename = basefilename+str(lesionFrequency)+'.npy'
            try:
                lesiondata = (np.load(filename, allow_pickle=True)).item()
                print('Loaded existing lesion analysis: ('+blcktxt[1:]+', '+contexttxt[1:]+', frequency: '+str(lesionFrequency)+')')
                lesioned_testaccuracy = lesiondata["lesioned_testaccuracy"]
                overall_lesioned_testaccuracy = lesiondata["overall_lesioned_testaccuracy"]
                print('{}-lesioned network, test performance: {:.2f}%'.format(whichLesion, lesioned_testaccuracy))

            except:
                # evaluate network at test with lesions
                print('Performing lesion tests...')
                tic = time.time()
                bigdict_lesionperf, lesioned_testaccuracy, overall_lesioned_testaccuracy = mnet.recurrent_lesion_test(*testParams, whichLesion, lesionFrequency)
                #bigdict_lesionperf, lesioned_testaccuracy, overall_lesioned_testaccuracy = mnet.recurrent_simplelesion_test(*testParams, whichLesion, lesionFrequency)
                #lesionHowMany = 'one'
                #bigdict_lesionperf, lesioned_testaccuracy, overall_lesioned_testaccuracy = mnet.recurrent_mostsimplelesion_test(*testParams, whichLesion, lesionHowMany)
                print('{}-lesioned network, test performance: {:.2f}%'.format(whichLesion, lesioned_testaccuracy))
                toc = time.time()
                print('Lesion evaluation took {:.1f} sec'.format(toc-tic))

                # save lesion analysis for next time
                lesiondata = {"bigdict_lesionperf":bigdict_lesionperf}
                lesiondata["lesioned_testaccuracy"] = lesioned_testaccuracy
                lesiondata["overall_lesioned_testaccuracy"] = overall_lesioned_testaccuracy
                np.save(filename, lesiondata)

            lesioned_tests.append(lesioned_testaccuracy)
            overall_lesioned_tests.append(overall_lesioned_testaccuracy)
            data = lesiondata["bigdict_lesionperf"]

            if whichContext==0:
                # let's also split it up to look at performance based on the different contexts
                perf = np.zeros((3,))
                counts = np.zeros((3,))
                for seq in range(data.shape[0]):
                    for compare_idx in range(data[i].shape[0]):
                        context = data[seq][compare_idx]["underlying_context"]-1
                        perf[context] += data[seq][compare_idx]["lesion_perf"]
                        counts[context] += 1
                meanperf = 100 * np.divide(perf, counts)
                for context in range(3):
                    print('context {} performance: {}/{} ({:.2f}%)'.format(context+1, perf[context], counts[context], meanperf[context]))
                    context_tests[i][context] = meanperf[context]
        # and evaluate the unlesioned performance as a benchmark
        try:
            regulartestdata = (np.load(regularfilename, allow_pickle=True)).item()
            print('Loaded regular test performance...')
            print(regulartestdata)
            normal_testaccuracy = regulartestdata["normal_testaccuracy"]
        except:
            print('Evaluating regular network test performance...')
            _, normal_testaccuracy = mnet.recurrent_test(*testParams)
            regulartestdata = {"normal_testaccuracy":normal_testaccuracy}
            np.save(regularfilename, regulartestdata)
        print('Regular network, test performance: {:.2f}%'.format(normal_testaccuracy))

        # unlesioned performance
        hnolesion, = plt.plot(0, normal_testaccuracy, 'x', color=colours[colourindex])

        if whichContext==0:
            # lesioned network performance
            plt.plot(freq, overall_lesioned_tests, '.', color='black')
            htotal, = plt.plot(freq, overall_lesioned_tests, color='black')
            hlesion, = plt.plot(freq, lesioned_tests, '.', color='blue', markersize=12 )
            plt.text(0,lesioned_tests[0]+1, '{:.2f}%'.format(lesioned_tests[0]), color='blue')

            # plot the performance divided up by context too
            context_handles = []
            for context in range(3):
                x = [freq[i]-offsets[context] for i in range(len(freq))]
                y = [context_tests[i][context] for i in range(context_tests.shape[0])]
                tmp, = plt.plot(x, y, '.', color=colours[context], markersize=8)
                context_handles.append(tmp)
        else:
            x = [freq[i]-offsets[whichmodel] for i in range(len(freq))]
            tmp, = plt.plot(x, lesioned_tests, '.', color=colours[colourindex], markersize=8)
            context_handles.append(tmp)
            yoffsets = [80, 85, 90]
            plt.text(0,yoffsets[whichmodel], '{:.2f}%'.format(lesioned_tests[0]), color=colours[colourindex])


    plt.xlabel('Compare trials lesioned immediately prior to assessment')
    plt.ylabel('Perf. post-lesion trial')
    if whichContext==0:
        plt.legend((localpolicy_optimal, globalpolicy_optimal, globaldistpolicy_optimal, hnolesion, htotal, hlesion, context_handles[0], context_handles[1], context_handles[2], dslope),('Optimal | local \u03C0, local #distr.','Optimal | global \u03C0, local #distr.','Optimal | global \u03C0, global #distr.','Unlesioned, perf. across sequence', 'Lesioned, perf. across sequence', 'Lesioned, perf. immediately post-lesion','" "    context A: 1-15','" "    context B: 1-10','" "    context C: 6-15', '-unity slope ref'))
    else:
        plt.legend((oldbenchmark1, contextA_localpolicy, contextBC_localpolicy, hnolesion, context_handles[0], context_handles[1], context_handles[2]),('previous benchmarks','optimal | context A','optimal | context B or C','Unlesioned, perf. across sequence','Lesioned, perf. post-lesion; context A: 1-15','" "    context B: 1-10','" "    context C: 6-15'))
    ax.set_ylim((10,103))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['one','all (in sequence)'])

    #plt.savefig('lesionFrequencyTest_numberlesion_constantcontext.pdf',bbox_inches='tight')
    plt.title('RNN ('+blcktxt[1:]+', '+contexttxt[1:]+whichContexttxt+')')
    whichTrialType = 'compare'
    MDSplt.autoSaveFigure('figures/lesionfreq_vs_testperf_simple_'+whichContexttxt, args, networkStyle, blockTrain, seqTrain, True, labelContext, False, noise_std, retainHiddenState, False, whichTrialType, allFullRange, 0, True)


# ---------------------------------------------------------------------------- #

if __name__ == '__main__':

    # dataset parameters
    createNewDataset = False          # re-generate the random train/test dataset each time?
    include_fillers = True           # True: task is like Fabrice's with filler trials; False: solely compare trials
    fileloc = 'datasets/'
    N = 15                           # global: max numerosity for creating one-hot vectors. HRS to turn local, this wont be changed.
    whichContext = 1                # 0: default, uses all 3 contexts in dataset i.e. all ranges. 1-3: just a single context, 1: 1-15; 2: 1-10; 3: 6-15.
    allFullRange = False             # default: False. True: to randomise the context range on each trial (but preserve things like that current compare trial != prev compare trial, and filler spacing)
    blockTrain = True                # whether to block the training by context
    seqTrain = True                  # whether there is sequential structure linking inputs A and B i.e. if at trial t+1 input B (ref) == input A from trial t
    labelContext = 'true'            # 'true', 'random', 'constant', does the input contain true markers of context (1-3), random ones (still 1-3), or constant (1)?
    retainHiddenState = True         # initialise the hidden state for each pair as the hidden state of the previous pair
    if not blockTrain:
        seqTrain = False             # cant have sequential AB training structure if contexts are intermingled. HRS to deprecate seqTrain, this will always be true.
    if whichContext>0:
        allFullRange = False         # cant intermingle over context ranges if you only have one context range. This sorts out filenames

    # which model we want to look at
    networkStyle = 'recurrent'       # 'recurrent' or 'mlp'. MLP now  unused, hasnt been tested for several updates.
    noise_std = 0.0                  # default: 0.0. Can be manipulated to inject iid noise into the recurrent hiden state between numerical inputs.
    params = [networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange, whichContext]

    # Train the network from scratch
    #trainAndSaveANetwork(params, createNewDataset, include_fillers)

    # Perform lesion tests on the network
    #nlesionBins = 2
    #performLesionTests(params, nlesionBins)

    # Assess performance after a lesion as a function of the 'seen' number
    testParams = mnet.setupTestParameters(fileloc, params)
    MDSplt.perfVdistContextMean(params, testParams)

    # Analyse the trained network
    #args, _, _ = mnet.defineHyperparams() # network training hyperparams
    #MDS_dict = analyseNetwork(fileloc, args, params)

    # Visualise the resultant network activations (RDMs and MDS)
    #generatePlots(MDS_dict, args, params)

# ---------------------------------------------------------------------------- #
