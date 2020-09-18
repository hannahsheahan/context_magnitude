# Miscellaneous analysis functions to go in here (some transferred from plotter.py)
# Author: Hannah Sheahan
# Date: 08/04/2020
# Issues: N/A
# Notes: N/A
# ---------------------------------------------------------------------------- #

import define_dataset as dset
import magnitude_network as mnet
import constants as const
import numpy as np
import scipy
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
import copy
import torch
from sklearn.linear_model import LogisticRegression


def get_model_names(args):
    """This function finds and return all the trained model file names that meet the criteria in args
    (ignoring model id).
    """
    # included factors in name from get_dataset_name()  (excluding random id for model instance)
    str_args = '_bs'+ str(args.batch_size_multi[0]) + '_lr' + str(args.lr_multi[0]) + '_ep' + str(args.epochs) + '_r' + str(args.recurrent_size) + '_h' + str(args.hidden_size) + '_bpl' + str(args.BPTT_len) + '_trlf' + str(args.train_lesion_freq)
    networkTxt = 'RNN' if args.network_style == 'recurrent' else 'MLP'
    contextlabelledtext = '_'+args.label_context+'contextlabel'
    hiddenstate = '_retainstate' if args.retain_hidden_state else '_resetstate'
    rangetxt = '_numrangeintermingled' if args.all_fullrange else '_numrangeblocked'

    # get all model files and then subselect the ones we want
    allfiles = os.listdir("models")
    files = []

    for file in allfiles:
        # check we've got the basics
        if ((rangetxt in file) and (contextlabelledtext in file)) and ((hiddenstate in file) and (networkTxt in file)):
            if str_args in file:
                files.append(file)
    return files


def get_id_from_name(modelname):
    """Take the model name and extract the model id number from the string.
    This is useful when looping through all saved models, you can assign the args.model_id param
    to this number so that subsequent analysis and figure generation naming includes the appropriate the model-id #.
    """
    id_ind = modelname.find('_id')+3
    pth_ind = modelname.find('.pth')
    return  modelname[id_ind:pth_ind]


def average_ref_numerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, givenContext, counter):
    """This function will average the hidden unit activations over one of the two numbers involved in the representation:
    either the reference or the judgement number. This is so that we can then compare to Fabrice's plots
     which are averaged over the previously presented number (input B).
    Prior to performing the MDS we want to know whether to flatten over a particular value
    i.e. if plotting for reference value, flatten over the judgement value and vice versa.
     - dimKeep = 'reference' or 'judgement'
    """
    # initializing
    uniqueValues = [int(np.unique(labels_judgeValues)[i]) for i in range(len(np.unique(labels_judgeValues)))]
    flat_activations = np.zeros((const.NCONTEXTS,len(uniqueValues),activations.shape[1]))
    flat_values = np.zeros((const.NCONTEXTS,len(uniqueValues),1))
    flat_outcomes = np.empty((const.NCONTEXTS,len(uniqueValues),1))
    flat_contexts = np.empty((const.NCONTEXTS,len(uniqueValues),1))
    flat_counter = np.zeros((const.NCONTEXTS,len(uniqueValues),1))
    divisor = np.zeros((const.NCONTEXTS,len(uniqueValues)))

    # which label to flatten over (we keep whichever dimension is dimKeep, and average over the other)
    if dimKeep == 'reference':
        flattenValues = labels_refValues
    else:
        flattenValues = labels_judgeValues

    # pick out all the activations that meet this condition for each context and then average over them
    for context in range(const.NCONTEXTS):
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
    flat_activations, flat_contexts, flat_values, flat_outcomes, flat_counter = [dset.flatten_first_dim(i) for i in [flat_activations, flat_contexts, flat_values, flat_outcomes, flat_counter]]
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


def diff_average_ref_numerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, givenContext, counter):
    """  This is a messy variant of average_ref_numerosity(), which averages over numbers which have the same difference (A-B).
    """
    uniqueValues = [i for i in range(-const.FULLR_SPAN+1,const.FULLR_SPAN-1)] # hacked for now
    #uniqueValues = [int(np.unique(labels_judgeValues)[i]) for i in range(len(np.unique(labels_judgeValues)))]
    flat_activations = np.zeros((const.NCONTEXTS,len(uniqueValues),activations.shape[1]))
    flat_values = np.zeros((const.NCONTEXTS,len(uniqueValues),1))
    flat_outcomes = np.empty((const.NCONTEXTS,len(uniqueValues),1))
    flat_contexts = np.empty((const.NCONTEXTS,len(uniqueValues),1))
    flat_counter = np.zeros((const.NCONTEXTS,len(uniqueValues),1))
    divisor = np.zeros((const.NCONTEXTS,len(uniqueValues)))

    # which label to flatten over (we keep whichever dimension is dimKeep, and average over the other)
    flattenValues = [labels_judgeValues[i] - labels_refValues[i] for i in range(len(labels_refValues))]

    # pick out all the activations that meet this condition for each context and then average over them
    for context in range(const.NCONTEXTS):
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
    flat_activations, flat_contexts, flat_values, flat_outcomes, flat_counter = [dset.flatten_first_dim(i) for i in [flat_activations, flat_contexts, flat_values, flat_outcomes, flat_counter]]
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


def performance_mean(number_differences, performance):
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


def perform_lesion_tests(args, testParams, basefilename):
    """
    This function perform_lesion_tests() performs lesion tests on a single network
    We will only consider performance after a single lesion, because the other metrics are boring sanity checks.
    """
    # lesion settings
    whichLesion = 'number'    # default: 'number'. That's all we care about really

    # file naming
    blcktxt = '_interleaved' if args.all_fullrange else '_temporalblocked'
    contexttxt = '_contextcued' if args.label_context=='true' else '_nocontextcued'
    regularfilename = basefilename + '_regular.npy'
    filename = basefilename+'.npy'

    # perform and save the lesion tests
    try:
        lesiondata = (np.load(filename, allow_pickle=True)).item()
    except:
        # evaluate network at test with lesions
        print('Performing lesion tests...')
        bigdict_lesionperf, lesioned_testaccuracy, overall_lesioned_testaccuracy = mnet.recurrent_lesion_test(*testParams, whichLesion, 0.0)
        print('{}-lesioned network, test performance: {:.2f}%'.format(whichLesion, lesioned_testaccuracy))

        # save lesion analysis for next time
        lesiondata = {"bigdict_lesionperf":bigdict_lesionperf}
        lesiondata["lesioned_testaccuracy"] = lesioned_testaccuracy
        lesiondata["overall_lesioned_testaccuracy"] = overall_lesioned_testaccuracy
        np.save(filename, lesiondata)

    # Evaluate the unlesioned performance as a benchmark
    try:
        regulartestdata = (np.load(regularfilename, allow_pickle=True)).item()
        normal_testaccuracy = regulartestdata["normal_testaccuracy"]
    except:
        print('Evaluating regular network test performance...')
        _, normal_testaccuracy = mnet.recurrent_test(*testParams)
        regulartestdata = {"normal_testaccuracy":normal_testaccuracy}
        np.save(regularfilename, regulartestdata)
    #print('Regular network, test performance: {:.2f}%'.format(normal_testaccuracy))

    return lesiondata, regulartestdata


def lesion_perf_by_numerosity(lesiondata):
    """This function determines how a given model performs post lesion on different numbers and contexts.
    """
    context_perf = [[] for i in range(const.NCONTEXTS)]
    context_numberdiffs = [[] for i in range(const.NCONTEXTS)]
    context_globalnumberdiff = [[] for i in range(const.NCONTEXTS)]

    # evaluate the context mean for each network assessment
    contextmean = np.zeros((lesiondata.shape[0],lesiondata.shape[1]))
    numberdiffs = np.zeros((lesiondata.shape[0],lesiondata.shape[1]))
    globalnumberdiffs = np.zeros((lesiondata.shape[0],lesiondata.shape[1]))
    perf = np.zeros((lesiondata.shape[0],lesiondata.shape[1]))
    globalmean = const.GLOBAL_MEAN

    for seq in range(lesiondata.shape[0]):
        for compare_idx in range(lesiondata.shape[1]):
            context = lesiondata[seq][compare_idx]["underlying_context"]
            if context==1:
                contextmean[seq][compare_idx] = const.CONTEXT_FULL_MEAN
            elif context==2:
                contextmean[seq][compare_idx] = const.CONTEXT_LOW_MEAN
            elif context==3:
                contextmean[seq][compare_idx] = const.CONTEXT_HIGH_MEAN

            # calculate difference between current number and context or global mean
            numberdiffs[seq][compare_idx] = np.abs(np.asarray(lesiondata[seq][compare_idx]["assess_number"]-contextmean[seq][compare_idx]))
            globalnumberdiffs[seq][compare_idx] = np.abs(np.asarray(lesiondata[seq][compare_idx]["assess_number"]-globalmean))
            perf[seq][compare_idx] = lesiondata[seq][compare_idx]["lesion_perf"]

            # context-specific
            context_perf[context-1].append(perf[seq][compare_idx])
            context_numberdiffs[context-1].append(numberdiffs[seq][compare_idx])
            context_globalnumberdiff[context-1].append(globalnumberdiffs[seq][compare_idx])

    # flatten across sequences and the trials in those sequences
    globalnumberdiffs = dset.flatten_first_dim(globalnumberdiffs)
    numberdiffs = dset.flatten_first_dim(numberdiffs)
    perf = dset.flatten_first_dim(perf)
    meanperf, uniquediffs = performance_mean(numberdiffs, perf)
    global_meanperf, global_uniquediffs = performance_mean(globalnumberdiffs, perf)

    # assess mean performance under each context
    context1_meanperf, context1_uniquediffs = performance_mean(context_numberdiffs[0], context_perf[0])
    context2_meanperf, context2_uniquediffs = performance_mean(context_numberdiffs[1], context_perf[1])
    context3_meanperf, context3_uniquediffs = performance_mean(context_numberdiffs[2], context_perf[2])
    context_perf = [context1_meanperf, context2_meanperf, context3_meanperf]
    context_numberdiffs = [context1_uniquediffs, context2_uniquediffs, context3_uniquediffs]

    return global_meanperf, context_perf, global_uniquediffs, context_numberdiffs


def model_behaviour_vs_theory(args, device):
    """This function determines the sum squared error between the rnn responses and the local vs global context models, for each RNN instance.
    """
    allmodels = get_model_names(args)
    SSE_local = [0 for i in range(len(allmodels))]
    SSE_global = [0 for i in range(len(allmodels))]

    for ind, m in enumerate(allmodels):
        args.model_id = get_id_from_name(m)
        testParams = mnet.setup_test_parameters(args, device)
        basefilename = const.LESIONS_DIRECTORY + 'lesiontests'+m[:-4]

        # perform or load the lesion tests
        lesiondata, regulartestdata = perform_lesion_tests(args, testParams, basefilename)
        n_sequences, n_lesions = lesiondata["bigdict_lesionperf"].shape

        for seq in range(n_sequences):
            for lesion in range(n_lesions):
                localmodel_perf = lesiondata["bigdict_lesionperf"][seq][lesion]["localmodel_perf"]
                globalmodel_perf = lesiondata["bigdict_lesionperf"][seq][lesion]["globalmodel_perf"]
                RNN_perf = lesiondata["bigdict_lesionperf"][seq][lesion]["lesion_perf"]
                SSE_local[ind] += (RNN_perf - localmodel_perf)**2
                SSE_global[ind] += (RNN_perf - globalmodel_perf)**2

    # Now compare the arrays of SSE for each deterministic model across the RNN instances
    Tstat, pvalue = scipy.stats.ttest_rel(SSE_local, SSE_global)
    print('Comparing local vs global context models:')
    print('local model, SSE: {}'.format(SSE_local))
    print('global model, SSE: {}'.format(SSE_global))
    print('Tstat: {}  p-value: {}'.format(Tstat, pvalue))


def average_perf_across_models(args):
    """Take the training records and determine the average train and test performance
    across all trained models that meet the conditions specified in args.
    """
    matched_models = get_model_names(args)
    all_training_records = os.listdir(const.TRAININGRECORDS_DIRECTORY)
    record_name = ''
    train_performance = []
    test_performance = []
    for ind, m in enumerate(matched_models):
        args.model_id = get_id_from_name(m)

        for training_record in all_training_records:
            if ('_id'+str(args.model_id)+'.' in training_record):
                if  ('trlf'+str(args.train_lesion_freq) in training_record) and (args.label_context in training_record):
                    print('Found matching model: id{}'.format(args.model_id))
                    # we've found the training record for a model we care about
                    with open(os.path.join(const.TRAININGRECORDS_DIRECTORY, training_record)) as record_file:
                        record = json.load(record_file)
                        train_performance.append(record["trainingPerformance"])
                        test_performance.append(record["testPerformance"])
                        record_name = training_record[:-5]

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

    plt.savefig(os.path.join(const.FIGURE_DIRECTORY, record_name + '.pdf'), bbox_inches='tight')


def cmdscale(D):
    """
    Classical multidimensional scaling (MDS)
    Author: Francis Song; song.francis@gmail.com
    Parameters
    ----------
    D : (n, n) array
        Symmetric distance matrix.
    Returns
    -------
    Y : (n, p) array
        Configuration matrix. Each column represents a dimension. Only the
        p dimensions corresponding to positive eigenvalues of B are returned.
        Note that each dimension is only determined up to an overall sign,
        corresponding to a reflection.

    e : (n,) array
        Eigenvalues of B.
    """
    # Number of points
    n = len(D)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n))/n

    # YY^T
    B = -H.dot(D**2).dot(H)/2

    # Diagonalize
    evals, evecs = np.linalg.eigh(B)

    # Sort by eigenvalue in descending order
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)

    return Y, evals


def get_paired_test_model_id(args):
    """Construct a bipartite graph linking train/test sets between the true cue,
    blocked v interleaved conditions. So that we can take the models trained under one condition
    (e.g. blocked) and test it under the dataset from the other (e.g. interleaved).
    This function will return the test set paired with the training args listed in args.
    """
    original_blocking = copy.deepcopy(args.all_fullrange)
    args.all_fullrange = False # blocked
    all_blocked_datasets = get_model_names(args)

    args.all_fullrange = True # interleaved
    all_interleaved_datasets = get_model_names(args)

    args.all_fullrange = original_blocking
    bipartite_graph = [[] for i in range(len(all_blocked_datasets))]
    test_id = None
    if len(all_blocked_datasets) == len(all_interleaved_datasets):
        # construct bipartite graph linking the elements in these lists
        for i in range(len(all_blocked_datasets)):
            if args.all_fullrange:  # interleaved training
                if ('id'+str(args.model_id)) in all_interleaved_datasets[i]:
                    test_id = get_id_from_name(all_blocked_datasets[i])
            else:                   # blocked training
                if ('id'+str(args.model_id)) in all_blocked_datasets[i]:
                    test_id = get_id_from_name(all_interleaved_datasets[i])
    else:
        print('Warning: blocked and interleaved datasets under args not the same size')

    return test_id


def analyse_network(args):
    """Perform MDS on:
        - the hidden unit activations for each unique input in each context.
        - the averaged hidden unit activations, averaged across the unique judgement values in each context.
        - the above for both a regular test set and the cross validation set (in case we need it later)
    """
    # load the MDS analysis if we already have it and move on
    datasetname, trained_modelname, analysis_name, _ = mnet.get_dataset_name(args)

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
        trainset, testset, crossvalset, np_trainset, np_testset, np_crossvalset = dset.load_input_data(const.DATASET_DIRECTORY, datasetname)

        if args.block_int_ttsplit:
            paired_modelid = anh.get_paired_test_model_id(args)

            # test on a different (interleaved) dataset
            train_modelid = args.model_id
            args.all_fullrange = not args.all_fullrange  # flip to test on opposite blocking/interleaved structure
            args.model_id = paired_modelid
            datasetname, _, _, _ = mnet.get_dataset_name(args)
            _, testset, crossvalset, _, np_testset, np_crossvalset = dset.load_input_data(const.DATASET_DIRECTORY, datasetname)

            # revert the original model parameters for naming the analyses based on the training conditions
            args.model_id = train_modelid
            args.all_fullrange = not args.all_fullrange   # flip to return to original train block/interleaving


        # pass each input through the model and determine the hidden unit activations
        setnames = ['test', 'crossval']
        for set in setnames:

            # Assess the network activations on either the regular test set or the cross-validation set
            if set=='test':
                test_loader = DataLoader(testset, batch_size=1, shuffle=False)
            elif set =='crossval':
                test_loader = DataLoader(crossvalset, batch_size=1, shuffle=False)

            for whichTrialType in ['compare', 'filler']:
                activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, time_index, counter, drift, temporal_trialtypes = mnet.get_activations(args, np_testset, trained_model, test_loader, whichTrialType)

                dimKeep = 'judgement'                      # representation of the currently presented number, averaging over previous number
                sl_activations, sl_contexts, sl_MDSlabels, sl_refValues, sl_judgeValues, sl_counter = average_ref_numerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, args.label_context, counter)
                diff_sl_activations, diff_sl_contexts, diff_sl_MDSlabels, diff_sl_refValues, diff_sl_judgeValues, diff_sl_counter, sl_diffValues = diff_average_ref_numerosity(dimKeep, activations, labels_refValues, labels_judgeValues, labels_contexts, MDSlabels, args.label_context, counter)

                # do MDS on the activations for the test set
                print('Performing MDS on trials of type: {} in {} set...'.format(whichTrialType, set))
                tic = time.time()

                D = pairwise_distances(activations, metric='correlation') # using correlation distance
                np.fill_diagonal(np.asarray(D), 0)
                MDS_activations, _ = cmdscale(D)

                D = pairwise_distances(sl_activations, metric='correlation') # using correlation distance
                np.fill_diagonal(np.asarray(D), 0)
                MDS_slactivations, _ = cmdscale(D)

                D = pairwise_distances(diff_sl_activations, metric='correlation') # using correlation distance
                np.fill_diagonal(np.asarray(D), 0)
                MDS_diff_slactivations, _ = cmdscale(D)

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


def average_activations_across_models(args):
    """ This function takes all models trained under the conditions in args, and averages
    the resulting test activations before MDS is performed, and then do MDS on the average activations.
     - Note:  messy but functional.
    """
    allmodels = get_model_names(args)
    MDS_meandict = {}
    MDS_meandict["filler_dict"] = {}

    # acitvations and related labels collapsed over previous target
    sl_activations = [[] for i in range(len(allmodels))]
    sl_contextlabel = [[] for i in range(len(allmodels))]
    sl_numberlabel = [[] for i in range(len(allmodels))]
    filler_sl_activations = [[] for i in range(len(allmodels))]
    filler_sl_contextlabel = [[] for i in range(len(allmodels))]
    filler_sl_numberlabel = [[] for i in range(len(allmodels))]

    if args.block_int_ttsplit:
        print('Retrieving networks analysed at test under opposite blocking/interleaving to training...')
    else:
        print('Retrieving networks analysed at test under the same blocking/interleaving as training...')

    for ind, m in enumerate(allmodels):
        args.model_id = get_id_from_name(m)
        print('Loading model: {}'.format(args.model_id))
        # Analyse the trained network (extract and save network activations)
        mdict = analyse_network(args)
        sl_activations[ind] = mdict["sl_activations"]
        sl_contextlabel[ind] = mdict["sl_contexts"]
        sl_numberlabel[ind] = mdict["sl_judgeValues"]
        filler_sl_activations[ind] = mdict["filler_dict"]["sl_activations"]
        filler_sl_contextlabel[ind] = mdict["filler_dict"]["sl_contexts"]
        filler_sl_numberlabel[ind] = mdict["filler_dict"]["sl_judgeValues"]

    MDS_meandict["sl_activations"] = np.mean(sl_activations, axis=0)
    MDS_meandict["sl_contexts"] = np.mean(sl_contextlabel, axis=0)
    MDS_meandict["sl_judgeValues"] = np.mean(sl_numberlabel, axis=0)
    MDS_meandict["filler_dict"]["sl_activations"] = np.mean(filler_sl_activations, axis=0)
    MDS_meandict["filler_dict"]["sl_contexts"] = np.mean(filler_sl_contextlabel, axis=0)
    MDS_meandict["filler_dict"]["sl_judgeValues"] = np.mean(filler_sl_numberlabel, axis=0)

    # Perform MDS on averaged activations for the compare trial data
    pairwise_data = pairwise_distances(MDS_meandict["sl_activations"], metric='correlation') # using correlation distance
    np.fill_diagonal(np.asarray(pairwise_data), 0)
    MDS_act, evals = cmdscale(pairwise_data)

    # Perform MDS on averaged activations for the filler trial data
    pairwise_data = pairwise_distances(MDS_meandict["filler_dict"]["sl_activations"], metric='correlation') # using correlation distance
    np.fill_diagonal(np.asarray(pairwise_data), 0)
    MDS_act_filler, evals = cmdscale(pairwise_data)

    MDS_meandict["MDS_slactivations"] = MDS_act
    MDS_meandict["filler_dict"]["MDS_slactivations"] = MDS_act_filler
    args.model_id = 0

    return MDS_meandict, args


def cross_line_rep_generalisation(args):
    """Load activations for all models specified by args, then train a linear classifier
    for one of the lines (with input being the hidden unit representation, and
    output a binary big/small classification). Then test on the other two lines.
    Compare generalisation performance across normalised (blocked) vs absolute
    (interleaved) codes.
    """
    for dim in ['high_dim','low_dim']:
        # whether to train/test on full high-D activations, or MDS activations
        if dim == 'high_dim':
            which_activations = 'sl_activations'
            act_string = 'highD_rep'
        else:
            which_activations = 'MDS_slactivations'
            act_string = 'MDS_rep'

        models_trainscores = []
        models_testscores = []
        fig,ax = plt.subplots(1,2, figsize=(10,4))
        for bin_blocking, blocking in enumerate([False, True]):
            args.all_fullrange = blocking # False = blocked; True = interleaved

            allmodels = get_model_names(args)

            if args.block_int_ttsplit:
                print('Retrieving networks analysed at test under opposite blocking/interleaving to training...')
            else:
                print('Retrieving networks analysed at test under the same blocking/interleaving as training...')

            # specify context indices for each line
            contextA = range(const.FULLR_SPAN)
            contextB = range(const.FULLR_SPAN,const.FULLR_SPAN+const.LOWR_SPAN)
            contextC = range(const.FULLR_SPAN+const.LOWR_SPAN, const.FULLR_SPAN+const.LOWR_SPAN+const.HIGHR_SPAN)
            contexts = [contextA, contextB, contextC]

            # median split labels for each line
            y_lineA = [-1 if i<const.CONTEXT_FULL_MEAN else 1 for i in range(const.FULLR_LLIM, const.FULLR_ULIM+1)]
            y_lineB = [-1 if i<const.CONTEXT_LOW_MEAN else 1 for i in range(const.LOWR_LLIM, const.LOWR_ULIM+1)]
            y_lineC = [-1 if i<const.CONTEXT_HIGH_MEAN else 1 for i in range(const.HIGHR_LLIM, const.HIGHR_ULIM+1)]
            y_labels = [y_lineA, y_lineB, y_lineC]

            # Repeat classifier generalisation analysis for each trained RNN model
            dist_test_scores = []
            dist_train_scores = []
            for ind, m in enumerate(allmodels):

                args.model_id = get_id_from_name(m)
                #print('Loading model: {}'.format(args.model_id))

                # Analyse the trained network (extract and save network activations)
                mdict = analyse_network(args)

                # train with MDS low-D representation as input
                activations = mdict[which_activations]
                #print(activations.shape)

                generalisation = []
                train_scores = []

                # train logistic regression classifier on one line, test on the other two
                for train_index in range(const.NCONTEXTS):
                    X_train = activations[contexts[train_index],:]
                    y_train = y_labels[train_index]

                    # train a binary (big/small) linear classifier
                    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
                    train_score = clf.score(X_train, y_train)

                    # test on the other two lines
                    test_sets = [j for j in range(const.NCONTEXTS) if j != train_index]
                    test_perf = []
                    for test_set in test_sets:
                        X_test = activations[contexts[test_set],:]
                        y_test = y_labels[test_set]
                        test_perf.append(clf.score(X_test, y_test))

                    # how well did this classifier predict big/small for the other lines?
                    generalisation.append(np.mean(test_perf))
                    train_scores.append(train_score)

                #print('mean train score: {}'.format(np.mean(train_scores)))
                #print('mean test score: {}'.format(np.mean(generalisation)))
                dist_test_scores.append(generalisation)
                dist_train_scores.append(train_scores)

            dist_train_scores = np.asarray(dist_train_scores).flatten()
            dist_test_scores = np.asarray(dist_test_scores).flatten()
            models_trainscores.append(dist_train_scores)
            models_testscores.append(dist_test_scores)

            ax[0].hist(dist_train_scores, bins=np.linspace(0,1,30), alpha=0.5)
            ax[0].set_xlabel('Classifier training score')
            ax[0].set_xlim((0,1))
            ax[1].hist(dist_test_scores, bins=np.linspace(0,1,30), alpha=0.5)
            ax[1].set_xlabel('Classifier test score')
            ax[1].set_xlim((0,1))

        ax[1].legend(['Context-blocked RNN\n(normalised code)','Context-interleaved RNN\n(absolute code)'])
        #ax[0].set_ylabel('Context-blocked RNN\n(normalised code)')
        #ax[0].set_ylabel('Context-interleaved RNN\n(absolute code)')
        fig.suptitle('Logistic regression binary classifier (big/small) trained on '+ act_string)

        plt.savefig(os.path.join(const.FIGURE_DIRECTORY,'gen_classifier_'+act_string+'.pdf'), bbox_inches='tight')
        models_trainscores = np.asarray(models_trainscores)
        models_testscores = np.asarray(models_testscores)

        # Do the blocked (normalised) vs interleaved (absolute) codes yield sig. diff.
        # generalisation performance? Do an unpaired t-test (because different trained models)
        print('context-blocked, mean generalisation performance: {:.3f}'.format(np.mean(models_testscores[0,:])))
        print('context-interleaved, mean generalisation performance: {:.3f}'.format(np.mean(models_testscores[1,:])))
        tstat, p = scipy.stats.ttest_ind(models_testscores[0,:], models_testscores[1,:])
        print('t-stat: {:.3f};  p-value: {:.3e}'.format(tstat, p))
