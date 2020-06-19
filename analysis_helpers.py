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

# ---------------------------------------------------------------------------- #

def getModelNames(args):
    """This function finds and return all the lesion analysis file names"""

    # included factors in name from getDatasetName()  (excluding random id for model instance)
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

# ---------------------------------------------------------------------------- #

def getIdfromName(modelname):
    """Take the model name and extract the model id number from the string.
    This is useful when looping through all saved models, you can assign the args.model_id param
    to this number so that subsequent analysis and figure generation naming includes the appropriate the model-id #."""
    id_ind = modelname.find('_id')+3
    pth_ind = modelname.find('.pth')
    return  modelname[id_ind:pth_ind]

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
    """  This is a messy variant of averageReferenceNumerosity(), which averages over numbers which have the same difference (A-B).
    """

    # initializing
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

def performLesionTests(args, testParams, basefilename):
    """
    This function performLesionTests() performs lesion tests on a single network
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

# ---------------------------------------------------------------------------- #

def lesionperfbyNumerosity(lesiondata):
    """This function determines how a given model performs post lesion on different numbers and contexts"""

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
    globalnumberdiffs = dset.flattenFirstDim(globalnumberdiffs)
    numberdiffs = dset.flattenFirstDim(numberdiffs)
    perf = dset.flattenFirstDim(perf)
    meanperf, uniquediffs = performanceMean(numberdiffs, perf)
    global_meanperf, global_uniquediffs = performanceMean(globalnumberdiffs, perf)

    # assess mean performance under each context
    context1_meanperf, context1_uniquediffs = performanceMean(context_numberdiffs[0], context_perf[0])
    context2_meanperf, context2_uniquediffs = performanceMean(context_numberdiffs[1], context_perf[1])
    context3_meanperf, context3_uniquediffs = performanceMean(context_numberdiffs[2], context_perf[2])
    context_perf = [context1_meanperf, context2_meanperf, context3_meanperf]
    context_numberdiffs = [context1_uniquediffs, context2_uniquediffs, context3_uniquediffs]

    return global_meanperf, context_perf, global_uniquediffs, context_numberdiffs

# ---------------------------------------------------------------------------- #

def getSSEForContextModels(args, device):
    """This function determines the sum squared error between the rnn responses and the local vs global context models, for each RNN instance."""

    allmodels = getModelNames(args)
    SSE_local = [0 for i in range(len(allmodels))]
    SSE_global = [0 for i in range(len(allmodels))]

    for ind, m in enumerate(allmodels):
        args.model_id = getIdfromName(m)
        testParams = mnet.setupTestParameters(args, device)
        basefilename = const.LESIONS_DIRECTORY + 'lesiontests'+m[:-4]

        # perform or load the lesion tests
        lesiondata, regulartestdata = performLesionTests(args, testParams, basefilename)
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
    print('local model, SSE: {}'.format(SSE_local))
    print('global model, SSE: {}'.format(SSE_global))
    print('Tstat: {}  p-value: {}'.format(Tstat, pvalue))

# ---------------------------------------------------------------------------- #

def averagePerformanceAcrossModels(args):
    """Take the training records and determine the average train and test performance
     across all trained models that meet the conditions specified in args."""

    matched_models = getModelNames(args)
    all_training_records = os.listdir(const.TRAININGRECORDS_DIRECTORY)
    record_name = ''
    train_performance = []
    test_performance = []
    for ind, m in enumerate(matched_models):
        args.model_id = getIdfromName(m)

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

# ---------------------------------------------------------------------------- #

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

# ---------------------------------------------------------------------------- #
