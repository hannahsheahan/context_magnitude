"""
Code published in the following paper:
 Sheahan, H.*, Luyckx, F.*, Nelli, S., Taupe, C., & Summerfield, C. (2020). Neural normalisation supports generalisation
   of abstract knowledge in humans and recurrent networks. ArXiv
* authors contributed equally

 This is a set of simulations for training a simple RNN on a relative magnitude judgement problem.
 The network is trained to answer the question: is input N > input N-t?
 Where t is between 3-5, i.e. the inputs to be compared in each sequence are separated by several 'filler' inputs.

 Author: Hannah Sheahan, sheahan.hannah@gmail.com
 Date: 04/12/2019
 Notes:
 - requires ffmpeg for 3D animation generation in generate_plots()
 Issues: N/A
"""
# ---------------------------------------------------------------------------- #

import magnitude_network as mnet
import define_dataset as dset
import plotter as mplt
import analysis_helpers as anh
import constants as const
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import theoretical_performance as theory
import numpy as np
import os
import scipy

def plot_postlesion(args, retrain_args, model_list):
    """HRS this is an absolute mess of a hack that needs tidying up.
    There will be a lot of overlap with other function in plotter and analysis_helpers too."""

    blockingtext = '_interleaved_orig' if args.all_fullrange else '_blocked_orig'

    # allocate some space
    global_meanperf = []
    global_uniquediffs = []
    full_context_numberdiffs, low_context_numberdiffs, high_context_numberdiffs = [[] for i in range(3)]
    full_context_perf, low_context_perf, high_context_perf = [[] for i in range(3)]

    data = [[] for i in range(len(model_list))]
    context_tests = np.zeros((const.NCONTEXTS, len(model_list)))
    perf = np.zeros((const.NCONTEXTS, len(model_list)))
    counts = np.zeros((const.NCONTEXTS, len(model_list)))
    unlesioned_test = np.zeros((len(model_list),))
    lesioned_test = np.zeros((len(model_list),))

    allmodels = model_list
    SSE_local = [0 for i in range(len(allmodels))]
    SSE_global = [0 for i in range(len(allmodels))]

    fig, ax = plt.subplots(1,2)
    frequencylist = [0.4]  # training frequencies of different networks to consider
    offsets = [0-.05,.2+0.02,.2+.25+0.04]  # for plotting
    overall_lesioned_tests = []

    for ind,id in enumerate(model_list):
        retrain_args.model_id = id
        args.model_id = retrain_args.model_id
        args.train_lesion_freq = retrain_args.train_lesion_freq
        args.epochs = retrain_args.epochs
        args.lr_multi = retrain_args.lr_multi
        args.retrain_decoder = True
        _, retrained_modelname, _, _ = mnet.get_dataset_name(args)

        # test model on blocked dataset
        testParams = mnet.setup_test_parameters(args, device)
        datasetname = const.RETRAINING_DATASET
        trainset, testset, _, _, _, _ = dset.load_input_data(const.DATASET_DIRECTORY, datasetname)
        testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

        testParams[3] = testloader
        basefilename = const.LESIONS_DIRECTORY + 'lesiontests'+retrained_modelname[7:-4]
        filename = basefilename+'.npy'
        print(filename)

        # perform or load the lesion tests
        lesiondata, regulartestdata = anh.perform_lesion_tests(args, testParams, basefilename)
        data[ind] = lesiondata["bigdict_lesionperf"]
        gp, cp, gd, cd = anh.lesion_perf_by_numerosity(data[ind])
        global_meanperf.append(gp)
        global_uniquediffs.append(gd)
        full_context_perf.append(cp[0])
        low_context_perf.append(cp[1])
        high_context_perf.append(cp[2])
        full_context_numberdiffs.append(cd[0])
        low_context_numberdiffs.append(cd[1])
        high_context_numberdiffs.append(cd[2])

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

        n_sequences, n_lesions = lesiondata["bigdict_lesionperf"].shape
        for seq in range(n_sequences):
            for lesion in range(n_lesions):
                localmodel_perf = lesiondata["bigdict_lesionperf"][seq][lesion]["localmodel_perf"]
                globalmodel_perf = lesiondata["bigdict_lesionperf"][seq][lesion]["globalmodel_perf"]
                RNN_perf = lesiondata["bigdict_lesionperf"][seq][lesion]["lesion_perf"]
                SSE_local[ind] += (RNN_perf - localmodel_perf)**2
                SSE_global[ind] += (RNN_perf - globalmodel_perf)**2


    # now determine mean +-sem over models of that lesion frequency
    mean_lesioned_test = np.nanmean(lesioned_test)
    sem_lesioned_test = np.std(lesioned_test)
    mean_unlesioned_test = np.nanmean(unlesioned_test)
    sem_unlesioned_test = np.std(unlesioned_test)
    mean_contextlesion_test = np.nanmean(context_tests,axis=1)
    sem_contextlesion_test = np.std(context_tests,axis=1)

    # plot post-lesion performance divided up by context
    count =0
    handles = []
    for context in range(const.NCONTEXTS):
        colour = context+1 if context<2 else 0
        tmp = ax[0].errorbar(count, mean_contextlesion_test[colour], sem_contextlesion_test[colour], color=const.CONTEXT_COLOURS[colour], markersize=5, ecolor='black', markeredgecolor='black')
        ax[0].errorbar(count, mean_contextlesion_test[colour], sem_contextlesion_test[colour], color=const.CONTEXT_COLOURS[colour], markersize=5, marker='o', ecolor='black', markeredgecolor='black')
        count +=1
        if context==0:
            handles.append(tmp)
    print('\n')

    # format plotting
    for i, freq in enumerate(frequencylist):
        ax[i].set_xlabel('context')
        ax[i].set_ylabel(r'p(correct | $\epsilon_{train}$ = '+str(freq)+')')
        ax[i].set_ylim((60,85))
        ax[i].set_xticks([0,1,2])
        ax[i].set_xticklabels(['low','high','full'])
    plt.legend(handles[0:1],['prediction', 'RNN'])
    plt.savefig(os.path.join(const.FIGURE_DIRECTORY, 'retrained_lesionfreq_trainedlesions_new_'+blockingtext+'.pdf'), bbox_inches='tight')

    # mean over models
    global_meanperf = np.array(global_meanperf)
    full_context_perf = np.array(full_context_perf)
    low_context_perf = np.array(low_context_perf)
    high_context_perf = np.array(high_context_perf)
    global_uniquediffs = np.array(global_uniquediffs)
    full_context_numberdiffs = np.array(full_context_numberdiffs)
    low_context_numberdiffs = np.array(low_context_numberdiffs)
    high_context_numberdiffs = np.array(high_context_numberdiffs)

    global_meanperf_mean, global_meanperf_sem = mplt.get_summarystats(global_meanperf, 0)
    full_context_perf_mean, full_context_perf_sem = mplt.get_summarystats(full_context_perf, 0)
    low_context_perf_mean, low_context_perf_sem = mplt.get_summarystats(low_context_perf, 0)
    high_context_perf_mean, high_context_perf_sem = mplt.get_summarystats(high_context_perf, 0)

    global_uniquediffs = np.mean(global_uniquediffs, axis=0)
    full_context_numberdiffs = np.mean(full_context_numberdiffs, axis=0)
    low_context_numberdiffs = np.mean(low_context_numberdiffs, axis=0)
    high_context_numberdiffs = np.mean(high_context_numberdiffs, axis=0)

    fig, ax = plt.subplots(1,2)

    # generate theoretical predictions under local and global context policies
    numberdiffs, globalnumberdiffs, perf = theory.simulate_theoretical_policies()

    # context-specific performance i.e. how did performance change with dist. to mean in each context
    xnumbers =  [full_context_numberdiffs, low_context_numberdiffs, high_context_numberdiffs]
    means = [full_context_perf_mean, low_context_perf_mean, high_context_perf_mean]
    stds = [full_context_perf_sem, low_context_perf_sem, high_context_perf_sem]

    for j in range(2):
        # plot model predictions under local or global predictions
        handles = theory.plot_theoretical_predictions(ax[j], numberdiffs, globalnumberdiffs, perf, j)

        for i in range(len(xnumbers)):
            anh.shadeplot(ax[j], xnumbers[i], means[i], stds[i], const.CONTEXT_COLOURS[i])
            h = ax[j].errorbar(xnumbers[i], means[i], stds[i], color=const.CONTEXT_COLOURS[i], fmt='o', markersize=5, markeredgecolor='black', ecolor='black')
            handles.append(h)

        ax[j].set_xlabel('context distance')
        ax[j].set_xlim([-0.5, 8])
        ax[j].set_ylim([0.47, 1.03])
        ax[j].set_xticks([0,2,4,6,8])

        if j ==0:
            ax[j].legend((handles[0], handles[-1]),('prediction global context','RNN'))
        else:
            ax[j].legend((handles[0], handles[-1]),('prediction local context','RNN'))

    whichTrialType = 'compare'
    plt.savefig(os.path.join(const.FIGURE_DIRECTORY, 'retrained_perf_v_distToContextMean_postlesion'+blockingtext+'.pdf'), bbox_inches='tight')

    # Now compare the arrays of SSE for each deterministic model across the RNN instances
    Tstat, pvalue = scipy.stats.ttest_rel(SSE_local, SSE_global)
    print('local model, SSE: {}'.format(SSE_local))
    print('global model, SSE: {}'.format(SSE_global))
    print('Tstat: {}  p-value: {}'.format(Tstat, pvalue))

    return SSE_local


if __name__ == '__main__':

    # set up dataset and network hyperparams (optionally via command line)
    args, device, multiparams = mnet.define_hyperparams()
    args.label_context = 'true'   # 'true' = context cued explicitly in input; 'constant' = context not cued explicity
    args.all_fullrange = False    # False = blocked; True = interleaved
    args.train_lesion_freq = 0.0  # 0.0 or 0.1  (also 0.2, 0.3, 0.4 for blocked & true context case)
    args.block_int_ttsplit = False # test on a different distribution (block/interleave) than training
    #args.model_id = 9999          # for visualising a particular trained model

    # Train a network from scratch and save it
    #mnet.train_and_save_network(args, device, multiparams)

    # Analyse the trained network (extract and save network activations)
    #MDS_dict = anh.analyse_network(args)

    # Check the average final performance for trained models matching args
    #anh.average_perf_across_models(args)

    # Visualise the resultant network activations (RDMs and MDS)
    #MDS_dict, args = anh.average_activations_across_models(args)
    #mplt.generate_plots(MDS_dict, args)  # (Figure 3 + extras)

    # Plot the lesion test performance
    #mplt.perf_vs_context_distance(args, device)     # Assess performance after a lesion vs context distance (Figure 2 and S1)
    #mplt.compare_lesion_tests(args, device)      # compare the performance across the different lesion frequencies during training (Figure 2)

    # Statistical tests: is network behaviour better fit by an agent using the local-context or global-context policy
    #anh.model_behaviour_vs_theory(args, device)

    # Load representations and check cross-line big/small generalisation
    #anh.cross_line_rep_generalisation(args)

    #anh.cross_line_rep_generalisation_human(args)

    # Load a trained network (no VI), freeze the first layer (recurrent) weights and then retrain the decoder with VI and save it
    retrain_args, _, _ = mnet.define_hyperparams()
    retrain_args.train_lesion_freq = 0.4
    retrain_args.epochs = 20
    retrain_args.lr_multi = [0.001]
    retrain_args.retrain_decoder = True
    #anh.retrain_decoder(args, retrain_args, device, multiparams)

    SSE_local = [[] for i in range(2)]
    #for ind, args.all_fullrange in enumerate([False, True]):
    #    if args.all_fullrange:
    #        , 2922, 1347, 6213, 8594, 1600, 5219]  # interleaved initial training
    #    else:
    #        model_list = [1033, 2498, 3791, 2289, 832, 9, 8120, 1259, 6196, 7388] # blocked initial training
    model_list = [1033] #,2498, 3791]
    SSE_local[0] = plot_postlesion(args, retrain_args, model_list)
