# numpy to matlab .m conversion
# Author: Hannah Sheahan, sheahan.hannah@gmail.com
# Date: 08/06/2020

import os
import numpy as np
import scipy.io
import constants as const
import scipy.stats as stats
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

class network_args():
    def __init__(self, blocking, label, train_lesion_freq):
        self.all_fullrange = True if blocking=='intermingled' else False
        self.blocking = blocking
        self.label_context = label
        self.network_style = 'recurrent'
        self.retain_hidden_state = True
        self.which_context = 0
        self.batch_size_multi = [0]
        self.lr_multi = [0]
        self.epochs = 0
        self.recurrent_size = 0
        self.hidden_size = 0
        self.BPTT_len = 0
        self.train_lesion_freq = train_lesion_freq
        self.model_id = 0
        self.noise_std = 0.0


def import_RNN_data(basepath, args):
    """ import the rnn activations for each of ten models we care about and compute correlation distance RDM"""
    rnn_files = os.listdir(basepath)
    rnn_files = [x for x in rnn_files if (args.blocking in x) and (args.label_context in x) ]
    rnn_files = [x for x in rnn_files if ('trlf' + str(args.train_lesion_freq) in x)]
    rnn_activations = []

    # data for each rnn instance
    for filename in rnn_files:
        filepath = os.path.join(basepath, filename)
        with open(filepath) as rnn_file:
            rnn_instance_data = np.load(filepath)
            rnn_activations.append(rnn_instance_data)

    # rearrange to order low -> high -> full to match fabrices'
    tmp = np.asarray(rnn_activations)
    full_activations = tmp[:,:const.FULLR_SPAN,:]
    low_activations = tmp[:,const.FULLR_SPAN:const.FULLR_SPAN+const.LOWR_SPAN,:]
    high_activations = tmp[:,const.FULLR_SPAN+const.LOWR_SPAN:,:]
    RNN_data = np.concatenate((low_activations, high_activations, full_activations), axis=1)

    # for each model instance, zscore the RNN data and compute a similarity matrix
    subjects_RNN_RDMs = np.zeros((RNN_data.shape[0], RNN_data.shape[1], RNN_data.shape[1]))
    for i in range(RNN_data.shape[0]):
        instance_RNN_data = RNN_data[i]
        instance_RNN_data = stats.zscore(instance_RNN_data, axis=None)  # z-score the RNN raw data
        instance_RNN_RDM = pairwise_distances(instance_RNN_data, metric='correlation')
        np.fill_diagonal(np.asarray(instance_RNN_RDM), 0)
        subjects_RNN_RDMs[i] = instance_RNN_RDM

    return subjects_RNN_RDMs   # correlation distance:  [model instance x conditions x activity]


# Define network args
basepath = 'activations_for_chris/training_lesioned/'
blocking = 'blocked' # ['intermingled', 'blocked']
contextlabel = 'constant' # ['true','constant']
lesion_freq = 0.0
args = network_args(blocking, contextlabel, lesion_freq)

# Load in all files meeting criteria into single numpy array
rnn_activations = import_RNN_data(basepath, args)
print(rnn_activations.shape)   # [subjects] x [low - high - full]

# Save as .m file
scipy.io.savemat(os.path.join(basepath,'RNN_RDM_'+blocking+'_'+contextlabel+'contextlabel_'+'lesionfreq'+str(lesion_freq))+'.mat', mdict={'rnn_activations': rnn_activations})
