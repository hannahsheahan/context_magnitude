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

if __name__ == '__main__':

    # set up dataset and network hyperparams via command line
    args, device, multiparams = mnet.defineHyperparams()
    args.label_context = 'true'   # 'true' = context cued explicitly in input; 'constant' = context not cued explicity
    args.all_fullrange = False    # False = blocked; True = interleaved
    args.train_lesion_freq = 0.1  # 0.0 or 0.1  (also 0.2, 0.3, 0.4 for blocked & true context case)
    args.block_int_ttsplit = False # test on a different distribution (block/interleave) than training
    # args.model_id = 646          # for visualising a particular trained model

    # Train a network from scratch and save it
    #mnet.trainAndSaveANetwork(args, device, multiparams)

    # Analyse the trained network (extract and save network activations)
    #MDS_dict = anh.analyseNetwork(args)

    # Check the average final performance for trained models matching args
    #anh.averagePerformanceAcrossModels(args)

    # Visualise the resultant network activations (RDMs and MDS)
    MDS_dict, args = anh.averageActivationsAcrossModels(args)
    mplt.generatePlots(MDS_dict, args)  # (Figure 3 + extras)

    # Plot the lesion test performance
    #mplt.perfVContextDistance(args, device)     # Assess performance after a lesion vs context distance (Figure 2 and S1)
    #mplt.compareLesionTests(args, device)      # compare the performance across the different lesion frequencies during training (Figure 2)

    # Statistical tests: is network behaviour better fit by an agent using the local-context or global-context policy
    #anh.getSSEForContextModels(args, device)

# ---------------------------------------------------------------------------- #
