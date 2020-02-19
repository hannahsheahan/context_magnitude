"""
This is a selection of functions for creating a dataset for network training
on the contextual magnitude mapping project with Fabrice. The dataset is structured
to match the experimental conditions in Fabrice's EEG experiment as well as possible.

The network defined in 'mag_network.py' is simple MLP is trained on a relational
magnitude problem: is input A > input B?

Author: Hannah Sheahan, sheahan.hannah@gmail.com
Date: 13/12/2019
Notes: N/A
Issues: N/A
"""
# ---------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import random
from sklearn.manifold import MDS
from sklearn.utils import shuffle
import copy

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

# ---------------------------------------------------------------------------- #

def turnOneHot(integer, maxSize):
    """This function will take as input an interger and output a one hot representation of that integer up to a max of maxSize."""
    oneHot = np.zeros((maxSize,1))
    oneHot[integer-1] = 1
    return oneHot

# ---------------------------------------------------------------------------- #

def turnOneHotToInteger(onehot):
    """This function will take as input a one hot representation and determine the integer interpretation"""
    integer = np.nonzero(onehot)[0]
    return integer+1  # because we are starting counting from 1 not 0

# ---------------------------------------------------------------------------- #

def flattenAllFirstDimArrays(*allarrays):
    """This function will flatten the first dimension of a series of input numpy arrays"""
    flatarrays = []
    for array in allarrays:
        array = flattenFirstDim(array)
        flatarrays.append(array)
    return  flatarrays

#--------------------------------------------------#

def flattenFirstDim(array):
    """This function with return a numpy array which flattens the first two dimensions together. Only works for 3d np arrays."""
    if len(array.shape) == 3:
        return array.reshape(array.shape[0]*array.shape[1], array.shape[2])
    elif len(array.shape) == 4:
        return array.reshape(array.shape[0]*array.shape[1], array.shape[2], array.shape[3])
    else:
        print('Error: the array you are trying to partially flatten is not the correct shape.')

#--------------------------------------------------#

class createDataset(Dataset):
    """A class to hold a dataset.
    - judgementValue i.e. input2
    - refValue i.e. input1
    - total concatenate input = [input2,input1]
    - label
    """

    def __init__(self, dataset, transform=None):
        """
        Args:
            datafile (string): name of numpy datafile
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        # load all original images too - yes memory intensive but useful. Note that this also removes the efficiency point of using dataloaders
        self.index = dataset['index']
        self.label = dataset['label']
        self.refValue = dataset['refValue']
        self.judgementValue = dataset['judgementValue']
        self.input = dataset['input']
        self.context = dataset['context']
        self.contextinput = dataset['contextinputs']
        self.index = (self.index).astype(int)
        self.data = {'index':self.index, 'label':self.label, 'refValue':self.refValue, 'judgementValue':self.judgementValue, 'input':self.input, 'context':self.context, 'contextinput':self.contextinput}
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # for retrieving either a single sample of data, or a subset of data

        # lets us retrieve several items at once - check that this is working correctly HRS
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'index':self.index[idx], 'label':self.label[idx], 'refValue':self.refValue[idx], 'judgementValue':self.judgementValue[idx], 'input':self.input[idx], 'context':self.context[idx], 'contextinput':self.contextinput[idx] }
        return sample

# ---------------------------------------------------------------------------- #

def loadInputData(fileloc,datasetname):
    print('Loading dataset: ' + datasetname + '.npy')
    # load an existing dataset
    data = np.load(fileloc+datasetname+'.npy', allow_pickle=True)
    numpy_trainset = data.item().get("trainset")
    numpy_testset = data.item().get("testset")

    # turn out datasets into pytorch Datasets
    trainset = createDataset(numpy_trainset)
    testset = createDataset(numpy_testset)

    return trainset, testset, numpy_trainset, numpy_testset

# ---------------------------------------------------------------------------- #

def createSeparateInputData(totalMaxNumerosity, fileloc, filename, blockedTraining=True, sequentialABTraining=True, labelContext='true'):
    """This function will create a dataset of inputs for training/testing a network on a relational magnitude task.
    - There are 3 contexts.
    - the inputs to this function determine the structure in the training and test sets e.g. are they blocked by context.
    - 18/02 updated for training on sequences with BPTT
    """

    # note that if there is no context blocking, we can't have sequential AB training structure.
    # Double Check this:
    if not blockedTraining:
        sequentialABTraining = False

    print('Generating dataset...')
    if labelContext=='true':
        print('- network has correct context labelling')
    elif labelContext=='random':
        print('- network has randomly assigned context labelling')
    elif labelContext=='constant':
        print('- network has constant (1) context labelling')
    if blockedTraining:
        print('- training is blocked by context')
    else:
        print('- training is temporally intermingled across contexts')
    if sequentialABTraining:
        print('- training orders A and B relative to each other in trial sequence (B @ trial t+1 == A @ trial t)')
    else:
        print('- training chooses random A and B at each time step')

    totalN = 5760         # how many examples we want to use (each of these in a sequence on numbers)
    Ntrain = 4800       # 8:2 train:test split
    Ntest = totalN - Ntrain
    Mblocks = 24          # same as fabrices experiment - there are 24 blocks across 3 different contexts
    Ncontexts = 3
    sequenceLength = 9
    trainindices = (np.asarray([i for i in range(Ntrain)])).reshape((Mblocks, int(Ntrain/Mblocks),1))
    testindices = (np.asarray([i for i in range(Ntrain,totalN)])).reshape((Mblocks, int(Ntest/Mblocks),1))

    for phase in ['train','test']:   # this method should balance context instances in train and test phases
        if phase == 'train':
            N = Ntrain
        else:
            N = totalN - Ntrain

        # perhaps set temporary N to N/24, then generate the data under each context and then shuffle order at the end?
        refValues = np.empty((Mblocks, int(N/Mblocks),sequenceLength, totalMaxNumerosity))
        judgementValues = np.empty((Mblocks, int(N/Mblocks),sequenceLength, totalMaxNumerosity))
        input = np.empty((Mblocks, int(N/Mblocks),sequenceLength, totalMaxNumerosity))
        contextinputs = np.empty((Mblocks, int(N/Mblocks), Ncontexts ))
        target = np.empty((Mblocks, int(N/Mblocks),sequenceLength))
        contexts = np.empty((Mblocks, int(N/Mblocks),Ncontexts))
        contextdigits = np.empty((Mblocks, int(N/Mblocks),1))
        blocks = np.empty((Mblocks, int(N/Mblocks),1))

        for block in range(Mblocks):

            # divide the blocks evenly across the 3 contexts
            if block < Mblocks/Ncontexts:        # 0-7     # context A
                context = 1
                minNumerosity = 1
                maxNumerosity = 15

            elif block < 2*(Mblocks/Ncontexts):  # 8-15    # context B
                context = 2
                minNumerosity = 1
                maxNumerosity = 10
            else:                                # 16-23   # context C
                context = 3
                minNumerosity = 6
                maxNumerosity = 15

            # generate some random numerosity data and label whether the random judgement integers are larger than the refValue
            firstTrialInContext = True              # reset the sequentialAB structure for each new context
            for sample in range(int(N/Mblocks)):
                input_sequence = []

                # generate adjacent sequences of inputs, where no two adjacent elements within (or between) a sequence are the same
                for item in range(sequenceLength):
                    if sequentialABTraining:
                        if firstTrialInContext and item==0:
                            refValue = random.randint(minNumerosity,maxNumerosity)
                        else:
                            refValue = copy.deepcopy(judgementValue)  # use the previous number and make sure its a copy not a reference to same piece of memory
                    else:
                        refValue = random.randint(minNumerosity,maxNumerosity)


                    judgementValue = random.randint(minNumerosity,maxNumerosity)
                    while refValue==judgementValue:    # make sure we dont do inputA==inputB for two adjacent inputs
                        judgementValue = random.randint(minNumerosity,maxNumerosity)

                    input1 = turnOneHot(refValue, totalMaxNumerosity)
                    input2 = turnOneHot(judgementValue, totalMaxNumerosity)

                    if np.all(input1==input2):
                        print('Warning: trial in dataset has two adjacent inputs the same number')
                        print('State of firstTrialInContext: {}'.format(firstTrialInContext))

                    # add our new inputs to our sequence
                    if firstTrialInContext and item==0:
                        input_sequence.append(input1)
                    input_sequence.append(input2)

                if firstTrialInContext:
                    input_sequence = input_sequence[:-1]  # make sure all sequences are the same length
                    judgementValue = turnOneHotToInteger(input_sequence[-1])  # and then make sure that the next sequence starts where this one left off (bit of a hack)
                    firstTrialInContext = False

                # Define a single context for the whole sequence
                if labelContext=='true':
                    contextinput = turnOneHot(context, 3)  # we will investigate 3 different contexts
                elif labelContext=='random':
                    # Note that NOT changing 'context' means that we should be able to see the correct range label in the RDM
                    #context = random.randint(1,3)
                    contextinput = turnOneHot(random.randint(1,3), 3)  # randomly assign each example to a context, (shuffling examples across context markers in training)
                elif labelContext=='constant':
                    # Note that NOT changing 'context' means that we should be able to see the correct range label in the RDM
                    contextinput = turnOneHot(1, 3) # just keep this constant across all contexts, so the input doesnt contain an explicit context indicator

                # determine the correct rel. magnitude judgement for each pair of adjacent numbers in the sequence
                refValue = None
                for i in range(sequenceLength):
                    if i==0:
                        target[block, sample, i] = None  # there is no feedback for the first presented number in sequence
                    else:
                        refValue = turnOneHotToInteger(input_sequence[i-1])
                        judgementValue = turnOneHotToInteger(input_sequence[i])

                        if judgementValue > refValue:
                            target[block, sample, i] = 1
                        else:
                            target[block, sample, i] = 0

                contextdigits[block, sample] = context
                tmp = copy.deepcopy(input_sequence)
                tmp[0] = np.zeros((15,1))  # the first element in sequence cannot be judgement element
                judgementValues[block, sample] = np.squeeze(np.asarray(tmp))
                tmp = copy.deepcopy(input_sequence)
                tmp[-1] = np.zeros((15,1)) # the final element in sequence cannot be reference element

                refValues[block, sample] = np.squeeze(np.asarray(tmp))
                contexts[block, sample] = np.squeeze(turnOneHot(context, 3))  # still captures context here even if we dont feed context label into network
                contextinputs[block, sample] = np.squeeze(contextinput)
                #input[block, sample] = np.squeeze(np.concatenate((input2,input1,contextinput)))  # for the MLP
                input[block, sample] = np.squeeze(np.asarray(input_sequence))             # for the RNN with BPTT
                blocks[block, sample] = block

        if phase=='train':

            # now shuffle the training block order so that we temporally separate contexts a bit but still blocked
            input, refValues, judgementValues, target, contexts, contextdigits, trainindices, blocks, contextinputs = shuffle(input, refValues, judgementValues, target, contexts, contextdigits, trainindices, blocks, contextinputs, random_state=0)

            # now flatten across the first dim of the structure
            input = flattenFirstDim(input)
            refValues = flattenFirstDim(refValues)
            judgementValues = flattenFirstDim(judgementValues)
            target = flattenFirstDim(target)
            contexts = flattenFirstDim(contexts)
            contextdigits = flattenFirstDim(contextdigits)
            trainindices = flattenFirstDim(trainindices)
            blocks = flattenFirstDim(blocks)
            contextinputs = flattenFirstDim(contextinputs)

            # if you want to destroy the trial by trial sequential context and all other structure, then shuffle again across the trial order
            if not blockedTraining:
                input, refValues, judgementValues, target, contexts, contextdigits, trainindices, blocks, contextinputs = shuffle(input, refValues, judgementValues, target, contexts, contextdigits, trainindices, blocks, contextinputs, random_state=0)
            trainset = { 'refValue':refValues, 'judgementValue':judgementValues, 'input':input, 'label':target, 'index':trainindices, 'context':contexts, 'contextdigits':contextdigits, 'contextinputs':contextinputs }
        else:

            # now shuffle the training block order so that we temporally separate contexts a bit but still blocked
            input, refValues, judgementValues, target, contexts, contextdigits, testindices, blocks, contextinputs = shuffle(input, refValues, judgementValues, target, contexts, contextdigits, testindices, blocks, contextinputs, random_state=0)

            # now flatten across the first dim of the structure
            input = flattenFirstDim(input)
            refValues = flattenFirstDim(refValues)
            judgementValues = flattenFirstDim(judgementValues)
            target = flattenFirstDim(target)
            contexts = flattenFirstDim(contexts)
            contextdigits = flattenFirstDim(contextdigits)
            testindices = flattenFirstDim(testindices)
            blocks = flattenFirstDim(blocks)
            contextinputs = flattenFirstDim(contextinputs)

            # now shuffle the first axis of the dataset (consistently across the dataset) before we divide into train/test sets
            if not blockedTraining: # this shuffling will destroy the trial by trial sequential context and all other structure
                input, refValues, judgementValues, target, contexts, contextdigits, testindices, contextinputs = shuffle(input, refValues, judgementValues, target, contexts, contextdigits, testindices, contextinputs, random_state=0)
            testset = { 'refValue':refValues, 'judgementValue':judgementValues, 'input':input, 'label':target, 'index':testindices, 'context':contexts, 'contextdigits':contextdigits, 'contextinputs':contextinputs }

    # save the dataset so  we can use it again
    dat = {"trainset":trainset, "testset":testset}
    np.save(fileloc+filename+'.npy', dat)

    # turn out datasets into pytorch Datasets
    trainset = createDataset(trainset)
    testset = createDataset(testset)

    return trainset, testset

# ---------------------------------------------------------------------------- #
