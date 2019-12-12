"""
 This is a first pass simulation for training a simple MLP on a relational magnitude problem
 i.e. the network will be trained to answer the question: is input 2 > input 1?

 Author: Hannah Sheahan
 Date: 04/12/2019
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

from itertools import product  # makes testing and comparing different hyperparams in tensorboard easy
import argparse                # makes defining the hyperparams and tools for running our network easier from the command line

#--------------------------------------------------#

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

#--------------------------------------------------#

def plot3MDSContexts(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, contextcolours, labelNumerosity, blockedTraining, saveFig):
    """This is a just function to plot the MDS of activations and label the dots with the colour of the context."""

    fig,ax = plt.subplots(1,3, figsize=(14,5))
    colours = get_cmap(10, 'magma')
    diffcolours = get_cmap(20, 'magma')
    for j in range(3):  # 3 MDS dimensions

        if j==0:
            dimA = 0
            dimB = 1
            ax[j].set_xlabel('dim 1')
            ax[j].set_ylabel('dim 2')
        elif j==1:
            dimA = 0
            dimB = 2
            ax[j].set_xlabel('dim 1')
            ax[j].set_ylabel('dim 3')
        elif j==2:
            dimA = 1
            dimB = 2
            ax[j].set_xlabel('dim 2')
            ax[j].set_ylabel('dim 3')

        for i in range((MDS_activations.shape[0])):
            # colour by context
            ax[j].set_title('context')
            ax[j].scatter(MDS_activations[i, dimA], MDS_activations[i, dimB], color=contextcolours[int(labels_contexts[i])-1])

        ax[j].axis('equal')
        ax[j].set(xlim=(-3, 3), ylim=(-3, 3))

    if blockedTraining:
        plt.suptitle('3-MDS of hidden activations: sequential relMagNet')
        if saveFig:
            plt.savefig('figures/3MDS_60hiddenactivations_relMagnet_contexts_sequential.pdf',bbox_inches='tight')
    else:
        plt.suptitle('3-MDS of hidden activations: relMagNet')
        if saveFig:
            plt.savefig('figures/3MDS_60hiddenactivations_relMagnet_contexts.pdf',bbox_inches='tight')

#--------------------------------------------------#

def plot3MDS(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, contextcolours, labelNumerosity, blockedTraining, saveFig=True):
    """This is a function to plot the MDS of activations and label according to numerosity and context"""
    # Plot the hidden activations for the 3 MDS dimensions
    fig,ax = plt.subplots(3,3, figsize=(14,15))
    colours = get_cmap(10, 'viridis')
    diffcolours = get_cmap(20, 'viridis')

    for k in range(3):
        for j in range(3):  # 3 MDS dimensions


            if j==0:
                dimA = 0
                dimB = 1
                ax[k,j].set_xlabel('dim 1')
                ax[k,j].set_ylabel('dim 2')
            elif j==1:
                dimA = 0
                dimB = 2
                ax[k,j].set_xlabel('dim 1')
                ax[k,j].set_ylabel('dim 3')
            elif j==2:
                dimA = 1
                dimB = 2
                ax[k,j].set_xlabel('dim 2')
                ax[k,j].set_ylabel('dim 3')

            for i in range((MDS_activations.shape[0])):
                if labelNumerosity:

                    # colour by numerosity
                    if k==0:
                        ax[k,j].scatter(MDS_activations[i, dimA], MDS_activations[i, dimB], color=diffcolours(int(10+labels_judgeValues[i]-labels_refValues[i])), edgecolors=contextcolours[int(labels_contexts[i])-1])
                    elif k==1:
                        ax[k,j].scatter(MDS_activations[i, dimA], MDS_activations[i, dimB], color=colours(int(labels_refValues[i])-1), edgecolors=contextcolours[int(labels_contexts[i])-1])
                    else:
                        im = ax[k,j].scatter(MDS_activations[i, dimA], MDS_activations[i, dimB], color=colours(int(labels_judgeValues[i])-1), edgecolors=contextcolours[int(labels_contexts[i])-1])
                        if j==2:
                            if i == (MDS_activations.shape[0])-1:
                                cbar = fig.colorbar(im, ticks=[0,1])
                else:
                    # colour by true/false label
                    if MDSlabels[i]==0:
                        colour = 'red'
                    else:
                        colour = 'green'
                    ax[k,j].scatter(MDS_activations[i, dimA], MDS_activations[i, dimB], color=colour)
                    if k==0:
                        ax[k,j].text(MDS_activations[i, dimA], MDS_activations[i, dimB]+0.05, str(labels_judgeValues[i][0]-labels_refValues[i][0]), color=colour)
                    elif k==1:
                        ax[k,j].text(MDS_activations[i, dimA], MDS_activations[i, dimB]+0.05, str(labels_refValues[i][0]), color=colour)
                    else:
                        ax[k,j].text(MDS_activations[i, dimA], MDS_activations[i, dimB]+0.05, str(labels_judgeValues[i][0]), color=colour)

                # some titles
                if k==0:
                    ax[k,j].set_title('value difference')
                    ax[k,j].axis('equal')
                elif k==1:
                    ax[k,j].set_title('reference')
                else:
                    ax[k,j].set_title('judgement')
                ax[k,j].set(xlim=(-3, 3), ylim=(-3, 3))  # set axes equal and the same for comparison

    if blockedTraining:
        plt.suptitle('3-MDS of hidden activations: sequential relMagNet')
        if saveFig:
            if labelNumerosity:
                cbar.ax.set_yticklabels(['1','15'])
                plt.savefig('figures/3MDS_60hiddenactivations_relMagnet_numbered_sequential.pdf',bbox_inches='tight')
            else:
                plt.savefig('figures/3MDS_60hiddenactivations_relMagnet_sequential.pdf',bbox_inches='tight')

    else:
        plt.suptitle('3-MDS of hidden activations: relMagNet')
        if saveFig:
            if labelNumerosity:
                cbar.ax.set_yticklabels(['1','15'])
                plt.savefig('figures/3MDS_60hiddenactivations_relMagnet_numbered.pdf',bbox_inches='tight')
            else:
                plt.savefig('figures/3MDS_60hiddenactivations_relMagnet.pdf',bbox_inches='tight')

#--------------------------------------------------#

def getActivations(trainset,trained_model):
    """ This will determine the hidden unit activations for each *unique* input in the training set
     there are many repeats of inputs in the training set so just doing it over the unique ones will help speed up our MDS by loads."""

    # determine the unique inputs for the training set (there are repeats)
    unique_inputs, uniqueind = np.unique(trainset["input"], axis=0, return_index=True)
    unique_labels = trainset["label"][uniqueind]
    unique_context = trainset["context"][uniqueind]
    unique_refValue = trainset["refValue"][uniqueind]
    unique_judgementValue = trainset["judgementValue"][uniqueind]

    # preallocate some space...
    labels_refValues = np.empty((len(uniqueind),1))
    labels_judgeValues = np.empty((len(uniqueind),1))
    contexts = np.empty((len(uniqueind),1))
    MDSlabels = np.empty((len(uniqueind),1))
    hdim = len(list(trained_model.fc1.parameters())[0])
    activations = np.empty(( len(uniqueind), hdim ))

    #  pass each input through the netwrk and see what happens to the hidden layer activations
    for sample in range(len(uniqueind)):
        sample_input = unique_inputs[sample]
        sample_label = unique_labels[sample]
        labels_refValues[sample] = turnOneHotToInteger(unique_refValue[sample])
        labels_judgeValues[sample] = turnOneHotToInteger(unique_judgementValue[sample])
        MDSlabels[sample] = sample_label
        contexts[sample] = turnOneHotToInteger(unique_context[sample])
        # get the activations for that input
        h1activations,_,_ = trained_model.get_activations(batchToTorch(torch.from_numpy(sample_input)))
        activations[sample] = h1activations.detach()

    return activations, MDSlabels, labels_refValues, labels_judgeValues, contexts

#--------------------------------------------------#

def printProgress(i, numiter):
    """This function prints to the screen the optimisation progress (at each iteration i, out of a total of numiter iterations)."""
    j = (i + 1) / numiter
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%% " % ('-'*int(20*j), 100*j))
    sys.stdout.flush()

#--------------------------------------------------#

def batchToTorch(originalimages):
    """Convert the input batch to a torch tensor"""
    #originalimages = originalimages.unsqueeze(1)   # change dim for the convnet
    originalimages = originalimages.type(torch.FloatTensor)  # convert torch tensor data type
    return originalimages

# ---------------------------------------------------------------------------- #

def train(args, model, device, train_loader, optimizer, criterion, epoch, printOutput=True):
    """ Train a neural network on the training set """
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()   # zero the parameter gradients

        # provide different inputs for different models
        #print('------------')
        inputs, labels = batchToTorch(data['input']), data['label'].type(torch.FloatTensor)
        output = model(inputs)
        output = np.squeeze(output, axis=1)

        loss = criterion(output, labels)
        loss.backward()         # passes the loss backwards to compute the dE/dW gradients
        optimizer.step()        # update our weights

        # evaluate performance
        train_loss += loss.item()

        pred = np.zeros((output.size()))
        for i in range((output.size()[0])):
            if output[i]>0.5:
                pred[i] = 1
            else:
                pred[i] = 0

        tmp = np.squeeze(np.asarray(labels))
        correct += (pred==tmp).sum().item()

        if batch_idx % args.log_interval == 0:
            if printOutput:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    return train_loss, accuracy

#--------------------------------------------------#

def test(args, model, device, test_loader, criterion, printOutput=True):
    """Test a neural network on the test set. """
    model.eval()
    test_loss = 0
    correct = 0

    # track class-specific performance too
    #nclasses = 1
    #class_correct = list(0. for i in range(nclasses))
    #class_total = list(0. for i in range(nclasses))

    with torch.no_grad():  # dont track the gradients
        for batch_idx, data in enumerate(test_loader):

            # provide different inputs for different models
            inputs, labels = batchToTorch(data['input']), data['label'].type(torch.FloatTensor)
            output = model(inputs)
            output = np.squeeze(output, axis=1)
            test_loss += criterion(output, labels).item()

            pred = np.zeros((output.size()))
            for i in range((output.size()[0])):
                if output[i]>0.5:
                    pred[i] = 1
                else:
                    pred[i] = 0

            tmp = np.squeeze(np.asarray(labels))
            correct += (pred==tmp).sum().item()

            # class-specific analysis
            #c = (pred.squeeze() == labels)
            #for i in range(c.shape[0]):
        #        label = labels[i]
    #            class_correct[label] += c[i].item()
#                class_total[label] += 1

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    classperformance = 0 #100 * np.divide(class_correct, class_total)  # HRS exclude class performance for now
    if printOutput:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy, classperformance

#--------------------------------------------------#

def logPerformance(writer, epoch, train_perf, test_perf):
    """ Write out the training and testing performance for this epoch to tensorboard.
          - 'writer' is a SummaryWriter instance
    Note: -  '_standard' means its the typical way people assess training performance vs test, which I think is not a fair comparison,
          because train performance will be average performance across the epoch while network is optimising/changing, vs test which is performance
          on the optimised network over that epoch.
          -  I am logging both this standard train metric and also train performance at the end of the epoch (which is a fairer comparison to test)
    """
    standard_train_loss, standard_train_accuracy, fair_train_loss, fair_train_accuracy = train_perf
    test_loss, test_accuracy = test_perf

    writer.add_scalar('Loss/training_standard', standard_train_loss, epoch)  # inputs: tag, value, iteration
    writer.add_scalar('Loss/training_fair', fair_train_loss, epoch)
    writer.add_scalar('Loss/testing', test_loss, epoch)
    writer.add_scalar('Accuracy/training_standard', standard_train_accuracy, epoch)
    writer.add_scalar('Accuracy/training_fair', fair_train_accuracy, epoch)
    writer.add_scalar('Accuracy/testing', test_accuracy, epoch)

#--------------------------------------------------#

class argsparser():
    """For holding network training arguments, usually entered via command line"""
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 64
        self.epochs = 50
        self.lr = 0.002
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 1000
        self.weight_decay = 0.00
        self.save_model = False

#--------------------------------------------------#

def defineHyperparams():
    """
    This will enable us to take different network training settings/hyperparameters in when we call main.py from the command line.
    e.g. python3 main.py --batch-size=12 --epochs=20 --save-model
    Or if you want it to execute train/test across multiple combinations of hyperparameters, you can define multiple as follows:
    e.g. python3 main.py --batch-size-multi 12 24 48 --lr-multi 0.01 0.001
    If you are running this from a notebook and not the command line, just adjust the params specified in the class argparser()
    """
    args = argsparser()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    command_line = True  # if running from jupyter notebook, set this to false and adjust argsparser() instead
    if command_line:
        parser = argparse.ArgumentParser(description='PyTorch network settings')
        parser.add_argument('--modeltype', default="aggregate", help='input type for selecting which network to train (default: "aggregate", concatenates pixel and location information)')
        parser.add_argument('--batch-size-multi', nargs='*', type=int, help='input batch size (or list of batch sizes) for training (default: 48)', default=[48])
        parser.add_argument('--lr-multi', nargs='*', type=float, help='learning rate (or list of learning rates) (default: 0.001)', default=[0.001])
        parser.add_argument('--batch-size', type=int, default=48, metavar='N', help='input batch size for training (default: 48)')
        parser.add_argument('--test-batch-size', type=int, default=48, metavar='N', help='input batch size for testing (default: 48)')
        parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.002, metavar='LR', help='learning rate (default: 0.001)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
        parser.add_argument('--weight_decay', type=int, default=0.0000, metavar='N', help='weight-decay for l2 regularisation (default: 0)')
        parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
        args = parser.parse_args()

    multiparams = [args.batch_size_multi, args.lr_multi]
    return args, device, multiparams

#--------------------------------------------------#

class separateinputMLP(nn.Module):
    """
        This is a simple 3-layer MLP which compares the magnitude of input nodes A (4) to input nodes B (4)
        """
    def __init__(self, D_in):
        super(separateinputMLP, self).__init__()
        self.fc1 = nn.Linear(D_in, 60)  # size input, size output
        self.fc2 = nn.Linear(60, 1)

    def forward(self, x):
        self.fc1_activations = F.relu(self.fc1(x))
        self.fc2_activations = self.fc2(self.fc1_activations)
        self.output = torch.sigmoid(self.fc2_activations)
        return self.output

    def get_activations(self, x):
        self.forward(x)  # update the activations with the particular input
        return self.fc1_activations, self.fc2_activations, self.output

#--------------------------------------------------#

def turnOneHot(integer, maxSize):
    """This function will take as input an interger and output a one hot representation of that integer up to a max of maxSize."""
    oneHot = np.zeros((maxSize,1))
    oneHot[integer-1] = 1
    return oneHot

#--------------------------------------------------#

def turnOneHotToInteger(onehot):
    """This function will take as input a one hot representation and determine the integer interpretation"""
    integer = np.nonzero(onehot)[0]
    return integer+1  # because we are starting counting from 1 not 0

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
        self.index = (self.index).astype(int)
        self.data = {'index':self.index, 'label':self.label, 'refValue':self.refValue, 'judgementValue':self.judgementValue, 'input':self.input, 'context':self.context}
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # for retrieving either a single sample of data, or a subset of data

        # lets us retrieve several items at once - check that this is working correctly HRS
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'index':self.index[idx], 'label':self.label[idx], 'refValue':self.refValue[idx], 'judgementValue':self.judgementValue[idx], 'input':self.input[idx], 'context':self.context[idx]}
        return sample

#--------------------------------------------------#

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
    if len(array.shape) != 3:
        print('Error: the array you are trying to partially flatten is not the correct shape.')
    else:
        return array.reshape(array.shape[0]*array.shape[1], array.shape[2])

#--------------------------------------------------#

def createSeparateInputData(totalMaxNumerosity, fileloc, filename, blockedTraining=True, sequentialABTraining=True):
    """This function will create a dataset of inputs for training/testing a network on a relational magnitude task.
    - There are 3 contexts.
    - the inputs to this function determine the structure in the training and test sets e.g. are they blocked by context.
    """

    # note that if there is no context blocking, we can't have sequential AB training structure.
    # Double Check this:
    if not blockedTraining:
        sequentialABTraining = False

    print('Generating dataset...')
    if blockedTraining:
        print('- training is blocked by context')
    else:
        print('- training is temporally intermingled across contexts')
    if sequentialABTraining:
        print('- training orders A and B relative to each other in trial sequence (B @ trial t+1 == A @ trial t)')
    else:
        print('- training chooses random A and B at each time step')

    totalN = 15000         # how many examples we want to use
    Ntrain = 12000       # 8:2 train:test split
    Ntest = totalN - Ntrain
    Mblocks = 24          # same as fabrices experiment - there are 24 blocks across 3 different contexts
    trainindices = (np.asarray([i for i in range(Ntrain)])).reshape((Mblocks, int(Ntrain/Mblocks),1))
    testindices = (np.asarray([i for i in range(Ntrain,totalN)])).reshape((Mblocks, int(Ntest/Mblocks),1))

    print(trainindices.shape)
    print(testindices.shape)
    Ncontexts = 3

    for phase in ['train','test']:   # this method should balance context instances in train and test phases
        if phase == 'train':
            N = Ntrain
        else:
            N = totalN - Ntrain

        #refValues = np.empty((N,totalMaxNumerosity))
        #judgementValues = np.empty((N,totalMaxNumerosity))
        #input = np.empty((N,totalMaxNumerosity*2+Ncontexts))
        #target = np.empty((N,1))
        #contexts = np.empty((N,Ncontexts))
        #contextdigits = np.empty((N,1))

        # perhaps set temporary N to N/24, then generate the data under each context and then shuffle order at the end?
        refValues = np.empty((Mblocks, int(N/Mblocks),totalMaxNumerosity))
        judgementValues = np.empty((Mblocks, int(N/Mblocks),totalMaxNumerosity))
        input = np.empty((Mblocks, int(N/Mblocks),totalMaxNumerosity*2+Ncontexts))
        target = np.empty((Mblocks, int(N/Mblocks),1))
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
                minNumerosity = 5
                maxNumerosity = 15

            # generate some random numerosity data and label whether the random judgement integers are larger than the refValue
            judgementValue = None              # reset the sequentialAB structure for each new context
            for sample in range(int(N/Mblocks)):
                if sequentialABTraining:
                    if judgementValue == None: # the first trial in a given context
                        refValue = random.randint(minNumerosity,maxNumerosity)
                    else:
                        refValue = copy.deepcopy(judgementValue)  # make sure its a copy not a reference to same piece of memory
                else:
                    refValue = random.randint(minNumerosity,maxNumerosity)

                judgementValue = random.randint(minNumerosity,maxNumerosity)
                while refValue==judgementValue:    # make sure we dont do inputA==inputB
                    judgementValue = random.randint(minNumerosity,maxNumerosity)

                input2 = turnOneHot(judgementValue, totalMaxNumerosity)
                input1 = turnOneHot(refValue, totalMaxNumerosity)
                contextinput = turnOneHot(context, 3)  # we will investigate 3 different contexts

                # determine the correct rel magnitude judgement
                if judgementValue > refValue:
                    target[block, sample] = 1
                else:
                    target[block, sample] = 0

                contextdigits[block, sample] = context
                judgementValues[block, sample] = np.squeeze(input2)
                refValues[block, sample] = np.squeeze(input1)
                contexts[block, sample] = np.squeeze(contextinput)
                input[block, sample] = np.squeeze(np.concatenate((input2,input1,contextinput)))
                blocks[block, sample] = block

        if phase=='train':

            # now shuffle the training block order so that we temporally separate contexts a bit but still blocked
            input, refValues, judgementValues, target, contexts, contextdigits, trainindices, blocks = shuffle(input, refValues, judgementValues, target, contexts, contextdigits, trainindices, blocks, random_state=0)

            # now flatten across the first dim of the structure
            input = flattenFirstDim(input)
            refValues = flattenFirstDim(refValues)
            judgementValues = flattenFirstDim(judgementValues)
            target = flattenFirstDim(target)
            contexts = flattenFirstDim(contexts)
            contextdigits = flattenFirstDim(contextdigits)
            trainindices = flattenFirstDim(trainindices)
            blocks = flattenFirstDim(blocks)

            # if you want to destroy the trial by trial sequential context and all other structure, then shuffle again across the trial order
            if not blockedTraining:
                input, refValues, judgementValues, target, contexts, contextdigits, trainindices, blocks = shuffle(input, refValues, judgementValues, target, contexts, contextdigits, trainindices, blocks, random_state=0)
            trainset = { 'refValue':refValues, 'judgementValue':judgementValues, 'input':input, 'label':target, 'index':trainindices, 'context':contexts, 'contextdigits':contextdigits }
        else:

            # now shuffle the training block order so that we temporally separate contexts a bit but still blocked
            input, refValues, judgementValues, target, contexts, contextdigits, testindices, blocks = shuffle(input, refValues, judgementValues, target, contexts, contextdigits, testindices, blocks, random_state=0)

            # now flatten across the first dim of the structure
            input = flattenFirstDim(input)
            refValues = flattenFirstDim(refValues)
            judgementValues = flattenFirstDim(judgementValues)
            target = flattenFirstDim(target)
            contexts = flattenFirstDim(contexts)
            contextdigits = flattenFirstDim(contextdigits)
            testindices = flattenFirstDim(testindices)
            blocks = flattenFirstDim(blocks)

            # now shuffle the first axis of the dataset (consistently across the dataset) before we divide into train/test sets
            if not blockedTraining: # this shuffling will destroy the trial by trial sequential context and all other structure
                input, refValues, judgementValues, target, contexts, contextdigits, testindices = shuffle(input, refValues, judgementValues, target, contexts, contextdigits, testindices, random_state=0)
            testset = { 'refValue':refValues, 'judgementValue':judgementValues, 'input':input, 'label':target, 'index':testindices, 'context':contexts, 'contextdigits':contextdigits }

    # save the dataset so  we can use it again
    dat = {"trainset":trainset, "testset":testset}
    np.save(fileloc+filename+'.npy', dat)

    # turn out datasets into pytorch Datasets
    trainset = createDataset(trainset)
    testset = createDataset(testset)

    return trainset, testset

#--------------------------------------------------#

def loadInputData(fileloc,datasetname):
    # load an existing dataset
    data = np.load(fileloc+datasetname+'.npy', allow_pickle=True)
    numpy_trainset = data.item().get("trainset")
    numpy_testset = data.item().get("testset")

    # turn out datasets into pytorch Datasets
    trainset = createDataset(numpy_trainset)
    testset = createDataset(numpy_testset)

    return trainset, testset, numpy_trainset, numpy_testset

#--------------------------------------------------#

def main():

    # define a network which can judge whether inputs of 1-N are greater than X
    N = 15    # total max numerosity for the greatest range we deal with

    # Define the training hyperparameters for our network (passed as args when calling main.py from command line)
    args, device, multiparams = defineHyperparams()

    # a dataset for us to work with
    createNewDataset = True
    fileloc = 'datasets/'

    blockedTraining = True           # this variable determines whether to block the training by context
    sequentialABTraining = True      # this variable determines whether the is sequential structure linking inputs A and B i.e. if at trial t+1 input B (ref) == input A from trial t
    if not blockedTraining:
        sequentialABTraining = False   # we cant have sequential AB training structure if contexts are intermingled

    if blockedTraining:
        if sequentialABTraining:
            datasetname = 'relmag_3contexts_blockedsequential_dataset'
        else:
            datasetname = 'relmag_3contexts_blockedonly_dataset'
    else:
        datasetname = 'relmag_3contexts_intermingledcontexts_dataset'

    if createNewDataset:
        trainset, testset = createSeparateInputData(N, fileloc, datasetname, blockedTraining, sequentialABTraining)
    else:
        trainset, testset, _, _ = loadInputData(fileloc, datasetname)

    # Repeat the train/test model assessment for different sets of hyperparameters
    for batch_size, lr in product(*multiparams):
        args.batch_size = batch_size
        args.test_batch_size = batch_size
        args.lr = lr
        print("Network training conditions: ")
        print(args)
        print("\n")

        # Define a model for training
        model = separateinputMLP(2*N+3).to(device)
        criterion = nn.BCELoss() #nn.CrossEntropyLoss()  #nn.BCELoss()   # binary cross entropy loss
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Define our dataloaders
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

        # Log the model on TensorBoard and label it with the date/time and some other naming string
        now = datetime.now()
        date = now.strftime("_%d-%m-%y_%H-%M-%S")
        comment = "_batch_size-{}_lr-{}_epochs-{}_wdecay-{}".format(args.batch_size, args.lr, args.epochs, args.weight_decay)
        writer = SummaryWriter(log_dir='results/runs/' + '_separateInputDataModel_'+ args.modeltype + date + comment)
        print("Open tensorboard in another shell to monitor network training (hannahsheahan$  tensorboard --logdir=runs)")

        # Train/test loop
        n_epochs = args.epochs
        printOutput = False

        print("Training network...")
        for epoch in range(1, n_epochs + 1):  # loop through the whole dataset this many times

            # train network
            standard_train_loss, standard_train_accuracy = train(args, model, device, trainloader, optimizer, criterion, epoch, printOutput)

            # assess network
            fair_train_loss, fair_train_accuracy, _ = test(args, model, device, trainloader, criterion, printOutput)
            test_loss, test_accuracy, _ = test(args, model, device, testloader, criterion, printOutput)

            # log performance
            train_perf = [standard_train_loss, standard_train_accuracy, fair_train_loss, fair_train_accuracy]
            test_perf = [test_loss, test_accuracy]
            print(standard_train_accuracy, test_accuracy)

            logPerformance(writer, epoch, train_perf, test_perf)
            printProgress(epoch-1, n_epochs)

        print("Training complete.")

    writer.close()

    # save the trained weights so we can easily look at them
    torch.save(model, 'trained_model_blockedsequentialcontexts.pth')


#--------------------------------------------------#

# Some interactive mode plotting code...
"""
# Now lets take a look at our weights and the responses to the inputs in the training set we trained on
blockedTraining = True
sequentialABTraining = True

if blockedTraining:
    if sequentialABTraining:
        trained_model = torch.load('trained_model_blockedsequentialcontexts.pth')
        datasetname = 'relmag_3contexts_blockedsequential_dataset'
    trained_model = torch.load('trained_model_sequentialcontexts.pth')
    datasetname = 'relmag_3contexts_sequential_dataset'
else:
    trained_model = torch.load('trained_model_intermingledcontexts.pth')
    datasetname = 'relmag_3contexts_dataset'

# lets load the dataset we used for training the model
fileloc = 'datasets/'
trainset, testset, np_trainset, np_testset = loadInputData(fileloc, datasetname)
loadActivations = True

# pass each input through the model and determine the hidden unit activations
activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts = getActivations(np_trainset,trained_model)

# do MDS on the activations for the training set
embedding = MDS(n_components=3)
MDS_activations = embedding.fit_transform(activations)

# plot the MDS of our hidden activations
saveFig = True
labelNumerosity = True
contextcolours = ['gold','dodgerblue', 'orangered']  #1-15, 1-10, 5-15 like fabrices colours
# plot the MDS with number labels
plot3MDS(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, contextcolours, labelNumerosity, blockedTraining, saveFig)

# plot the MDS with output labels (true/false labels)
labelNumerosity = False
plot3MDS(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, contextcolours, labelNumerosity, blockedTraining, saveFig)

# plot the MDS with context labels
plot3MDSContexts(MDS_activations, MDSlabels, labels_refValues, labels_judgeValues, labels_contexts, contextcolours, labelNumerosity, blockedTraining, saveFig)

"""

#--------------------------------------------------#

# to run from the command line
if __name__ == '__main__':
    main()
