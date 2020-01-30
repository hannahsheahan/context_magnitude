"""
This is a selection of functions and classes relating to pytorch network training
on the contextual magnitude mapping project with Fabrice.
A simple MLP is trained on a relational magnitude problem: is input A > input B?

Author: Hannah Sheahan, sheahan.hannah@gmail.com
Date: 13/12/2019
Notes: N/A
Issues: N/A
"""
# ---------------------------------------------------------------------------- #
import define_dataset as dset
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from sklearn.manifold import MDS
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# for training I/O
from datetime import datetime
from itertools import product
import argparse

# ---------------------------------------------------------------------------- #

def printProgress(i, numiter):
    """This function prints to the screen the optimisation progress (at each iteration i, out of a total of numiter iterations)."""
    j = (i + 1) / numiter
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%% " % ('-'*int(20*j), 100*j))
    sys.stdout.flush()

# ---------------------------------------------------------------------------- #

def batchToTorch(originalimages):
    """Convert the input batch to a torch tensor"""
    #originalimages = originalimages.unsqueeze(1)   # change dim for the convnet
    originalimages = originalimages.type(torch.FloatTensor)  # convert torch tensor data type
    return originalimages

# ---------------------------------------------------------------------------- #

def recurrent_train(args, model, device, train_loader, optimizer, criterion, epoch, retainHiddenState, printOutput=True):
    """ Train a recurrent neural network on the training set.
    This now trains whilst retaining the hidden state across all trials in the training sequence
    but being evaluated just on pairs of inputs and considering each input pair as a trial for minibatching.
     """
    model.train()
    train_loss = 0
    correct = 0

    # how to extract our paired inputs and context from our dataset
    Arange = range(15)
    Brange = range(15,30)
    contextrange = range(30,33)

    # On the very first trial on training, reset the hidden weights to zeros
    hidden = torch.zeros(args.batch_size, model.recurrent_size)
    latentstate = torch.zeros(args.batch_size, model.recurrent_size)

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()   # zero the parameter gradients
        inputs, labels = batchToTorch(data['input']), data['label'].type(torch.FloatTensor)

        # reformat the paired input so that it works for our recurrent model
        context = inputs[:, contextrange]
        inputA = torch.cat((inputs[:, Arange], context),1)
        inputB = torch.cat((inputs[:, Brange], context),1)
        recurrentinputs = [inputA, inputB]

        if not retainHiddenState:
            hidden = torch.zeros(args.batch_size, model.recurrent_size)  # only if you want to reset hidden recurrent weights
        else:
            # Note: we can still update the gradients every two steps and discard the gradients before that, and just keep the hidden state
            # to reflect the recent statistics as an initialisation rather than something we continue to backprop through.
            hidden = latentstate

        # perform two-steps of recurrence
        for i in range(2):
            # inject some noise ~= forgetting of the previous number and starting state
            noise = torch.from_numpy(np.reshape(np.random.normal(0, model.hidden_noise, hidden.shape[0]*hidden.shape[1]), (hidden.shape)))
            hidden.add_(noise)
            output, hidden = model(recurrentinputs[i], hidden)  # this hidden state will be preserved across trials
            # Since our trials are sequential and overlapping, store hidden state after only one input has been passed in and combined with original hidden state
            if i==0:
                latentstate = hidden.detach()

        loss = criterion(output, labels)
        loss.backward()         # passes the loss backwards to compute the dE/dW gradients
        optimizer.step()        # update our weights

        # evaluate performance
        train_loss += loss.item()
        output = np.squeeze(output, axis=1)
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

# ---------------------------------------------------------------------------- #

def recurrent_test(args, model, device, test_loader, criterion, retainHiddenState, printOutput=True):
    """Test a recurrent neural network on the test set.
    Note that this will need to be modified such that it tracks test performance JUST
    on the subset of trials towards the end of each block (initially for a proof of concept)
    and then later across all trials regardless of position in block. So the later trials in a block
    will have a clearer context signal because of the stats of previous inputs held in the recurrent hidden state.
    """
    model.eval()
    test_loss = 0
    correct = 0

    # how to extract our paired inputs and context from our dataset
    Arange = range(15)
    Brange = range(15,30)
    contextrange = range(30,33)

    # reset hidden recurrent weights on the very first trial ***HRS be really careful with this, ***HRS this is not right yet.
    hidden = torch.zeros(args.batch_size, model.recurrent_size)

    with torch.no_grad():  # dont track the gradients
        for batch_idx, data in enumerate(test_loader):
            inputs, labels = batchToTorch(data['input']), data['label'].type(torch.FloatTensor)

            # reformat the paired input so that it works for our recurrent model
            context = inputs[:, contextrange]
            inputA = torch.cat((inputs[:, Arange], context),1)
            inputB = torch.cat((inputs[:, Brange], context),1)
            recurrentinputs = [inputA, inputB]

            if not retainHiddenState:  # only if you want to reset hidden state between trials
                hidden = torch.zeros(args.batch_size, model.recurrent_size)

            # perform a two-step recurrence
            for i in range(2):
                # inject some noise ~= forgetting of the previous number
                noise = torch.from_numpy(np.reshape(np.random.normal(0, model.hidden_noise, hidden.shape[0]*hidden.shape[1]), (hidden.shape)))
                hidden.add_(noise)
                output, hidden = model(recurrentinputs[i], hidden)

            test_loss += criterion(output, labels).item()

            output = np.squeeze(output, axis=1)
            pred = np.zeros((output.size()))
            for i in range((output.size()[0])):
                if output[i]>0.5:
                    pred[i] = 1
                else:
                    pred[i] = 0

            tmp = np.squeeze(np.asarray(labels))
            correct += (pred==tmp).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    if printOutput:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy

# ---------------------------------------------------------------------------- #

def train(args, model, device, train_loader, optimizer, criterion, epoch, printOutput=True):
    """ Train a neural network on the training set """
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()   # zero the parameter gradients
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

# ---------------------------------------------------------------------------- #

def test(args, model, device, test_loader, criterion, printOutput=True):
    """Test a neural network on the test set. """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():  # dont track the gradients
        for batch_idx, data in enumerate(test_loader):
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

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    if printOutput:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy


# ---------------------------------------------------------------------------- #

def getActivations(trainset,trained_model,networkStyle, retainHiddenState, train_loader):
    """ This will determine the hidden unit activations for each *unique* input in the training set
     there are many repeats of inputs in the training set so just doing it over the unique ones will help speed up our MDS by loads.
     If retainHiddenState is set to True, then we will evaluate the activations while considering the hidden state retained across several trials and blocks, at the end of a block.
    """

    # how to extract our paired inputs and context from our dataset
    Arange = range(15)
    Brange = range(15,30)
    contextrange = range(30,33)

    # determine the unique inputs for the training set (there are repeats)
    # Grab the indices of the FINAL instances of each unique input in the training set
    # ***HRS change this to consider activations at all instances, then average these activations
    #  to get the mean per unique input. That should take some of the wonkiness out of the MDS lines.
    unique_inputs, uniqueind = np.unique(np.flip(trainset["input"]), axis=0, return_index=True)
    finaluniqueind = len(trainset["input"])-1 - uniqueind
    finaluniqueind = np.sort(finaluniqueind)  # order important for how we record activations in the hidden state recurrent network

    unique_labels = trainset["label"][finaluniqueind]
    unique_context = trainset["context"][finaluniqueind]
    unique_refValue = trainset["refValue"][finaluniqueind]
    unique_judgementValue = trainset["judgementValue"][finaluniqueind]

    # preallocate some space...
    labels_refValues = np.empty((len(finaluniqueind),1))
    labels_judgeValues = np.empty((len(finaluniqueind),1))
    contexts = np.empty((len(finaluniqueind),1))
    MDSlabels = np.empty((len(finaluniqueind),1))
    hdim = trained_model.hidden_size
    activations = np.empty((len(finaluniqueind), hdim))

    #  pass each input through the netwrk and see what happens to the hidden layer activations

    if not ((networkStyle=='recurrent') and retainHiddenState):
        for sample in range(len(finaluniqueind)):
            sample_input = batchToTorch(torch.from_numpy(unique_inputs[sample]))
            sample_label = unique_labels[sample]
            labels_refValues[sample] = dset.turnOneHotToInteger(unique_refValue[sample])
            labels_judgeValues[sample] = dset.turnOneHotToInteger(unique_judgementValue[sample])
            MDSlabels[sample] = sample_label
            contexts[sample] = dset.turnOneHotToInteger(unique_context[sample])

            # get the activations for that input
            if networkStyle=='mlp':
                h1activations,h2activations,_ = trained_model.get_activations(sample_input)
            elif networkStyle=='recurrent':
                if not retainHiddenState:
                    # reformat the paired input so that it works for our recurrent model
                    context = sample_input[contextrange]
                    inputA = (torch.cat((sample_input[Arange], context),0)).unsqueeze(0)
                    inputB = (torch.cat((sample_input[Brange], context),0)).unsqueeze(0)
                    recurrentinputs = [inputA, inputB]
                    h0activations = torch.zeros(1,trained_model.recurrent_size) # # reset hidden recurrent weights ***HRS hardcoding of hidden unit size for now

                    # pass inputs through the recurrent network
                    for i in range(2):
                        h0activations,h1activations,_ = trained_model.get_activations(recurrentinputs[i], h0activations)

                    activations[sample] = h1activations.detach()

    else:
        # Do a single pass through the whole training set and look out for the final instances of each unique input.
        # ***HRS Note that this will need a bit of adjusting when we remove context markers from the inputs.
        # Pass the network through the whole training set, retaining the current state until we extract the activation of the inputs of interest.
        # reset hidden recurrent weights on the very first trial ***HRS be really careful with this, ***HRS this is not right yet.

        # ***HRS this section of code needs checking think its ok but not 100% sure that elements are matched across arrays
        h0activations = torch.zeros(1, trained_model.recurrent_size)
        sample = 0
        for batch_idx, data in enumerate(train_loader):
            sample_id = finaluniqueind[sample]
            inputs, labels = batchToTorch(data['input']), data['label'].type(torch.FloatTensor)

            # reformat the paired input so that it works for our recurrent model
            context = inputs[:, contextrange]
            inputA = torch.cat((inputs[:, Arange], context),1)
            inputB = torch.cat((inputs[:, Brange], context),1)
            recurrentinputs = [inputA, inputB]

            # perform a two-step recurrence
            for i in range(2):
                h0activations,h1activations,_ = trained_model.get_activations(recurrentinputs[i], h0activations)

            if batch_idx==sample_id:
                activations[sample] = h1activations.detach()
                sample_label = unique_labels[sample]
                labels_refValues[sample] = dset.turnOneHotToInteger(unique_refValue[sample])
                labels_judgeValues[sample] = dset.turnOneHotToInteger(unique_judgementValue[sample])
                MDSlabels[sample] = sample_label
                contexts[sample] = dset.turnOneHotToInteger(unique_context[sample])
                sample += 1

    # finally, reshape the output activations and labels so that we can easily interpret RSA on the activations

    # sort all variables first by context order
    context_ind = np.argsort(contexts, axis=0)
    contexts = np.take_along_axis(contexts, context_ind, axis=0)
    activations = np.take_along_axis(activations, context_ind, axis=0)
    MDSlabels = np.take_along_axis(MDSlabels, context_ind, axis=0)
    labels_refValues = np.take_along_axis(labels_refValues, context_ind, axis=0)
    labels_judgeValues = np.take_along_axis(labels_judgeValues, context_ind, axis=0)

    # within each context, sort according to numerosity of the judgement value
    for context in range(1,4):
        ind = [i for i in range(contexts.shape[0]) if contexts[i]==context]
        numerosity_ind = np.argsort(labels_judgeValues[ind], axis=0) + ind[0]
        labels_judgeValues[ind] = np.take_along_axis(labels_judgeValues, numerosity_ind, axis=0)
        labels_refValues[ind] = np.take_along_axis(labels_refValues, numerosity_ind, axis=0)
        contexts[ind] = np.take_along_axis(contexts, numerosity_ind, axis=0)
        MDSlabels[ind] = np.take_along_axis(MDSlabels, numerosity_ind, axis=0)
        activations[ind] = np.take_along_axis(activations, numerosity_ind, axis=0)

    return activations, MDSlabels, labels_refValues, labels_judgeValues, contexts

# ---------------------------------------------------------------------------- #

class separateinputMLP(nn.Module):
    """
        This is a simple 3-layer MLP which compares the magnitude of input nodes A (4) to input nodes B (4)
        """
    def __init__(self, D_in):
        super(separateinputMLP, self).__init__()
        self.hidden_size = 60
        self.fc1 = nn.Linear(D_in, self.hidden_size)  # size input, size output
        self.fc2 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        self.fc1_activations = F.relu(self.fc1(x))
        self.fc2_activations = self.fc2(self.fc1_activations)
        self.output = torch.sigmoid(self.fc2_activations)
        return self.output

    def get_activations(self, x):
        self.forward(x)  # update the activations with the particular input
        return self.fc1_activations, self.fc2_activations, self.output

# ---------------------------------------------------------------------------- #

class OneStepRNN(nn.Module):
    """
    This is a simple recurrent network which compares the magnitude of two inputs (A and B), which are passed in sequentially.
    A in passed in first, B second.
    Both the recurrent layer and the output layer have relu activations.
    input_size = 15 + 3 for context
    Reference: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
    """
    def __init__(self, D_in, batch_size, D_out, noise_std):
        super(OneStepRNN, self).__init__()
        self.recurrent_size = 33  # to match to the parallel MLP
        self.hidden_size = 60
        self.hidden_noise = noise_std
        self.input2hidden = nn.Linear(D_in + self.recurrent_size, self.recurrent_size)
        self.input2fc1 = nn.Linear(D_in + self.recurrent_size, self.hidden_size)  # size input, size output
        self.fc1tooutput = nn.Linear(self.hidden_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu')) # uniform random xavier weight initialisation

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        self.hidden = F.relu(self.input2hidden(combined))
        self.fc1_activations = F.relu(self.input2fc1(combined))
        self.output = torch.sigmoid(self.fc1tooutput(self.fc1_activations))
        return self.output, self.hidden

    def get_activations(self, x, hidden):
        self.forward(x, hidden)  # update the activations with the particular input
        return self.hidden, self.fc1_activations, self.output

    def get_noise(self):
        return self.hidden_noise

# ---------------------------------------------------------------------------- #

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
        parser.add_argument('--batch-size-multi', nargs='*', type=int, help='input batch size (or list of batch sizes) for training (default: 48)', default=[1])
        parser.add_argument('--lr-multi', nargs='*', type=float, help='learning rate (or list of learning rates) (default: 0.001)', default=[0.001])
        parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: 48)')
        parser.add_argument('--test-batch-size', type=int, default=12, metavar='N', help='input batch size for testing (default: 48)')
        parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
        parser.add_argument('--weight_decay', type=int, default=0.0000, metavar='N', help='weight-decay for l2 regularisation (default: 0)')
        parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
        args = parser.parse_args()

    multiparams = [args.batch_size_multi, args.lr_multi]
    return args, device, multiparams

# ---------------------------------------------------------------------------- #

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

# ---------------------------------------------------------------------------- #

class argsparser():
    """For holding network training arguments, usually entered via command line"""
    def __init__(self):
        self.batch_size = 24
        self.test_batch_size = 24
        self.epochs = 50
        self.lr = 0.002
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 1000
        self.weight_decay = 0.00
        self.save_model = False

# ---------------------------------------------------------------------------- #

def getDatasetName(networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState):

    if blockTrain:
        blockedtext = '_blocked'
    else:
        blockedtext = ''
    if seqTrain:
        seqtext = '_sequential'
    else:
        seqtext = ''
    if retainHiddenState:
        hiddenstate = '_retainedhidden'
    else:
        hiddenstate = '_resethidden'
    if labelContext=='true':
        contextlabelledtext = '_contextlabelled'
    elif labelContext=='random':
        contextlabelledtext = '_randomcontextlabel'
    elif labelContext=='constant':
        contextlabelledtext = '_constantcontextlabel'

    datasetname = 'dataset'+contextlabelledtext+blockedtext+seqtext
    if networkStyle=='recurrent':
        trained_modelname = 'models/'+networkStyle+'_trainedmodel'+contextlabelledtext+blockedtext+seqtext+hiddenstate+'_'+str(noise_std)+'.pth'
    else:
        trained_modelname = 'models/'+networkStyle+'_trainedmodel'+contextlabelledtext+blockedtext+seqtext+hiddenstate+'.pth'

    return datasetname, trained_modelname

# ---------------------------------------------------------------------------- #

def trainMLPNetwork(args, device, multiparams, trainset, testset, N):
    """This function performs the train/test loop for different parameter settings
     input by the user in multiparams.
     - Train/test performance is logged with a SummaryWriter
     - the trained model is returned
     """
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
            fair_train_loss, fair_train_accuracy = test(args, model, device, trainloader, criterion, printOutput)
            test_loss, test_accuracy = test(args, model, device, testloader, criterion, printOutput)

            # log performance
            train_perf = [standard_train_loss, standard_train_accuracy, fair_train_loss, fair_train_accuracy]
            test_perf = [test_loss, test_accuracy]
            print(standard_train_accuracy, test_accuracy)
            logPerformance(writer, epoch, train_perf, test_perf)
            printProgress(epoch-1, n_epochs)

        print("Training complete.")

    writer.close()
    return model

# ---------------------------------------------------------------------------- #

def trainRecurrentNetwork(args, device, multiparams, trainset, testset, N, noise_std, retainHiddenState):
    """This function performs the train/test loop for different parameter settings
     input by the user in multiparams.
     - Train/test performance is logged with a SummaryWriter
     - the trained recurrent model is returned
     - note that the train and test set must be divisible by args.batch_size, do to the shaping of the recurrent input
     """

    # Repeat the train/test model assessment for different sets of hyperparameters
    for batch_size, lr in product(*multiparams):
        args.batch_size = batch_size
        args.test_batch_size = batch_size
        args.lr = lr
        print("Network training conditions: ")
        print(args)
        print("\n")

        # Define a model for training
        model = OneStepRNN(N+3, args.batch_size, 1, noise_std).to(device)
        criterion = nn.BCELoss() #nn.CrossEntropyLoss()   # binary cross entropy loss
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
        trainingPerformance, testPerformance = [[] for i in range(2)]

        print("Training network...")

        # Take baseline performance measures
        _, base_train_accuracy = recurrent_test(args, model, device, trainloader, criterion, retainHiddenState, printOutput)
        _, base_test_accuracy = recurrent_test(args, model, device, testloader, criterion, retainHiddenState, printOutput)
        trainingPerformance.append(base_train_accuracy)
        testPerformance.append(base_test_accuracy)

        for epoch in range(1, n_epochs + 1):  # loop through the whole dataset this many times

            # train network
            standard_train_loss, standard_train_accuracy = recurrent_train(args, model, device, trainloader, optimizer, criterion, epoch, retainHiddenState, printOutput)

            # assess network
            fair_train_loss, fair_train_accuracy = recurrent_test(args, model, device, trainloader, criterion, retainHiddenState, printOutput)
            test_loss, test_accuracy = recurrent_test(args, model, device, testloader, criterion, retainHiddenState, printOutput)

            # log performance
            train_perf = [standard_train_loss, standard_train_accuracy, fair_train_loss, fair_train_accuracy]
            test_perf = [test_loss, test_accuracy]
            trainingPerformance.append(standard_train_accuracy)
            testPerformance.append(test_accuracy)
            print(standard_train_accuracy, test_accuracy)
            logPerformance(writer, epoch, train_perf, test_perf)
            printProgress(epoch-1, n_epochs)

        print("Training complete.")
        # save this training curve
        record = {"trainingPerformance":trainingPerformance, "testPerformance":testPerformance, "args":vars(args), "model":"recurrent_nocontext" }
        randnum = str(random.randint(0,10000))
        dat = json.dumps(record)
        f = open("trainingrecords/"+randnum+"_modeltrainingrecord_nocontext.json","w")
        f.write(dat)
        f.close()

    writer.close()
    return model

# ---------------------------------------------------------------------------- #
