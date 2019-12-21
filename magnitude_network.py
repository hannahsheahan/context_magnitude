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

def recurrent_train(args, model, device, train_loader, optimizer, criterion, epoch, printOutput=True):
    """ Train a recurrent neural network on the training set """
    model.train()
    train_loss = 0
    correct = 0

    # how to extract our paired inputs and context from our dataset
    Arange = range(15)
    Brange = range(15,30)
    contextrange = range(30,33)

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()   # zero the parameter gradients
        inputs, labels = batchToTorch(data['input']), data['label'].type(torch.FloatTensor)

        # reformat the paired input so that it works for our recurrent model
        context = inputs[:, contextrange]
        inputA = torch.cat((inputs[:, Arange], context),1)
        inputB = torch.cat((inputs[:, Brange], context),1)
        recurrentinputs = [inputA, inputB]

        # reset hidden recurrent weights
        hidden = torch.zeros(args.batch_size, 60) # ***HRS hardcoding of hidden unit size for now

        # perform a two-step recurrence
        for i in range(2):
            # inject some noise ~= forgetting of the previous number and starting state
            noise = torch.from_numpy(np.reshape(np.random.normal(0, model.hidden_noise, hidden.shape[0]*hidden.shape[1]), (hidden.shape)))
            hidden.add_(noise)
            output, hidden = model(recurrentinputs[i], hidden)

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

def recurrent_test(args, model, device, test_loader, criterion, printOutput=True):
    """Test a recurrent neural network on the test set. """
    model.eval()
    test_loss = 0
    correct = 0

    # how to extract our paired inputs and context from our dataset
    Arange = range(15)
    Brange = range(15,30)
    contextrange = range(30,33)

    with torch.no_grad():  # dont track the gradients
        for batch_idx, data in enumerate(test_loader):
            inputs, labels = batchToTorch(data['input']), data['label'].type(torch.FloatTensor)

            # reformat the paired input so that it works for our recurrent model
            context = inputs[:, contextrange]
            inputA = torch.cat((inputs[:, Arange], context),1)
            inputB = torch.cat((inputs[:, Brange], context),1)
            recurrentinputs = [inputA, inputB]

            # reset hidden recurrent weights
            hidden = torch.zeros(args.batch_size, 60) # ***HRS hardcoding of hidden unit size for now
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

def getActivations(trainset,trained_model,networkStyle):
    """ This will determine the hidden unit activations for each *unique* input in the training set
     there are many repeats of inputs in the training set so just doing it over the unique ones will help speed up our MDS by loads.
     """
    # how to extract our paired inputs and context from our dataset
    Arange = range(15)
    Brange = range(15,30)
    contextrange = range(30,33)
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
    hdim = 60 #trained_model.hidden_size
    activations = np.empty((len(uniqueind), hdim))

    #  pass each input through the netwrk and see what happens to the hidden layer activations
    for sample in range(len(uniqueind)):
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
            # reformat the paired input so that it works for our recurrent model
            context = sample_input[contextrange]
            inputA = (torch.cat((sample_input[Arange], context),0)).unsqueeze(0)
            inputB = (torch.cat((sample_input[Brange], context),0)).unsqueeze(0)
            recurrentinputs = [inputA, inputB]
            h1activations = torch.zeros(1,60) # # reset hidden recurrent weights ***HRS hardcoding of hidden unit size for now

            # pass inputs through the recurrent network
            for i in range(2):
                h1activations,h2activations,_ = trained_model.get_activations(recurrentinputs[i], h1activations)


        activations[sample] = h1activations.detach() #h1activations.detach()

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
        self.hidden_size = 60
        self.fc1size = 60
        self.hidden_noise = noise_std
        self.input2hidden = nn.Linear(D_in + self.hidden_size, self.hidden_size)
        self.input2fc1 = nn.Linear(D_in + self.hidden_size, self.fc1size)  # size input, size output
        self.fc1tooutput = nn.Linear(self.fc1size, 1)

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
        parser.add_argument('--batch-size-multi', nargs='*', type=int, help='input batch size (or list of batch sizes) for training (default: 48)', default=[24])
        parser.add_argument('--lr-multi', nargs='*', type=float, help='learning rate (or list of learning rates) (default: 0.001)', default=[0.003])
        parser.add_argument('--batch-size', type=int, default=24, metavar='N', help='input batch size for training (default: 48)')
        parser.add_argument('--test-batch-size', type=int, default=24, metavar='N', help='input batch size for testing (default: 48)')
        parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='learning rate (default: 0.001)')
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

def getDatasetName(networkStyle, noise_std, blockTrain, seqTrain, labelContext):

    if blockTrain:
        blockedtext = '_blocked'
    else:
        blockedtext = ''
    if seqTrain:
        seqtext = '_sequential'
    else:
        seqtext = ''
    if labelContext:
        contextlabelledtext = '_contextlabelled'
    else:
        contextlabelledtext = '_nocontextlabel'

    datasetname = 'dataset'+contextlabelledtext+blockedtext+seqtext
    if networkStyle=='recurrent':
        trained_modelname = 'models/'+networkStyle+'_trainedmodel'+contextlabelledtext+blockedtext+seqtext+'_'+str(noise_std)+'.pth'
    else:
        trained_modelname = 'models/'+networkStyle+'_trainedmodel'+contextlabelledtext+blockedtext+seqtext+'.pth'

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

def trainRecurrentNetwork(args, device, multiparams, trainset, testset, N, noise_std):
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

        print("Training network...")
        for epoch in range(1, n_epochs + 1):  # loop through the whole dataset this many times

            # train network
            standard_train_loss, standard_train_accuracy = recurrent_train(args, model, device, trainloader, optimizer, criterion, epoch, printOutput)

            # assess network
            fair_train_loss, fair_train_accuracy = recurrent_test(args, model, device, trainloader, criterion, printOutput)
            test_loss, test_accuracy = recurrent_test(args, model, device, testloader, criterion, printOutput)

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
