# MLP for paula
# 20/04/2020
#Author: Hannah Sheahan, sheahan.hannah@gmail.com

# ---------------------------------------------------------------------------- #
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# for training I/O
import argparse

# ---------------------------------------------------------------------------- #

def train(args, model, device, train_loader, optimizer, criterion, epoch, printOutput=True):
    """
    Train a neural network on the training set
    """

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
    """
    Test a neural network on the test set.
    """
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

class myMLP(nn.Module):
    """
    * HRS this is obsolete and no longer used *
        This is a simple 3-layer MLP which compares the magnitude of input nodes A (4) to input nodes B (4)
    """
    def __init__(self, D_in, hiddensize):
        super(myMLP, self).__init__()
        self.hidden_size = hiddensize   # was 60, now increase to 200 to prevent bottleneck in capacity.
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

class argsparser():
    """For holding network training arguments, usually entered via command line but simplified here"""
    def __init__(self):
        self.batch_size = 24
        self.epochs = 50
        self.lr = 0.0001
        self.momentum = 0.9
        self.no_cuda = False
        self.seed = 1
        self.weight_decay = 0.00
        self.save_model = False
        self.hidden_size = 100
        self.input_size = 10

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
    use_cuda = False #not args.no_cuda and torch.cuda.is_available()  # use cpu not gpu
    device = torch.device("cuda" if use_cuda else "cpu")

    return args, device

# ---------------------------------------------------------------------------- #

def printProgress(i, numiter):
    """This function prints to the screen the optimisation progress (at each iteration i, out of a total of numiter iterations)."""
    j = i/numiter
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%% " % ('-'*int(20*j), 100*j))
    sys.stdout.flush()

# ---------------------------------------------------------------------------- #

class createDataset(Dataset):
    """A class to hold a dataset.
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
        self.input = dataset['input']
        self.data = {'index':self.index, 'label':self.label,  'input':self.input}
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # for retrieving either a single sample of data, or a subset of data

        # lets us retrieve several items at once - check that this is working correctly HRS
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'index':self.index[idx], 'label':self.label[idx], 'input':self.input[idx] }
        return sample

# ---------------------------------------------------------------------------- #

def trainMLPNetwork(args, device, trainset, testset):
    """
    This function performs the train/test loop for training a simple MLP
    """

    print("Network training conditions: ")
    print(args)
    print("\n")

    # Define a model for training
    model = myMLP(args.input_size, args.hidden_size).to(device)
    criterion = nn.BCELoss()   # binary cross entropy loss
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Define our dataloaders
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # Train/test loop
    n_epochs = args.epochs
    printOutput = False

    print("Training network...")
    for epoch in range(1, n_epochs + 1):  # loop through the whole dataset this many times

        # train network
        train_loss, train_accuracy = train(args, model, device, trainloader, optimizer, criterion, epoch, printOutput)

        # assess network
        test_loss, test_accuracy = test(args, model, device, testloader, criterion, printOutput)

        # log performance
        print(train_accuracy, test_accuracy)
        printProgress(epoch, n_epochs)

    print("Training complete.")

    return model

# ---------------------------------------------------------------------------- #

# some example code for paula

# define train and test sets using our Dataset-inherited class (so that we can use the pytorch DataLoaders)
numpy_trainset = ...
numpy_testset = ...
trainset = createDataset(numpy_trainset)
testset = createDataset(numpy_testset)

# set up hyperparameter settings
args, device = defineHyperparams()

# train and test network
trained_model = trainMLPNetwork(args, device, trainset, testset)
