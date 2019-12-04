"""
 This is a first pass simulation for training a simple MLP on a relational magnitude problem
 i.e. the network will be trained to answer the question: is input 2 > input 1?

 Author: Hannah Sheahan
 Date: 03/12/2019
 Notes: N/A
 Issues: N/A
"""
# ---------------------------------------------------------------------------- #

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import random
import json

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from itertools import product  # makes testing and comparing different hyperparams in tensorboard easy
import argparse                # makes defining the hyperparams and tools for running our network easier from the command line

#--------------------------------------------------#

def printProgress(i, numiter):
    """
    This function prints to the screen the optimisation progress (at each iteration i, out of a total of numiter iterations)."""

    j = (i + 1) / numiter
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%% " % ('-'*int(20*j), 100*j))
    sys.stdout.flush()

#--------------------------------------------------#

def imageBatchToTorch(originalimages):
    #originalimages = originalimages.unsqueeze(1)   # change dim for the convnet
    originalimages = originalimages.type(torch.FloatTensor)  # convert torch tensor data type
    return originalimages

# ---------------------------------------------------------------------------- #

def train(args, model, device, train_loader, optimizer, criterion, epoch, printOutput=True):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()   # zero the parameter gradients

        # provide different inputs for different models
        #print('------------')
        inputs, labels = imageBatchToTorch(data['input']), data['label'].type(torch.FloatTensor)
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
            inputs, labels = imageBatchToTorch(data['input']), data['label'].type(torch.FloatTensor)
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
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 64
        self.epochs = 10
        self.lr = 0.001
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
        parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
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

#--------------------------------------------------#

class separateinputMLP(nn.Module):
    """
        This is a simple 3-layer MLP which compares the magnitude of input nodes A to input nodes B
        """
    def __init__(self, D_in):
        super(separateinputMLP, self).__init__()
        self.fc1 = nn.Linear(D_in, 100)  # size input, size output
        self.fc2 = nn.Linear(100, 1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

#--------------------------------------------------#

def turnOneHot(integer, maxSize):
    # this function will take as input an interger and output a one hot representation of that integer up to a max of maxSize
    oneHot = np.zeros((maxSize,1))
    oneHot[integer-1] = 1
    return oneHot

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
        self.index = (self.index).astype(int)
        self.data = {'index':self.index, 'label':self.label, 'refValue':self.refValue, 'judgementValue':self.judgementValue, 'input':self.input}
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # for retrieving either a single sample of data, or a subset of data

        # lets us retrieve several items at once - check that this is working correctly HRS
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'index':self.index[idx], 'label':self.label[idx], 'refValue':self.refValue[idx], 'judgementValue':self.judgementValue[idx], 'input':self.input[idx]}
        return sample

#--------------------------------------------------#

def createSeparateInputData(maxOnehotSize, fileloc, filename):

    N = 10000         # how many examples we want to use
    Ntrain = 8000     # 8:2 train:test split

    minNumerosity = 1
    maxNumerosity = 5 if maxOnehotSize > 5 else maxOnehotSize

    refValues = np.empty((N,maxOnehotSize))
    judgementValues = np.empty((N,maxOnehotSize))
    input = np.empty((N,maxOnehotSize*2))
    target = np.empty((N,1))

    # generate some random numerosity data and label whether the random judgement integers are larger than the refValue
    for sample in range(N):
        judgementValue = random.randint(minNumerosity,maxNumerosity)
        refValue = random.randint(minNumerosity,maxNumerosity)
        input2 = turnOneHot(judgementValue, maxOnehotSize)
        input1 = turnOneHot(refValue, maxOnehotSize)

        # determine the correct rel magnitude judgement
        if judgementValue > refValue:
            target[sample] = 1
        else:
            target[sample] = 0

        judgementValues[sample] = np.squeeze(input2)
        refValues[sample] = np.squeeze(input1)
        input[sample] = np.squeeze(np.concatenate((input2,input1)))

    trainindices = np.asarray([i for i in range(Ntrain)])
    testindices = np.asarray([i for i in range(Ntrain,N)])

    trainset = { 'refValue':refValues[0:Ntrain], 'judgementValue':judgementValues[0:Ntrain], 'input':input[0:Ntrain], 'label':target[0:Ntrain], 'index':trainindices }
    testset = { 'refValue':refValues[Ntrain:], 'judgementValue':judgementValues[Ntrain:], 'input':input[Ntrain:], 'label':target[Ntrain:], 'index':testindices }

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
    trainset = data.item().get("trainset")
    testset = data.item().get("testset")

    # turn out datasets into pytorch Datasets
    trainset = createDataset(trainset)
    testset = createDataset(testset)

    return trainset, testset

#--------------------------------------------------#

def main():

    # define a network which can judge whether inputs of 1-N are greater than X
    N = 4    # range(N) are the numbers for our 'input 2' aka our judgement input

    # Define the training hyperparameters for our network (passed as args when calling main.py from command line)
    args, device, multiparams = defineHyperparams()

    # a dataset for us to work with
    createNewDataset = False
    fileloc = 'datasets/'
    datasetname = 'relmag_min1max5_dataset'
    if createNewDataset:
        trainset, testset = createSeparateInputData(N, fileloc, datasetname)
    else:
        trainset, testset = loadInputData(fileloc, datasetname)

    # Repeat the train/test model assessment for different sets of hyperparameters
    for batch_size, lr in product(*multiparams):
        args.batch_size = batch_size
        args.test_batch_size = batch_size
        args.lr = lr
        print("Network training conditions: ")
        print(args)
        print("\n")

        # Define a model for training
        model = separateinputMLP(2*N).to(device)
        criterion = nn.BCELoss() #nn.CrossEntropyLoss()  #nn.BCELoss()   # binary cross entropy loss
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Define our dataloaders
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)

        # Log the model on TensorBoard and label it with the date/time and some other naming string
        now = datetime.now()
        date = now.strftime("_%d-%m-%y_%H-%M-%S")
        comment = "_batch_size-{}_lr-{}_epochs-{}_wdecay-{}".format(args.batch_size, args.lr, args.epochs, args.weight_decay)
        writer = SummaryWriter(log_dir='results/runs/' + '_separateInputDataModel_'+ args.modeltype + date + comment)
        print("Open tensorboard in another shell to monitor network training (hannahsheahan$  tensorboard --logdir=runs)")

        # Show an example sample batch of training data...
        #showRandomBatch(trainloader)

        # Optionally save some figures etc to tensorboard about the network and inputs, input low-D embeddings etc
        #saveTbMetadata(trainloader, writer)

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
        if args.save_model:
            torch.save(model.state_dict(), "omniglot_original_cnn.pt")

    writer.close()

    # save the trained weights so we can easily look at them
    torch.save(model, 'trained_model.pth')

    # Now lets look at our trained weights
    for name, param in model.named_parameters():
        print(name)


# to run from the command line
if __name__ == '__main__':
    main()
