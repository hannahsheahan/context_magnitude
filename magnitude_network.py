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
import MDSplotter as MDSplt
import numpy as np
import copy
import sys
import random
import json
import math

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
    j = i/numiter
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

def plot_grad_flow(params, args, layers, ave_grads, max_grads, batch_number):
    """
    This function will take a look at the gradients in all layers of our model during training.
    This will help us to see whether our gradients are flowing well through our RNN
    For my RNN it really needs to be divided up so that the loop is over recurrent steps rather than an unfolded model.

    - Edited from code on forum: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7 by Roshan Rane (downloaded 20/02/2020)
    - ave_grads are the gradients from our model. Call this function jsut after each backwards pass.
    """
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow, update #{}".format(batch_number))

    # save figure of gradient flow (this will take ages if you do it every loop)
    networkStyle, noise_std, blockTrain, seqTrain, givenContext, retainHiddenState, allFullRange = params
    #MDSplt.autoSaveFigure('figures/gradients/gradflow_{}_'.format(batch_number), args, 'recurrent', blockTrain, seqTrain, True, givenContext, False, noise_std, retainHiddenState, False, 'compare', True)

# ---------------------------------------------------------------------------- #

def recurrent_train(params, args, model, device, train_loader, optimizer, criterion, epoch, retainHiddenState, printOutput=True):
    """ Train a recurrent neural network on the training set.
    This now trains whilst retaining the hidden state across all trials in the training sequence
    but being evaluated just on pairs of inputs and considering each input pair as a trial for minibatching.
     """
    model.train()
    train_loss = 0
    correct = 0

    # On the very first trial on training, reset the hidden weights to zeros
    hidden = torch.zeros(args.batch_size, model.recurrent_size)
    latentstate = torch.zeros(args.batch_size, model.recurrent_size)

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()   # zero the parameter gradients
        inputs, labels, contextsequence, trialtype = batchToTorch(data['input']), data['label'].type(torch.FloatTensor)[0].unsqueeze(1).unsqueeze(1), batchToTorch(data['contextinput']), batchToTorch(data['trialtypeinput']).unsqueeze(2)

        # initialise everything for our recurrent model
        recurrentinputs = []
        sequenceLength = inputs.shape[1]
        n_comparetrials = np.nansum(np.nansum(trialtype))
        layers, ave_grads, max_grads = [[] for i in range(3)]
        loss = 0

        for i in range(sequenceLength):
            context = contextsequence[:,i]
            if trialtype[0,i]==0:  # remove context indicator on the filler trials
                context_in = torch.full_like(context, 0)
            else:
                context_in = copy.deepcopy(context)

            inputX = torch.cat((inputs[:, i], context_in, trialtype[:,i]),1)
            recurrentinputs.append(inputX)

        if not retainHiddenState:
            hidden = torch.zeros(args.batch_size, model.recurrent_size)  # only if you want to reset hidden recurrent weights
        else:
            hidden = latentstate # keep hidden state to reflect recent statistics of previous inputs

        # perform N-steps of recurrence
        for item_idx in range(sequenceLength):
            # inject some noise ~= forgetting of the previous number
            noise = torch.from_numpy(np.reshape(np.random.normal(0, model.hidden_noise, hidden.shape[0]*hidden.shape[1]), (hidden.shape)))
            hidden.add_(noise)
            output, hidden = model(recurrentinputs[item_idx], hidden)
            if item_idx==(sequenceLength-2):  # extract the hidden state just before the last input in the sequence is presented
                latentstate = hidden.detach()

            # for 'compare' trials only, evaluate performance at every comparison between the current input and previous 'compare' input
            if item_idx>0 and (trialtype[0,item_idx]==1):
                loss += criterion(output, labels[item_idx])   # accumulate the loss (autograd should sort this out for us: https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
                output = np.squeeze(output, axis=1)
                pred = np.zeros((output.size()))
                for i in range((output.size()[0])):
                    if output[i]>0.5:
                        pred[i] = 1
                    else:
                        pred[i] = 0

                tmp = np.squeeze(np.asarray(labels[item_idx]))
                correct += (pred==tmp).sum().item()

        loss.backward()             # passes the loss backwards to compute the dE/dW gradients

        # record and visualise our gradients (code from: Roshan Rane, https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10)
        if (batch_idx % 100) == 0:
            for n, p in model.named_parameters():
                if(p.requires_grad) and ("bias" not in n):
                    if p.grad is not None:
                        layers.append(n)
                        ave_grads.append(p.grad.abs().mean())
                        max_grads.append(p.grad.abs().max())
                    else:
                        # this might just trigger on leaf variables
                        print("Warning: p.grad gradient is type None at batch {} in sequence of length {}".format(batch_idx, sequenceLength) )
            plot_grad_flow(params, args, layers, ave_grads, max_grads, batch_idx)   # plot our gradients to check that they are flowing well through the RNN

        optimizer.step()            # update our weights
        train_loss += loss.item()   # keep track for display purposes

        if batch_idx % args.log_interval == 0:
            if printOutput:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader.dataset)*(n_comparetrials-1)
    accuracy = 100. * correct / (len(train_loader.dataset)*(n_comparetrials-1))
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

    # reset hidden recurrent weights on the very first trial ***HRS be really careful with this, ***HRS this is not right yet.
    hidden = torch.zeros(args.batch_size, model.recurrent_size)
    latentstate = torch.zeros(args.batch_size, model.recurrent_size)

    with torch.no_grad():  # dont track the gradients
        for batch_idx, data in enumerate(test_loader):
            inputs, labels, contextsequence, trialtype = batchToTorch(data['input']), data['label'].type(torch.FloatTensor)[0].unsqueeze(1).unsqueeze(1), batchToTorch(data['contextinput']), batchToTorch(data['trialtypeinput']).unsqueeze(2)


            # reformat the input sequences for our recurrent model
            recurrentinputs = []
            sequenceLength = inputs.shape[1]
            n_comparetrials = np.nansum(np.nansum(trialtype))

            for i in range(sequenceLength):
                context = contextsequence[:,i]
                if trialtype[0,i]==0:  # remove context indicator on the filler trials
                    context_in = torch.full_like(context, 0)
                else:
                    context_in = copy.deepcopy(context)

                inputX = torch.cat((inputs[:, i], context_in, trialtype[:,i]),1)
                recurrentinputs.append(inputX)



            if not retainHiddenState:  # only if you want to reset hidden state between trials
                hidden = torch.zeros(args.batch_size, model.recurrent_size)
            else:
                hidden = latentstate

            # perform a N-step recurrence for the whole sequence of numbers in the input
            for item_idx in range(sequenceLength):

                # inject some noise ~= forgetting of the previous number
                noise = torch.from_numpy(np.reshape(np.random.normal(0, model.hidden_noise, hidden.shape[0]*hidden.shape[1]), (hidden.shape)))
                hidden.add_(noise)
                output, hidden = model(recurrentinputs[item_idx], hidden)
                if item_idx==(sequenceLength-2):  # extract the hidden state just before the last input in the sequence is presented
                    latentstate = hidden.detach()

                if item_idx>0 and (trialtype[0,item_idx]==1):
                    test_loss += criterion(output, labels[item_idx]).item()
                    output = np.squeeze(output, axis=1)
                    pred = np.zeros((output.size()))
                    for i in range((output.size()[0])):
                        if output[i]>0.5:
                            pred[i] = 1
                        else:
                            pred[i] = 0

                    tmp = np.squeeze(np.asarray(labels[item_idx]))
                    correct += (pred==tmp).sum().item()

    test_loss /= len(test_loader.dataset)*(n_comparetrials-1)  # there are n_comparetrials-1 instances of feedback per sequence
    accuracy = 100. * correct / (len(test_loader.dataset)*(n_comparetrials-1))
    if printOutput:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset)*(n_comparetrials-1), accuracy))
    return test_loss, accuracy

# ---------------------------------------------------------------------------- #

def recurrent_lesion_test(args, model, device, test_loader, criterion, retainHiddenState, printOutput=True, whichLesion='context', lesionFrequency=1):
    """
    ***HRS new version that does it properly but will take ~120x longer to run.

    Test a recurrent neural network on the test set, while lesioning occasional inputs.

    Lesioning inputs: either the context part of the input, or the number input can be lesioned
    - lesionFrequency is the percentage (0-1) of compare trials that get randomly 'forgotten'.
       The prediction is that (for a network in which no explicit indicator of context is provided) even if we assess performance only on trials immediately after a lesion,
       as the number of lesions in the dataset increases, the ability to determine the context will decrease and performance should approach 50% chance.
       This should happen as the only way to use context to increase the P()
    - another way to test this prediction is just to look at the impact of lesioning a single trial in the dataset, and testing on the compare trial immediately following, but vary the position in the sequence at which the lesion happens.

    - should make sure that the lesioned inputs are not used to evaluate performance on (unless you explicitly want to test the impact of distance from previous number to context mean)
    """
    model.eval()

    # reset hidden recurrent weights on the very first trial
    hidden = torch.zeros(args.batch_size, model.recurrent_size)
    latentstate = torch.zeros(args.batch_size, model.recurrent_size)
    totaln_comparetrials = 0
    overallcomparisons = 0
    aggregateLesionPerf = 0
    aggregatePerf = 0
    allLesionAssessments = []

    with torch.no_grad():  # dont track the gradients
        # for each sequence
        for batch_idx, data in enumerate(test_loader):
            inputs, labels, contextsequence, contextinputsequence, trialtype = batchToTorch(data['input']), data['label'].type(torch.FloatTensor)[0].unsqueeze(1).unsqueeze(1), batchToTorch(data['context']), batchToTorch(data['contextinput']), batchToTorch(data['trialtypeinput']).unsqueeze(2)
            #print('Assessing lesions on sequence: {}'.format(batch_idx))
            # setup
            recurrentinputs = []
            sequenceLength = inputs.shape[1]
            sequenceAssessment = []

            # organise the inputs for each trial in our sequence
            for i in range(sequenceLength):
                context = contextinputsequence[:,i]
                inputcontext = copy.deepcopy(context)
                if trialtype[0,i]==0:
                    inputcontext = torch.full_like(context, 0)  # all filler trials should have no context input to it
                inputX = torch.cat((inputs[:, i], inputcontext, trialtype[:, i]),dim=1)
                recurrentinputs.append(inputX)

            # consider each number in the sequence
            for assess_idx in range(sequenceLength):
                lesionRecord = np.zeros((sequenceLength,))  # reset out lesion record
                context = dset.turnOneHotToInteger(contextsequence[:,assess_idx])[0]  # the true underlying context for this input

                # each time we repeat this exercise we need to use the original hidden state from previous sequence
                if not retainHiddenState:  # only if you want to reset hidden state between trials
                    hidden = torch.zeros(args.batch_size, model.recurrent_size)
                else:
                    hidden = latentstate

                # if its a comparison trial, we will use it to assess performance and lesion our sequence up to this point
                if trialtype[:,assess_idx]==1:
                    # Look backwards from the assessment point, lesion the immediately previous compare trial,
                    # and then every prior compare trial with frequency F
                    isPrevCompareTrial = True
                    for item_idx in range(assess_idx-1,-1,-1):
                         # lesion the compare trial immediately preceeding the assessment trial
                        if (trialtype[:,item_idx]==1) and isPrevCompareTrial:
                            lesionRecord[item_idx] = 1
                            isPrevCompareTrial = False
                        else:
                            # now lesion each other compare trial with frequency F
                            if (trialtype[:,item_idx]==1):
                                if (random.random() < lesionFrequency):
                                    lesionRecord[item_idx] = 1

                    # now that we have performed our lesions up to our assessment trial, pass this sequence through the network and assess performance
                    assess_number = dset.turnOneHotToInteger(inputs[:,assess_idx][0])[0]
                    tmpinputs = copy.deepcopy(recurrentinputs)
                    overallperf = 0
                    ncomparetrials = 0

                    # debug sanity checking that lesioning is working correctly. It seems to be! :)
                    #print('Sequence #: {}'.format(batch_idx))
                    #print('Assessment trial: {}'.format(assess_idx))
                    #print(lesionRecord)
                    #print('----')

                    for trial in range(assess_idx+1):

                        # if trial designated for lesioning, apply the lesion ***HRS to check these indices are correct for lesions
                        if lesionRecord[trial]==1:
                            if whichLesion=='number':
                                tmpinputs[trial][0:15] = torch.full_like(tmpinputs[trial][0:15], 0)
                            else:
                                tmpinputs[trial][15:18] = torch.full_like(tmpinputs[trial][15:18], 0)

                        # inject some noise ~= forgetting of the previous number
                        noise = torch.from_numpy(np.reshape(np.random.normal(0, model.hidden_noise, hidden.shape[0]*hidden.shape[1]), (hidden.shape)))
                        hidden.add_(noise)
                        output, hidden = model(tmpinputs[trial], hidden)

                        # assess aggregate performance on whole sequence (including all lesions)
                        if trialtype[:,trial]==1:
                            ncomparetrials += 1
                            output = np.squeeze(output, axis=1)
                            pred = np.zeros((output.size()))
                            for i in range((output.size()[0])):
                                if output[i]>0.5:
                                    pred[i] = 1
                                else:
                                    pred[i] = 0
                            tmp = np.squeeze(np.asarray(labels[trial]))
                            overallperf += (pred==tmp).sum().item()
                            overallcomparisons += 1
                            # once we get to the assessment trial, assess performance
                            if trial==assess_idx:
                                lesionperf = (pred==tmp).sum().item()

                    # now record the actual lesion frequency prior to the immediately preceeding lesion (to compare to random process)
                    nlesions = np.sum(lesionRecord) -1
                    # prevent divide by zero
                    if ncomparetrials == 2:
                        actualLesionF = 1.0
                    else:
                        actualLesionF = nlesions / (ncomparetrials-2) # excluding the assessment trial and the compare trial before it

                    mydict = {"assess_number":assess_number, "lesion_perf":lesionperf, "overall_perf":overallperf, "desired_lesionF":lesionFrequency, "actual_lesionF":actualLesionF, "underlying_context":context,  "assess_idx":assess_idx, "compare_idx":ncomparetrials }
                    sequenceAssessment.append(mydict)
                    aggregateLesionPerf += lesionperf
                    aggregatePerf += overallperf
                    totaln_comparetrials += 1

                # extract the hidden state just before the last input in the sequence is presented, for passing to next sequence
                # since the network has only processed sequences up to compare trials, we need to pass the whole sequence through again now! inefficient, yes
                if assess_idx==(sequenceLength-2):
                    for i in range(assess_idx+1):
                        noise = torch.from_numpy(np.reshape(np.random.normal(0, model.hidden_noise, hidden.shape[0]*hidden.shape[1]), (hidden.shape)))
                        hidden.add_(noise)
                        output, hidden = model(tmpinputs[i], hidden)  # this should be the sequence of trials that are all lesioned with probability F
                    latentstate = hidden.detach()  # ***HRS the exact trial and lesion sequence that this is passed on from will need checking, but should be approximately right

            allLesionAssessments.append(sequenceAssessment)


    # turn into numpy matrix
    allLesionAssessments = np.asarray(allLesionAssessments)

    # summary stats
    summarylesionperf = 100. *(aggregateLesionPerf / totaln_comparetrials)
    summaryperf = 100. *(aggregatePerf / overallcomparisons)
    print('Mean lesion accuracy: {}/{} ({:.2f}%)'.format(aggregateLesionPerf, totaln_comparetrials, summarylesionperf))
    print('Mean overall accuracy: {}/{} ({:.2f}%)'.format(aggregatePerf, overallcomparisons, summaryperf))

    return allLesionAssessments, summarylesionperf, summaryperf

# ---------------------------------------------------------------------------- #

def _recurrent_lesion_test(args, model, device, test_loader, criterion, retainHiddenState, printOutput=True, whichLesion='context', lesionFrequency=1):
    """Test a recurrent neural network on the test set, while lesioning occasional inputs.

    Lesioning inputs: either the context part of the input, or the number input can be lesioned
    - lesionFrequency is the percentage (0-1) of compare trials that get randomly 'forgotten'.
       The prediction is that (for a network in which no explicit indicator of context is provided) even if we assess performance only on trials immediately after a lesion,
       as the number of lesions in the dataset increases, the ability to determine the context will decrease and performance should approach 50% chance.
       This should happen as the only way to use context to increase the P()
    - another way to test this prediction is just to look at the impact of lesioning a single trial in the dataset, and testing on the compare trial immediately following, but vary the position in the sequence at which the lesion happens.

    - should make sure that the lesioned inputs are not used to evaluate performance on (unless you explicitly want to test the impact of distance from previous number to context mean)
    """
    model.eval()
    test_loss = 0
    correct = 0
    overalltest_correct = 0

    # reset hidden recurrent weights on the very first trial
    hidden = torch.zeros(args.batch_size, model.recurrent_size)
    latentstate = torch.zeros(args.batch_size, model.recurrent_size)
    totaln_lesiontrials = 0
    totaln_comparetrials = 0

    with torch.no_grad():  # dont track the gradients
        for batch_idx, data in enumerate(test_loader):
            inputs, labels, contextsequence, trialtype = batchToTorch(data['input']), data['label'].type(torch.FloatTensor)[0].unsqueeze(1).unsqueeze(1), batchToTorch(data['contextinput']), batchToTorch(data['trialtypeinput']).unsqueeze(2)

            # reformat the input sequences for our recurrent model
            recurrentinputs = []
            sequenceLength = inputs.shape[1]
            lesionRecord = np.zeros((sequenceLength,))
            n_comparetrials = np.nansum(np.nansum(trialtype))

            # organise the inputs for each trial in our sequence
            for i in range(sequenceLength):
                context = contextsequence[:,i]
                inputcontext = copy.deepcopy(context)
                if trialtype[0,i]==1: # compare trial that we can lesion
                    if (random.random() < lesionFrequency) and (i>0):
                        # lesion the input, but dont lesion the first input
                        if whichLesion=='context':
                            inputcontext = torch.full_like(context, 0)
                        else:
                            inputs[:,i] = torch.full_like(inputs[:,i], 0)
                        lesionRecord[i] = 1
                    else:
                        lesionRecord[i] = 0
                else:
                    inputcontext = torch.full_like(context, 0)  # all filler trials should have no context input to it

                inputX = torch.cat((inputs[:, i], inputcontext, trialtype[:, i]),dim=1)
                recurrentinputs.append(inputX)


            if not retainHiddenState:  # only if you want to reset hidden state between trials
                hidden = torch.zeros(args.batch_size, model.recurrent_size)
            else:
                hidden = latentstate

            wasLesioned = False
            n_lesionAssessmentTrials = 0
            n_totalAssessmentTrials = 0

            # perform a N-step recurrence for the whole sequence of numbers in the input
            for item_idx in range(sequenceLength):

                # inject some noise ~= forgetting of the previous number
                noise = torch.from_numpy(np.reshape(np.random.normal(0, model.hidden_noise, hidden.shape[0]*hidden.shape[1]), (hidden.shape)))
                hidden.add_(noise)
                output, hidden = model(recurrentinputs[item_idx], hidden)

                if item_idx==(sequenceLength-2):  # extract the hidden state just before the last input in the sequence is presented
                    latentstate = hidden.detach()

                if item_idx>0 and (trialtype[0,item_idx]==1):

                    # the current trial NOT a lesioned trial, but was the previous compare trial WAS lesioned
                    if wasLesioned and (lesionRecord[item_idx]==0):
                        n_lesionAssessmentTrials += 1
                        test_loss += criterion(output, labels[item_idx]).item()
                        output = np.squeeze(output, axis=1)
                        pred = np.zeros((output.size()))
                        for i in range((output.size()[0])):
                            if output[i]>0.5:
                                pred[i] = 1
                            else:
                                pred[i] = 0

                        tmp = np.squeeze(np.asarray(labels[item_idx]))
                        correct += (pred==tmp).sum().item()

                    # was this input lesioned? (keeping track for the next compare trial)
                    wasLesioned = True if lesionRecord[item_idx]==1 else False

                    # now take the overall performance as a sanity check
                    n_totalAssessmentTrials +=1
                    output = np.squeeze(output, axis=1)
                    pred = np.zeros((output.size()))
                    for i in range((output.size()[0])):
                        if output[i]>0.5:
                            pred[i] = 1
                        else:
                            pred[i] = 0

                    tmp = np.squeeze(np.asarray(labels[item_idx]))
                    overalltest_correct += (pred==tmp).sum().item()

            totaln_lesiontrials += n_lesionAssessmentTrials
            totaln_comparetrials += n_totalAssessmentTrials

    # prevent divide by zero
    if totaln_lesiontrials == 0:
        totaln_lesiontrials = 1
    test_loss /= totaln_lesiontrials  # there are n_lesiontrials instances of feedback per sequence
    accuracy = 100. * correct / (totaln_lesiontrials)
    overalltest_accuracy = 100. * overalltest_correct / (totaln_comparetrials)

    if printOutput:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, totaln_lesiontrials, accuracy))
    return test_loss, accuracy, overalltest_accuracy

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

def getActivations(trainset,trained_model,networkStyle, retainHiddenState, train_loader, whichType='compare'):
    """ This will determine the hidden unit activations for each *unique* input pair in the training set.
     There are many repeats of inputs in the training set.If retainHiddenState is set to True,
     then we will evaluate the activations while considering the hidden state retained across several trials and blocks.
     This will lead to slightly different activations for each instance of a particular input pair.
     We therefore take our activation for that unique input pair as the average activation over all instances of the pair in the training set.
      - HRS could change ativation assessment to be for the test set, but shouldn't make a big difference.

      - HRS note that this function could really do with a cleanup. It's a long ugly mess.

    """

    # reformat the input sequences for our recurrent model
    recurrentinputs = []
    sequenceLength = trainset["input"].shape[1]

    # constants
    TRIAL_FILLER  = 0
    TRIAL_COMPARE = 1
    TRIAL_TYPE = TRIAL_COMPARE if whichType=='compare' else TRIAL_FILLER

    # determine the unique inputs for the training set (there are repeats)
    # consider activations at all instances, then average these activations to get the mean per unique input.
    trainset_input_n_context, seq_record = [[] for i in range(2)]
    for seq in range(len(trainset["input"])):

        inputA, inputB = [None for i in range(2)]
        for item_idx in range(len(trainset["input"][seq])):
            trialtype = trainset["trialtypeinputs"][seq, item_idx]

            if TRIAL_TYPE==TRIAL_COMPARE:
                if trialtype==TRIAL_TYPE:
                    inputA = trainset["input"][seq, item_idx]
                    if inputB is not None:
                        if np.all(inputA==inputB):
                            print('Warning: adjacent trial types are same number {}, both of type compare at item {} in sequence {}'.format(dset.turnOneHotToInteger(inputA)[:], item_idx,seq))
                        context = trainset["context"][seq, item_idx]  # the actual underlying range context, not the label
                        trainset_input_n_context.append(np.append(np.append(inputA, inputB), context))
                        seq_record.append([seq, item_idx])

                    inputB = trainset["input"][seq, item_idx]  # set the previous state to be the current state

            elif TRIAL_TYPE==TRIAL_FILLER:
                if trialtype==TRIAL_TYPE:
                    inputA = trainset["input"][seq, item_idx]
                    context = trainset["context"][seq, item_idx]  # the actual underlying range context, not the label
                    trainset_input_n_context.append(np.append(inputA, context))
                    seq_record.append([seq, item_idx])


    #trainset_input_n_context = [np.append(trainset["input"][i, j],trainset["contextinput"][i]) for i in range(len(trainset["input"]))]  # ignore the context label, but consider the true underlying context
    unique_inputs_n_context, uniqueind = np.unique(trainset_input_n_context, axis=0, return_index=True)
    N_unique = (unique_inputs_n_context.shape)[0]
    sequence_id = [seq_record[uniqueind[i]][0] for i in range(len(uniqueind))]
    seqitem_id = [seq_record[uniqueind[i]][1] for i in range(len(uniqueind))]
    num_unique = len(uniqueind)
    trainsize = trainset["label"].shape[0]

    unique_inputs = np.asarray([trainset["input"][sequence_id[i]][seqitem_id[i]] for i in range(len(sequence_id))])
    unique_labels = np.asarray([trainset["label"][sequence_id[i]][seqitem_id[i]] for i in range(len(sequence_id))])
    unique_context = np.asarray([trainset["context"][sequence_id[i]][seqitem_id[i]] for i in range(len(sequence_id))])
    unique_refValue = np.asarray([trainset["refValue"][sequence_id[i]][seqitem_id[i]] for i in range(len(sequence_id))])
    unique_judgementValue = np.asarray([trainset["judgementValue"][sequence_id[i]][seqitem_id[i]] for i in range(len(sequence_id))])

    # preallocate some space...
    labels_refValues = np.empty((len(uniqueind),1))
    labels_judgeValues = np.empty((len(uniqueind),1))
    contexts = np.empty((len(uniqueind),1))
    time_index = np.empty((len(uniqueind),1))
    MDSlabels = np.empty((len(uniqueind),1))
    hdim = trained_model.hidden_size
    rdim = trained_model.recurrent_size
    activations = np.empty((len(uniqueind), hdim))
    temporal_context = np.zeros((trainsize,sequenceLength))            # for tracking the evolution of context in the training set
    temporal_trialtypes = np.zeros((trainsize,sequenceLength))
    temporal_activation_drift = np.zeros((trainsize, rdim))

    #  Tally activations for each unique context/input instance, then divide by the count (i.e. take the mean across instances)
    aggregate_activations = np.zeros((len(uniqueind), hdim))  # for adding each instance of activations to
    counter = np.zeros((len(uniqueind),1)) # for counting how many instances of each unique input/context we find

    #  pass each input through the network and see what happens to the hidden layer activations
    if not ((networkStyle=='recurrent') and retainHiddenState):
        for sample in range(len(uniqueind)):
            sample_input = batchToTorch(torch.from_numpy(unique_inputs[sample]))
            sample_label = unique_labels[sample]
            labels_refValues[sample] = dset.turnOneHotToInteger(unique_refValue[sample])
            labels_judgeValues[sample] = dset.turnOneHotToInteger(unique_judgementValue[sample])
            MDSlabels[sample] = sample_label
            contexts[sample] = dset.turnOneHotToInteger(unique_context[sample])
            time_index[sample] = 0  # doesnt mean anything for these not-sequential cases
            counter[sample] = 0     # we dont care how many instances of each unique input for these non-sequential cases

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
        # Do a single pass through the whole training set and look out for ALL instances of each unique input.
        # Pass the network through the whole training set, retaining the current state until we extract the activation of the inputs of interest.
        # reset hidden recurrent weights on the very first trial ***HRS be careful with this

        h0activations = torch.zeros(1, trained_model.recurrent_size)
        latentstate = torch.zeros(1, trained_model.recurrent_size)

        for batch_idx, data in enumerate(train_loader):
            #inputs, labels, context, contextinput = batchToTorch(data['input']), data['label'].type(torch.FloatTensor)[0], data['context'], batchToTorch(data['contextinput'])
            inputs, labels, contextsequence, contextinputsequence, trialtype = batchToTorch(data['input']), data['label'].type(torch.FloatTensor)[0].unsqueeze(1).unsqueeze(1), batchToTorch(data['context']), batchToTorch(data['contextinput']), batchToTorch(data['trialtypeinput']).unsqueeze(2)
            #print('context {}'.format(np.squeeze(dset.turnOneHotToInteger(context))[0]))
            recurrentinputs = []
            sequenceLength = inputs.shape[1]
            temporal_trialtypes[batch_idx] = data['trialtypeinput']

            for i in range(sequenceLength):
                temporal_context[batch_idx, i] = dset.turnOneHotToInteger(contextsequence[:,i])[0]
                contextin = contextinputsequence[:,i]
                if trialtype[0,i]==0:  # remove context indicator on the filler trials
                    contextinput = torch.full_like(contextin, 0)
                else:
                    contextinput = copy.deepcopy(contextin)

                inputX = torch.cat((inputs[:, i], contextinput, trialtype[:,i]),1)
                recurrentinputs.append(inputX)

            h0activations = latentstate  # because we have overlapping sequential trials

            inputA, inputB = [None for i in range(2)]
            # perform N-steps of recurrence
            for item_idx in range(sequenceLength):
                h0activations,h1activations,_ = trained_model.get_activations(recurrentinputs[item_idx], h0activations)
                if item_idx==(sequenceLength-2):  # extract the hidden state just before the last input in the sequence is presented
                    latentstate = h0activations.detach()

                context = contextsequence[:,item_idx]

                # for 'compare' trials only, evaluate performance at every comparison between the current input and previous 'compare' input
                if trialtype[0,item_idx] == TRIAL_TYPE:
                    if TRIAL_TYPE==TRIAL_COMPARE:    # if we are looking at act. for the compare trials only
                        inputA = recurrentinputs[item_idx][0][:15]
                        if inputB is not None:
                            input_n_context = np.append(np.append(inputA, inputB), context)  # actual underlying range context
                            for i in range(N_unique):
                                if np.all(unique_inputs_n_context[i,:]==input_n_context):
                                    #print('{}, {}, context {}'.format(dset.turnOneHotToInteger(inputA)[0],  dset.turnOneHotToInteger(inputB)[0] , np.squeeze(dset.turnOneHotToInteger(context))[0]  ))
                                    index = i
                                    break

                            activations[index] = h1activations.detach()
                            labels_refValues[index] = dset.turnOneHotToInteger(unique_refValue[index])
                            labels_judgeValues[index] = dset.turnOneHotToInteger(unique_judgementValue[index])
                            MDSlabels[index] = unique_labels[index]
                            contexts[index] = dset.turnOneHotToInteger(unique_context[index])
                            time_index[index] = batch_idx
                        inputB = recurrentinputs[item_idx][0][:15]  # previous state <= current state

                    else:  # for filler trials only, consider just the current number and context

                        inputA = recurrentinputs[item_idx][0][:15]
                        input_n_context = np.append(inputA, context)  # actual underlying range context
                        for i in range(N_unique):
                            if np.all(unique_inputs_n_context[i,:]==input_n_context):
                                index = i
                                break
                        activations[index] = h1activations.detach()
                        labels_refValues[index] = dset.turnOneHotToInteger(unique_refValue[index])
                        labels_judgeValues[index] = dset.turnOneHotToInteger(unique_judgementValue[index])
                        MDSlabels[index] = unique_labels[index]
                        contexts[index] = dset.turnOneHotToInteger(unique_context[index])
                        time_index[index] = batch_idx

                    if item_idx > 0:
                        # Aggregate activity associated with each instance of each input
                        aggregate_activations[index] += activations[index]
                        counter[index] += 1    # captures how many instances of each unique input there are in the training set

            temporal_activation_drift[batch_idx, :] = latentstate   # HRS this is not correct yet since only taking end of sequence

        # Now turn the aggregate activations into mean activations by dividing by the number of each unique input/context instance
        for i in range(counter.shape[0]):
            if counter[i]==0:
                counter[i]=1  # prevent divide by zero
                print('Warning: index ' + str(i) + ' input had no instances?')

        activations = np.divide(aggregate_activations, counter)

    # finally, reshape the output activations and labels so that we can easily interpret RSA on the activations

    # sort all variables first by context order
    context_ind = np.argsort(contexts, axis=0)
    contexts = np.take_along_axis(contexts, context_ind, axis=0)
    activations = np.take_along_axis(activations, context_ind, axis=0)
    MDSlabels = np.take_along_axis(MDSlabels, context_ind, axis=0)
    labels_refValues = np.take_along_axis(labels_refValues, context_ind, axis=0)
    labels_judgeValues = np.take_along_axis(labels_judgeValues, context_ind, axis=0)
    time_index = np.take_along_axis(time_index, context_ind, axis=0)
    counter = np.take_along_axis(counter, context_ind, axis=0)

    # within each context, sort according to numerosity of the judgement value
    for context in range(1,4):
        ind = [i for i in range(contexts.shape[0]) if contexts[i]==context]
        numerosity_ind = np.argsort(labels_judgeValues[ind], axis=0) + ind[0]
        labels_judgeValues[ind] = np.take_along_axis(labels_judgeValues, numerosity_ind, axis=0)
        labels_refValues[ind] = np.take_along_axis(labels_refValues, numerosity_ind, axis=0)
        contexts[ind] = np.take_along_axis(contexts, numerosity_ind, axis=0)
        MDSlabels[ind] = np.take_along_axis(MDSlabels, numerosity_ind, axis=0)
        activations[ind] = np.take_along_axis(activations, numerosity_ind, axis=0)
        time_index[ind] = np.take_along_axis(time_index, numerosity_ind, axis=0)
        counter[ind] = np.take_along_axis(counter, numerosity_ind, axis=0)

    drift = {"temporal_activation_drift":temporal_activation_drift, "temporal_context":temporal_context}

    return activations, MDSlabels, labels_refValues, labels_judgeValues, contexts, time_index, counter, drift, temporal_trialtypes

# ---------------------------------------------------------------------------- #

class separateinputMLP(nn.Module):
    """
        This is a simple 3-layer MLP which compares the magnitude of input nodes A (4) to input nodes B (4)
        """
    def __init__(self, D_in):
        super(separateinputMLP, self).__init__()
        self.hidden_size = 500   # was 60, now increase to 500 to prevent bottleneck in capacity.
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

    def __init__(self, D_in, D_out, noise_std, recurrent_size, hidden_size):
        super(OneStepRNN, self).__init__()
        self.recurrent_size = recurrent_size  # was 33 default to match to the parallel MLP; now larger to prevent bottleneck on context rep
        self.hidden_size = hidden_size   # was 60 default
        self.hidden_noise = noise_std
        self.input2hidden = nn.Linear(D_in + self.recurrent_size, self.recurrent_size)
        self.input2fc1 = nn.Linear(D_in + self.recurrent_size, self.hidden_size)  # size input, size output
        self.fc1tooutput = nn.Linear(self.hidden_size, 1)

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
        parser.add_argument('--lr-multi', nargs='*', type=float, help='learning rate (or list of learning rates) (default: 0.001)', default=[0.00005])
        parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: 48)')
        parser.add_argument('--test-batch-size', type=int, default=1, metavar='N', help='input batch size for testing (default: 48)')
        parser.add_argument('--epochs', type=int, default=6, metavar='N', help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.001)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
        parser.add_argument('--weight_decay', type=int, default=0.0000, metavar='N', help='weight-decay for l2 regularisation (default: 0)')
        parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
        parser.add_argument('--recurrent-size', type=int, default=200, metavar='N', help='number of nodes in recurrent layer (default: 33)')
        parser.add_argument('--hidden-size', type=int, default=200, metavar='N', help='number of nodes in hidden layer (default: 60)')
        parser.add_argument('--BPTT-len', type=int, default=120, metavar='N', help='length of sequences that we backprop through (default: 120 = whole block length)')
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
        self.recurrent_size = 33
        self.hidden_size = 60
        self.BPTT_len = 120

# ---------------------------------------------------------------------------- #

def getDatasetName(args, networkStyle, noise_std, blockTrain, seqTrain, labelContext, retainHiddenState, allFullRange):

    # conver the hyperparameter settings into a string ID
    str_args = '_bs'+ str(args.batch_size_multi[0]) + '_lr' + str(args.lr_multi[0]) + '_ep' + str(args.epochs) + '_r' + str(args.recurrent_size) + '_h' + str(args.hidden_size) + '_bpl' + str(args.BPTT_len)

    networkTxt = 'RNN' if networkStyle == 'recurrent' else 'MLP'
    if blockTrain:
        blockedtext = '_blck'
    else:
        blockedtext = ''
    if seqTrain:
        seqtext = '_seq'
    else:
        seqtext = ''
    if retainHiddenState:
        hiddenstate = '_retainstate'
    else:
        hiddenstate = '_resetstate'
    if labelContext=='true':
        contextlabelledtext = '_truecontextlabel'
    elif labelContext=='random':
        contextlabelledtext = '_randcontextlabel'
    elif labelContext=='constant':
        contextlabelledtext = '_constcontextlabel'

    rangetxt = '_numrangeintermingled' if allFullRange else '_numrangeblocked'


    datasetname = 'dataset'+contextlabelledtext+blockedtext+seqtext+rangetxt + '_bpl' + str(args.BPTT_len)
    analysis_name = 'network_analysis/'+'MDSanalysis_'+networkTxt+contextlabelledtext+blockedtext+seqtext+rangetxt+hiddenstate+'_n'+str(noise_std)+str_args

    if networkStyle=='recurrent':
        trained_modelname = 'models/'+networkTxt+'_trainedmodel'+contextlabelledtext+blockedtext+seqtext+rangetxt+hiddenstate+'_n'+str(noise_std)+str_args+'.pth'
    else:
        trained_modelname = 'models/'+networkTxt+'_trainedmodel'+contextlabelledtext+blockedtext+seqtext+rangetxt+hiddenstate+str_args+'.pth'

    trainingrecord_name = '_trainingrecord_'+ networkTxt + contextlabelledtext+blockedtext+seqtext+rangetxt+hiddenstate+'_n'+str(noise_std)+str_args

    return datasetname, trained_modelname, analysis_name, trainingrecord_name

# ---------------------------------------------------------------------------- #

def trainMLPNetwork(args, device, multiparams, trainset, testset, N, params):
    """This function performs the train/test loop for different parameter settings
     input by the user in multiparams.
     - Train/test performance is logged with a SummaryWriter
     - the trained model is returned
     """

    _, _, _, trainingrecord_name = getDatasetName(args, *params)

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
        writer = SummaryWriter(log_dir='results/runs/' + trainingrecord_name + args.modeltype + date + comment)
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
            printProgress(epoch, n_epochs)

        print("Training complete.")

    writer.close()
    return model

# ---------------------------------------------------------------------------- #

def trainRecurrentNetwork(args, device, multiparams, trainset, testset, N, params):
    """This function performs the train/test loop for different parameter settings
     input by the user in multiparams.
     - Train/test performance is logged with a SummaryWriter
     - the trained recurrent model is returned
     - note that the train and test set must be divisible by args.batch_size, do to the shaping of the recurrent input
     """
    _, noise_std, _, _, _, retainHiddenState, _ = params
    _, _, _, trainingrecord_name = getDatasetName(args, *params)

    # Repeat the train/test model assessment for different sets of hyperparameters
    for batch_size, lr in product(*multiparams):
        args.batch_size = batch_size
        args.test_batch_size = batch_size
        args.lr = lr
        print("Network training conditions: ")
        print(args)
        print("\n")

        # Define a model for training
        #torch.manual_seed(1)         # if we want the same default weight initialisation every time
        model = OneStepRNN(N+3+1, 1, noise_std, args.recurrent_size, args.hidden_size).to(device)

        criterion = nn.BCELoss() #nn.CrossEntropyLoss()   # binary cross entropy loss
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Define our dataloaders
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

        # Log the model on TensorBoard and label it with the date/time and some other naming string
        now = datetime.now()
        date = now.strftime("_%d-%m-%y_%H-%M-%S")
        comment = "_batch_size-{}_lr-{}_epochs-{}_wdecay-{}".format(args.batch_size, args.lr, args.epochs, args.weight_decay)
        writer = SummaryWriter(log_dir='results/runs/' + trainingrecord_name + args.modeltype + date + comment)
        print("Open tensorboard in another shell to monitor network training (hannahsheahan$  tensorboard --logdir=runs)")

        # Train/test loop
        n_epochs = args.epochs
        printOutput = False
        trainingPerformance, testPerformance = [[] for i in range(2)]

        print("Training network...")

        # Take baseline performance measures
        optimizer.zero_grad()
        _, base_train_accuracy = recurrent_test(args, model, device, trainloader, criterion, retainHiddenState, printOutput)
        _, base_test_accuracy = recurrent_test(args, model, device, testloader, criterion, retainHiddenState, printOutput)
        print('Baseline train: {:.2f}%, Baseline test: {:.2f}%'.format(base_train_accuracy, base_test_accuracy))
        trainingPerformance.append(base_train_accuracy)
        testPerformance.append(base_test_accuracy)
        printProgress(0, n_epochs)

        for epoch in range(1, n_epochs + 1):  # loop through the whole dataset this many times

            # train network
            standard_train_loss, standard_train_accuracy = recurrent_train(params, args, model, device, trainloader, optimizer, criterion, epoch, retainHiddenState, printOutput)

            # assess network
            fair_train_loss, fair_train_accuracy = recurrent_test(args, model, device, trainloader, criterion, retainHiddenState, printOutput)
            test_loss, test_accuracy = recurrent_test(args, model, device, testloader, criterion, retainHiddenState, printOutput)

            # log performance
            train_perf = [standard_train_loss, standard_train_accuracy, fair_train_loss, fair_train_accuracy]
            test_perf = [test_loss, test_accuracy]
            trainingPerformance.append(standard_train_accuracy)
            testPerformance.append(test_accuracy)
            print('Train: {:.2f}%, Test: {:.2f}%'.format(standard_train_accuracy, test_accuracy))
            logPerformance(writer, epoch, train_perf, test_perf)
            printProgress(epoch, n_epochs)

        print("Training complete.")
        # save this training curve
        record = {"trainingPerformance":trainingPerformance, "testPerformance":testPerformance, "args":vars(args), "model":"recurrent_truecontext" }
        randnum = str(random.randint(0,10000))
        dat = json.dumps(record)
        f = open("trainingrecords/"+randnum + trainingrecord_name+".json","w")
        f.write(dat)
        f.close()

    writer.close()
    return model

# ---------------------------------------------------------------------------- #
