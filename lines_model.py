"""
# Build a model that captures the basic structure of the neural (EEG brain) and rnn representations
# generated when performing a relational magnitude task.

# Author: Hannah Sheahan
# Date: 27/04/2020
# Issues: N/A
# Notes:
# - neural EEG task designed and data collected by Fabrice Luyckx
# - Fabrice built a model like this in matlab, this is an independent attempt to build a similar model in python
# - in early development (27/04/2020)
"""
# ---------------------------------------------------------------------------- #
from __future__ import division

import numpy as np
import sklearn
import scipy.optimize as optimize
import scipy.stats as stats
import math
import os
import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
import constants as const
import MDSplotter as MDSplt
import random

from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
import matplotlib.colors as mplcol
import sklearn.decomposition

RNN_BASEPATH = 'lines_model/rnn_RDMs_nolesion/' #'lines_model/rnn_RDMs_nolesion/' #'lines_model/rnn_RDMs/'
EEG_BASEPATH = 'lines_model/brain_RDMs/'
EEG_FILE = 'avRDM_magn.mat'
contextcolours = ['dodgerblue', 'orangered', np.asarray([1.0, 0.69803921, 0.0, 1.0]), 'black']   # 1-16, 1-11, 6-16 like fabrices colours

# ---------------------------------------------------------------------------- #
def cmdscale(D):
    """
    Classical multidimensional scaling (MDS)
    Author: Francis Song; song.francis@gmail.com
    Parameters
    ----------
    D : (n, n) array
        Symmetric distance matrix.
    Returns
    -------
    Y : (n, p) array
        Configuration matrix. Each column represents a dimension. Only the
        p dimensions corresponding to positive eigenvalues of B are returned.
        Note that each dimension is only determined up to an overall sign,
        corresponding to a reflection.

    e : (n,) array
        Eigenvalues of B.
    """
    # Number of points
    n = len(D)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n))/n

    # YY^T
    B = -H.dot(D**2).dot(H)/2

    # Diagonalize
    evals, evecs = np.linalg.eigh(B)

    # Sort by eigenvalue in descending order
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)

    return Y, evals

# ---------------------------------------------------------------------------- #
def import_EEG_data(basepath, filename):
    """ import the .mat eeg data for all subjects """

    filepath = os.path.join(basepath, filename)
    with open(filepath) as eeg_file:
        eeg_activations = loadmat(filepath)
        eeg_activations = np.asarray(eeg_activations['magndat'])
        eeg_activations = np.transpose(eeg_activations, (2,0,1))

    return eeg_activations          # subjects x condition x condition [low -> high -> full]
# ---------------------------------------------------------------------------- #

def import_RNN_data(basepath, args):
    """ import the rnn activations for each of ten models we care about"""
    rnn_files = os.listdir(basepath)
    rnn_files = [x for x in rnn_files if (args.blocking in x) and (args.context_label in x) ]
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
    rnn_ordered_activations = np.concatenate((low_activations, high_activations, full_activations), axis=1)

    return rnn_ordered_activations   # correlation distance:  [model instance x conditions x activity]
# ---------------------------------------------------------------------------- #

class network_args():
    def __init__(self, blocking, label, train_lesion_freq):
        self.blocking = blocking
        self.context_label = label
        self.network_style = 'recurrent'
        self.all_fullrange = False
        self.label_context = 'true'
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

# ---------------------------------------------------------------------------- #
def plotRDM(activations, args, label, saveFig):
    np.fill_diagonal(activations, None) # ignore the diagonal
    fig = plt.figure(figsize=(5,3))
    ax = plt.gca()
    im = plt.imshow(activations, zorder=2, cmap='viridis', interpolation='nearest')

    cbar = fig.colorbar(im)
    cbar.set_label('disimilarity')
    ax.set_title('Averaged activations')
    labelticks = ['25-35', '30-40', '25-40']
    ticks = [0, const.LOWR_SPAN, const.HIGHR_SPAN+const.LOWR_SPAN]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labelticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labelticks)
    MDSplt.autoSaveFigure('figures/'+label, args, False, False, 'compare', saveFig)

# ---------------------------------------------------------------------------- #

def rotate_axes(x,y,theta):
    # theta is in degrees
    theta_rad = theta * (math.pi/180)  # convert to radians
    x_new = x * math.cos(theta_rad) + y * math.sin(theta_rad)
    y_new =  -x * math.sin(theta_rad) + y * math.cos(theta_rad)
    return x_new, y_new

# ---------------------------------------------------------------------------- #

def plot_components(data, theta=0, axislimits=None):

    fig,ax = plt.subplots(1,3, figsize=(18,5))
    rbg_contextcolours = [mplcol.to_rgba(i) for i in contextcolours]
    white = (1.0, 1.0, 1.0, 1.0)
    contextlabel = np.zeros((data.shape[0],))
    contextlabel[11:11+11] = 1
    contextlabel[11+11:] = 2
    numberlabel = [[i for i in range(const.LOWR_LLIM,const.LOWR_ULIM+1)],[i for i in range(const.HIGHR_LLIM,const.HIGHR_ULIM+1)], [i for i in range(const.FULLR_LLIM,const.FULLR_ULIM+1)]]
    numberlabel = [i+24 for sublist in numberlabel for i in sublist]

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

        ax[j].set_title('context')

        contextA = range(const.LOWR_SPAN)
        contextB = range(const.LOWR_SPAN,const.LOWR_SPAN+const.HIGHR_SPAN)
        contextC = range(const.LOWR_SPAN+const.HIGHR_SPAN, const.FULLR_SPAN+const.LOWR_SPAN+const.HIGHR_SPAN)
        x,y = rotate_axes(data[contextA, dimA], data[contextA, dimB], theta)
        ax[j].plot(x, y, color=contextcolours[0])
        x,y = rotate_axes(data[contextB, dimA], data[contextB, dimB], theta)
        ax[j].plot(x,y, color=contextcolours[1])
        x,y = rotate_axes(data[contextC, dimA], data[contextC, dimB], theta)
        ax[j].plot(x,y, color=contextcolours[2])

        Ns = [11,11,16]
        markercount=0
        lastc = -1
        for i in range((data.shape[0])):

            # create colour gradient within each context to signal numerosity
            c = int(contextlabel[i])
            if c!=lastc:
                markercount=0
            lastc = int(contextlabel[i])
            graded_contextcolours = np.zeros((4, Ns[c]))
            for p in range(4):
                graded_contextcolours[p] = np.linspace(white[p],rbg_contextcolours[c][p],Ns[c])
            gradedcolour = np.asarray([graded_contextcolours[p][markercount] for p in range(len(graded_contextcolours))])

            # colour by context
            x,y = rotate_axes(data[i, dimA], data[i, dimB], theta)
            ax[j].scatter(x,y, color=gradedcolour, edgecolor=contextcolours[int(contextlabel[i])], s=80, linewidths=2)
            #ax[j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=contextcolours[int(contextlabel[i])], edgecolor=contextcolours[int(contextlabel[i])], s=80, linewidths=2)
            markercount +=1
            # label numerosity in white inside the marker
            firstincontext = [0,10,11,11+10,11+11, 11+11+15]
            if i in firstincontext:
                ax[j].text(x,y, str(int(numberlabel[i])), color=contextcolours[int(contextlabel[i])], size=15, horizontalalignment='center', verticalalignment='center')

        ax[j].axis('equal')
        if axislimits is not None:
            ax[j].set(xlim=axislimits, ylim=axislimits)
    return fig

# --------------------------------------------------------------------------- #

def plotMDS(pairwise_data, args, label, theta, axislimits, saveFig):
    """ plot the first 3 MDS dims, separating by /underlying context colour and showing
    graded colour opacity scaling with numerosity.
    """

    # old method which produces weird results for EEG
    #randseed = 3   # so that we get the same MDS each time
    #embedding = MDS(n_components=3, random_state=randseed, dissimilarity='precomputed', max_iter=5000)
    #MDS_act = embedding.fit_transform(pairwise_data)

    # new method which does what matlab does in cmdscale
    np.fill_diagonal(np.asarray(pairwise_data), 0)
    MDS_act, evals = cmdscale(pairwise_data)
    fig = plot_components(MDS_act, theta, axislimits)
    n = MDSplt.autoSaveFigure('figures/'+label, args, False, False, 'compare', saveFig)

# ---------------------------------------------------------------------------- #

def plotPCA(data, args, label, saveFig):
    pca = sklearn.decomposition.PCA(n_components=3)
    PCs = pca.fit_transform(data)
    fig = plot_components(PCs)
    MDSplt.autoSaveFigure('figures/'+label, args, False, False, 'compare', saveFig)

# ---------------------------------------------------------------------------- #
def generatePlots(rnn_RDM, eeg_RDM, rnn_args, eeg_args, saveFig):

    # PCA plots
    #plotPCA(rnn_RDM, rnn_args, 'rnn_pca_', saveFig)
    #plotPCA(eeg_RDM, eeg_args, 'eeg_pca_', saveFig)

    # MDS plots
    plotMDS(rnn_RDM, rnn_args, 'rnn_mds_', 90, (-0.65,0.65), saveFig)
    plotMDS(eeg_RDM, eeg_args, 'eeg_mds_',  -20, (-0.35,0.35), saveFig)

    # RDM matrices
    plotRDM(rnn_RDM, rnn_args, 'rnn_rdm_', saveFig)
    plotRDM(eeg_RDM, eeg_args, 'eeg_rdm_', saveFig)

# ---------------------------------------------------------------------------- #

def generate_lines(params):
    """Build 3x parallel lines in 3d based on the parameters in params"""
    xB,yB,zB, xC,yC,zC, lenLong, lenShort = params
    x0,y0,z0 = [0,0,0]
    npoints_long = 16
    npoints_short = 11
    allow_centre_offsets = True
    keep_parallel = True

    # define centre of each line as free point in 3d space
    if allow_centre_offsets:
        centre_A = [x0,y0,z0]  # coordinate origin for centre of line A, long line
        centre_B = [xB,yB,zB]
        centre_C = [xC,yC,zC]

    # align axes with the coordinate system for defining the long line
    if keep_parallel:
        lineA_direction = [1,0,0]
        lineB_direction = [1,0,0]
        lineC_direction = [1,0,0]

    # each point on the line to be equally spaced (big assumption)
    # constrain lengths of lines B and C (short) to be the same
    points_A = np.zeros((npoints_long,3))
    points_B = np.zeros((npoints_short,3))
    points_C = np.zeros((npoints_short,3))
    for i in range(3):
        directed_points = lineA_direction[i] *lenLong/2
        points_A[:,i] = np.linspace(centre_A[i]-directed_points, centre_A[i]+directed_points, num=npoints_long)

        directed_points = lineB_direction[i] *lenShort/2
        points_B[:,i] = np.linspace(centre_B[i]-directed_points, centre_B[i]+directed_points, num=npoints_short)

        directed_points = lineC_direction[i] *lenShort/2
        points_C[:,i] = np.linspace(centre_C[i]-directed_points, centre_C[i]+directed_points, num=npoints_short)

    # calculate the correlation RDM for these lines
    return np.concatenate((points_B, points_C, points_A), axis=0)  # low -> high -> full

# ---------------------------------------------------------------------------- #

def makelinesmodel(metric, xB,yB,zB, xC,yC,zC, lenLong, lenShort):
    """Construct a model of 3 lines (parallel or free) in 3d space"""
    # we will use the x (first) input to signal the distance metric (unconventional but we dont need to evaluate as a function of x)
    params = [xB,yB,zB, xC,yC,zC, lenLong, lenShort]
    all_lines = generate_lines(params)
    model_RDM = pairwise_distances(all_lines, metric=metric)

    return model_RDM.flatten()

# ---------------------------------------------------------------------------- #
def nanArray(size):
    # create a numpy array of size 'size' (tuple) filled with nans
    tmp = np.zeros(size)
    tmp.fill(np.nan)
    return tmp

# ---------------------------------------------------------------------------- #
def fitmodelRDM(data, numiter):
    """Fit the model using euclidean distance (because its the only one that really makes sense for a deterministic lines model)"""
    metric = 'euclidean'
    fitted_params = nanArray((numiter,8))
    fitted_SSE = nanArray((numiter,1))

    for i in range(numiter):
        # Randomise the initial parameter starting point for those we are fitting (first iteration will just use our guesses)
        init_params = [random.random(),random.random(),random.random(),\
                       random.random(),random.random(),random.random(),\
                       random.random(), random.random()]
        try:
            params, params_covariance = optimize.curve_fit(makelinesmodel, metric, data, p0=init_params)
            SSE = sum((makelinesmodel(metric, *params) - data)**2)
            fitted_params[i] = params
            fitted_SSE[i] = SSE

        except RuntimeError:
            print("\n Error: max number of fit iterations hit. Moving on...")

    return fitted_params, fitted_SSE

# ---------------------------------------------------------------------------- #

def bestfitRDM(data):
    flat_data = data.flatten()
    fitted_params, fitted_SSE = fitmodelRDM(flat_data, 100)
    opt_iter = np.nanargmin(fitted_SSE)
    opt_params = fitted_params[opt_iter][:]
    print('Best-fit parameters:')
    print(opt_params)

    # generate euclidean distance RDM under these best fit parameters
    bestfit_rdm = makelinesmodel('euclidean', *opt_params)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(flat_data.reshape(38,-1).T)
    ax[0].set_title('rnn data RDM')
    ax[1].imshow(bestfit_rdm.reshape(38,-1).T)
    ax[1].set_title('best fit RDM')
    plt.savefig('figures/rdm_fitting_test.pdf',bbox_inches='tight')

    divisivenorm_ratio = opt_params[-1]/opt_params[-2]
    print('Best-fit divisive normalisation ratio: {}'.format(divisivenorm_ratio))
    print('where scale is 0.6875 (abs) -> 1 (norm)')
    subtractive_ratio = np.mean((np.abs(opt_params[0]), np.abs(opt_params[3])))/(opt_params[-2]-opt_params[-1])
    print('Best-fit subtractive normalisation ratio (1=totally offset, 0=totally centred): {:.2f} '.format(subtractive_ratio))

    return opt_params, subtractive_ratio, divisivenorm_ratio
# ---------------------------------------------------------------------------- #


# saving settings
saveFig = True
RNN_args = network_args('blocked', 'truecontext', 0.0)
EEG_args = network_args('blocked', 'truecontext', 0.0)

# import rnn representations data
RNN_data = import_RNN_data(RNN_BASEPATH, RNN_args)
mean_RNN_data = np.mean(RNN_data, axis=0)  # mean across model instances
mean_RNN_RDM = pairwise_distances(mean_RNN_data, metric='correlation')
mean_RNN_euclRDM = pairwise_distances(mean_RNN_data, metric='euclidean')

# import the eeg representations data
EEG_data = import_EEG_data(EEG_BASEPATH, EEG_FILE)
mean_EEG_RDM = np.mean(EEG_data, axis=0)

# plot low dimensional representations of the rnn and eeg data
#generatePlots(mean_RNN_RDM, mean_EEG_RDM, RNN_args, EEG_args, saveFig)



# construct a lines model
# the lines model will take some parameters and generate 3 lines in 3D space
# it will then generate a (correlation distance) RDM based on those lines
# we will then fit the correlation distance model RDM to the actual (rnn or eeg) data


# generate a fake rdm to test our fitting on
#fake_params = [1,0,0, -1,-1,0, 11, 11]
#fake_rdm = makelinesmodel('euclidean', *fake_params)

rnn_fitparams = np.zeros((RNN_data.shape[0], 8))
sub_ratios = np.zeros((RNN_data.shape[0],))
div_ratios = np.zeros((RNN_data.shape[0],))
for i in range(RNN_data.shape[0]):
    subject_RNN_eclRDM = RNN_data[i]
    subject_RNN_eclRDM = pairwise_distances(subject_RNN_eclRDM, metric='euclidean')
    np.fill_diagonal(np.asarray(subject_RNN_eclRDM), 0)
    opt_params, subtractive_ratio, divisivenorm_ratio = bestfitRDM(subject_RNN_eclRDM)
    rnn_fitparams[i] = opt_params
    sub_ratios[i] = subtractive_ratio
    div_ratios[i] = divisivenorm_ratio

print('subtractive ratios:')
print(sub_ratios)
print('divisive ratios:')
print(div_ratios)

print('stat. significance:')
[t,p] = stats.ttest_1samp(sub_ratios, 1)  # is there significant subtractive normalisation vs totally offset?
print('subtractive normalisation: p={}'.format(p))
[t,p] = stats.ttest_1samp(div_ratios, 0.6875)  # is there significant divisive normalisation vs totally absolute?
print('divisive normalisation: p={}'.format(p))


# Looks like for the unlesioned model its totally absolute and for the lesioned model its halfway between abs and normalised

# Now take the best fit parameters for the lines model, generate a *correlation* distance RDM from it,
# and visualize that fit on top of the correlation distance data
#bestfit_lines = generate_lines(opt_params)
#fig = plot_components(bestfit_lines)
#MDSplt.autoSaveFigure('figures/bestfit', RNN_args, False, False, 'compare', saveFig)


# ---------------------------------------------------------------------------- #
