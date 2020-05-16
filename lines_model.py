"""
# Build a model that captures the basic structure of the neural (EEG brain) and rnn representations
# generated when performing a relational magnitude task.

# Author: Hannah Sheahan
# Date: 27/04/2020
# Issues: N/A
# Notes:
# - neural EEG task designed and data collected by Fabrice Luyckx
# - both Fabrice and Steph built models like this in matlab, this is an independent attempt to build a similar model in python
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
import copy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import constants as const
import MDSplotter as MDSplt
import random

from scipy.spatial import procrustes
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import correlation
from sklearn.manifold import MDS
#from astropy.stats import rayleightest
from pycircstat.tests import rayleigh

import matplotlib.colors as mplcol
import sklearn.decomposition
from matplotlib import animation
from mpl_toolkits import mplot3d

RNN_BASEPATH = 'lines_model/rnn_RDMs_nolesion/' #'lines_model/rnn_RDMs_nolesion/' #'lines_model/rnn_RDMs/'
EEG_BASEPATH = 'lines_model/brain_RDMs/'
EEG_FILE = 'avRDM_magn.mat'
SAVE_FILELOC = 'linesmodel_parameters/'
contextcolours = ['dodgerblue', 'orangered', np.asarray([1.0, 0.69803921, 0.0, 1.0]), 'black']   # 1-16, 1-11, 6-16 like fabrices colours
modelcolours = ['mediumblue','saddlebrown','darkgoldenrod']

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

def import_all_data(metric, RNN_args):
    # import rnn activations data
    RNN_data = import_RNN_data(RNN_BASEPATH, RNN_args)
    mean_RNN_data = np.mean(RNN_data, axis=0)  # mean across model instances
    mean_RNN_data = stats.zscore(mean_RNN_data, axis=None)  # z-score the RNN raw data

    # compute RDM for RNN data
    mean_RNN_RDM = pairwise_distances(mean_RNN_data, metric=metric)
    np.fill_diagonal(np.asarray(mean_RNN_RDM), 0)

    # for each model instance, zscore the RNN data and compute a similarity matrix
    subjects_RNN_RDMs = np.zeros((RNN_data.shape[0], RNN_data.shape[1], RNN_data.shape[1]))
    for i in range(RNN_data.shape[0]):
        instance_RNN_data = RNN_data[i]
        instance_RNN_data = stats.zscore(instance_RNN_data, axis=None)  # z-score the RNN raw data
        instance_RNN_RDM = pairwise_distances(instance_RNN_data, metric=metric)
        np.fill_diagonal(np.asarray(instance_RNN_RDM), 0)
        subjects_RNN_RDMs[i] = instance_RNN_RDM

    # import the eeg activations data (already in correlation distance)
    subjects_EEG_RDMs = import_EEG_data(EEG_BASEPATH, EEG_FILE)
    mean_EEG_RDM = np.mean(subjects_EEG_RDMs, axis=0)

    return mean_RNN_RDM, mean_EEG_RDM, subjects_RNN_RDMs, subjects_EEG_RDMs

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

def plot_components(data, theta=0, axislimits=None, ax=None):

    if ax is None:
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

def unit_vector(x):
    # create a unit vector from the input array
    x = np.asarray(x)
    normed_x = x / np.linalg.norm(x)
    normed_x = list(normed_x)
    return normed_x

# ---------------------------------------------------------------------------- #

def reconstrain_params(params, fit_args):
    """ This forces back the constraints on the parameters that exist within the fitting function generate_lines(),
     in case these values deviated during fitting (but dont worry they wont have affected SSE). This is just for parameter interpretation
     because even generating predictions will reconstrain these lines.
     ***HRS note that this could be put inside generate_lines() for better practice to ensure they are they same"""

    _, keep_parallel, div_norm, sub_norm, _, _, lenShort = fit_args

    if sub_norm == 'centred':
        params[0] = 0      # set this line-centre parameter to zero
        params[3] = 0      # set this line-centre parameter to zero
    elif sub_norm == 'offset':
        lenLong = lenShort*params[6]
        mag_centre_low = -2.5 * (lenLong/const.N_POINTS_LONG)
        mag_centre_high = -mag_centre_low
        params[0] = mag_centre_low
        params[3] = mag_centre_high

    if div_norm == 'normalised':
        params[6] = 1
    elif div_norm == 'absolute':
        params[6] = const.N_POINTS_LONG/const.N_POINTS_SHORT

    if keep_parallel:
        params[7:] = [1,0,0,1,0,0]  # set lines B and C parallel to line A

    # normalise the direction of the fitted lines B and C directions for interpretability
    params[7:10] = unit_vector(params[7:10])
    params[10:] = unit_vector(params[10:])

    return params

# ---------------------------------------------------------------------------- #

def generate_lines(params, fit_args):
    """Build 3x parallel lines in 3d based on the parameters in params"""
    xB,yB,zB, xC,yC,zC, lenRatio, dirxB, diryB, dirzB, dirxC, diryC, dirzC = params
    _, keep_parallel, div_norm, sub_norm, _, _, lenShort = fit_args

    x0,y0,z0 = [0,0,0]

    # align axes with the coordinate system for defining the long line
    if keep_parallel:
        lineA_direction = [1,0,0]
        lineB_direction = [1,0,0]
        lineC_direction = [1,0,0]
    else:
        lineA_direction = [1,0,0]
        lineB_direction = [dirxB, diryB, dirzB]
        lineC_direction = [dirxC, diryC, dirzC]

    # normalise the line directions
    lineA_direction = unit_vector(lineA_direction)
    lineB_direction = unit_vector(lineB_direction)
    lineC_direction = unit_vector(lineC_direction)

    # constrain divisive normalisation inside the function
    if div_norm == 'normalised':
        lenRatio = 1  # line length ratio must be perfectly equal to 1 i.e. totally normalised
    elif div_norm == 'absolute':
        lenRatio = const.N_POINTS_LONG/const.N_POINTS_SHORT # line length ratio must be proportional to numerical range
    elif div_norm == 'unconstrained':
        pass
    lenLong = lenShort*lenRatio

    # constrain subtractive normalisation inside the function
    if sub_norm == 'centred':
        centre_A = [x0,y0,z0]
        centre_B = [x0,yB,zB]
        centre_C = [x0,yC,zC]
    elif sub_norm == 'offset':
        # constrain the centre of the three lines to be perfectly offset in the magnitude dimension
        mag_centre_low = -2.5 * (lenLong/const.N_POINTS_LONG)
        mag_centre_high = -mag_centre_low
        centre_A = [x0,y0,z0]
        centre_B = [mag_centre_low,yB,zB]
        centre_C = [mag_centre_high,yC,zC]
    elif sub_norm == 'unconstrained':
        centre_A = [x0,y0,z0]  # coordinate origin for centre of line A, long line
        centre_B = [xB,yB,zB]
        centre_C = [xC,yC,zC]

    # each point on the line to be equally spaced (big assumption)
    # constrain lengths of lines B and C (short) to be the same
    points_A = np.zeros((const.N_POINTS_LONG,3))
    points_B = np.zeros((const.N_POINTS_SHORT,3))
    points_C = np.zeros((const.N_POINTS_SHORT,3))
    for i in range(3):
        directed_points = lineA_direction[i] *lenLong/2
        points_A[:,i] = np.linspace(centre_A[i]-directed_points, centre_A[i]+directed_points, num=const.N_POINTS_LONG)

        directed_points = lineB_direction[i] *lenShort/2
        points_B[:,i] = np.linspace(centre_B[i]-directed_points, centre_B[i]+directed_points, num=const.N_POINTS_SHORT)

        directed_points = lineC_direction[i] *lenShort/2
        points_C[:,i] = np.linspace(centre_C[i]-directed_points, centre_C[i]+directed_points, num=const.N_POINTS_SHORT)

    # calculate the correlation RDM for these lines
    return np.concatenate((points_B, points_C, points_A), axis=0)  # low -> high -> full

# ---------------------------------------------------------------------------- #

def makelinesmodel(fit_args, xB,yB,zB, xC,yC,zC, lenRatio, dirxB,diryB,dirzB, dirxC,diryC,dirzC):
    """Construct a model of 3 lines (parallel or free) in 3d space
    - dummy is an unused dummy variable"""
    # we will use the x (first) input to signal the distance metric (unconventional but we dont need to evaluate as a function of x)
    params = [xB,yB,zB, xC,yC,zC, lenRatio, dirxB,diryB,dirzB, dirxC,diryC,dirzC]
    all_lines = generate_lines(params, fit_args)
    all_lines = stats.zscore(all_lines, axis=None) # zscore the lines model before computing similarity matrix
    model_RDM = pairwise_distances(all_lines, metric='euclidean')

    #zscore the model RDM to make sure its on same scale as data
    model_RDM = stats.zscore(model_RDM, axis=None)
    model_uppertriRDM, _ = matrix_to_uppertri(model_RDM)  # just use the upper triangle of matrix for fitting

    return model_uppertriRDM

# ---------------------------------------------------------------------------- #

def nanArray(size):
    # create a numpy array of size 'size' (tuple) filled with nans
    tmp = np.zeros(size)
    tmp.fill(np.nan)
    return tmp

# ---------------------------------------------------------------------------- #

def fitfunc(params, fit_args, data):
    """Fit lines model to data using a different function
       - for passing into scipy.optimize.minimize
       - returns a scalar loss (SSE)
    """
    modelRDM_uppertri = makelinesmodel(fit_args, *params)
    loss = sum((modelRDM_uppertri - data)**2)
    return loss

# ---------------------------------------------------------------------------- #

def fitmodelRDM(data, fit_args):
    """Fit the model using euclidean distance (because its the only one that really makes sense for a deterministic lines model)"""

    uppertri_data, ind = data
    cost_function, keep_parallel, div_norm, sub_norm, n_iter, metric, lenShort = fit_args

    n_params = 13
    fitted_params = nanArray((n_iter,n_params))
    fitted_loss = nanArray((n_iter,1))

    # Note that introducing constraints into optimize.minimize() forces the use of a different solver that doesnt fit as well,
    # so instead we will test our different (constrained) hypotheses by fixing some parameters ~within~ the fitting function.
    # Strangely this method results in closer fits (lower SSE).
    for i in range(n_iter):
        #
        try:
            # keep these bounds, they seem reasonable and the fits dont get close to them at all
            lower_bound = [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1]
            upper_bound = [1,   1,  1,  1,  1,  1, 2,  1,  1,  1,  1,  1,  1]

            # Randomise the initial parameter starting point for those we are fitting (first iteration will just use our guesses)
            init_params = [random.uniform(lower_bound[i],upper_bound[i]) for i in range(len(lower_bound))]

            # select a loss function for the fit: either SSE (for euclidean fits) or summed correlation distance
            bounds = [(lower_bound[i],upper_bound[i]) for i in range(len(init_params))]
            result = optimize.minimize(fitfunc, init_params, args=(fit_args, uppertri_data), bounds=bounds )
            params = result.x

            # force back in the constraints on the params in case they deviated during fitting (as repeated in generate_lines())
            params = reconstrain_params(params, fit_args)

            loss = sum((makelinesmodel(fit_args, *params) - uppertri_data)**2)
            fitted_params[i] = params
            fitted_loss[i] = loss

        except RuntimeError:
            print("\n Error: max number of fit iterations hit. Moving on...")

    return fitted_params, fitted_loss

# ---------------------------------------------------------------------------- #

def matrix_to_uppertri(x):
    """Return the flattened elements and indices of the upper triangular portion of a matrix."""
    ind = np.triu_indices(x.shape[0])
    flatx = [x[ind[0][i], ind[1][i]] for i in range(len(ind[0]))]
    flatx = np.asarray(flatx)
    return flatx, ind

# ---------------------------------------------------------------------------- #

def uppertri_to_matrix(x, ind, n):
    """Transform the flattened upper triangular portion of an originally symmetric matrix, back into a full matrix."""
    uppertri = np.zeros((n,n))
    uppertri[ind] = x
    zeroed_diag = np.zeros((n,n)) + uppertri
    np.fill_diagonal(zeroed_diag, 0)
    fullmatrix = uppertri + np.transpose(zeroed_diag)
    return fullmatrix

# ---------------------------------------------------------------------------- #

def bestfitRDM(data, fit_args, model_system, model_id=0):
    # fits a model RDM to the input data RDM
    flat_data = data.flatten()
    uppertri_data, ind = matrix_to_uppertri(data)

    fitted_params, fitted_SSE = fitmodelRDM((uppertri_data, ind), fit_args)
    opt_iter = np.nanargmin(fitted_SSE)
    opt_params = fitted_params[opt_iter][:]
    opt_SSE = np.min(fitted_SSE)
    print('Bestfit SSE: {}'.format(opt_SSE))
    print('Best-fit parameters:')
    print(opt_params)

    # generate euclidean distance RDM under these best fit parameters
    bestfit_rdm_uppertri = makelinesmodel(fit_args, *opt_params)
    bestfit_rdm = uppertri_to_matrix(bestfit_rdm_uppertri, ind, data.shape[0])

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(flat_data.reshape(38,-1).T)
    ax[0].set_title('rnn data RDM')
    ax[1].imshow(bestfit_rdm.reshape(38,-1).T)
    ax[1].set_title('best fit RDM')
    plt.savefig('figures/bestfit_rdm_' + get_model_description(model_id, model_system, fit_args) + '.pdf',bbox_inches='tight')
    plt.close()

    """
    divisivenorm_ratio = opt_params[-1]/opt_params[-2]
    print('Best-fit divisive normalisation ratio: {}'.format(divisivenorm_ratio))
    print('where scale is 0.6875 (abs) -> 1 (norm)')
    subtractive_ratio = np.mean((np.abs(opt_params[0]), np.abs(opt_params[3])))/(opt_params[-2]-opt_params[-1])
    print('Best-fit subtractive normalisation ratio (1=totally offset, 0=totally centred): {:.2f} '.format(subtractive_ratio))
    """

    return opt_params, opt_SSE
# ---------------------------------------------------------------------------- #

def plot_bestmodel_data(mtx1, mtx2, model_system):
    # mtx1 = data procrustes transformed
    # mtx2 = model proucrustes transformed
    fig, ax = plt.subplots(1,3, figsize=(18,5))
    # Plot the data using our pretty plotting function
    plot_components(mtx1, 0, None, ax)

    # Plot the model on top
    for j in range(3):  # 3 MDS dimensions
        if j==0:
            dimA = 0
            dimB = 1
        elif j==1:
            dimA = 0
            dimB = 2
        elif j==2:
            dimA = 1
            dimB = 2

        contextA = range(const.LOWR_SPAN)
        contextB = range(const.LOWR_SPAN,const.LOWR_SPAN+const.HIGHR_SPAN)
        contextC = range(const.LOWR_SPAN+const.HIGHR_SPAN, const.FULLR_SPAN+const.LOWR_SPAN+const.HIGHR_SPAN)
        ax[j].plot(mtx2[contextA, dimA], mtx2[contextA, dimB], color=modelcolours[0])
        ax[j].plot(mtx2[contextB, dimA], mtx2[contextB, dimB], color=modelcolours[1])
        ax[j].plot(mtx2[contextC, dimA], mtx2[contextC, dimB], color=modelcolours[2])
        ax[j].axis('equal')

    plt.savefig('figures/bestfit_procrustes_' + model_system + '.pdf', bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------- #

def animate3DMDS(MDS_data, MDS_model, args, metric, saveFig=True):
    """ This function will plot the numerosity labeled, context-marked MDS projections
     of the hidden unit activations on a 3D plot, animate/rotate that plot to view it
     from different angles and optionally save it as a mp4 file.
    """

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    contextA = range(const.LOWR_SPAN)
    contextB = range(const.LOWR_SPAN,const.LOWR_SPAN+const.HIGHR_SPAN)
    contextC = range(const.LOWR_SPAN+const.HIGHR_SPAN, const.FULLR_SPAN+const.LOWR_SPAN+const.HIGHR_SPAN)
    numberlabel = [[i for i in range(const.LOWR_LLIM,const.LOWR_ULIM+1)],[i for i in range(const.HIGHR_LLIM,const.HIGHR_ULIM+1)], [i for i in range(const.FULLR_LLIM,const.FULLR_ULIM+1)]]
    numberlabel = [i+24 for sublist in numberlabel for i in sublist]

    def init():
        points = [contextA, contextB, contextC] #if labelContext else [contextA]

        for i in range(len(points)):
            # plot the MDS of the data
            ax.scatter(MDS_data[points[i], 0], MDS_data[points[i], 1], MDS_data[points[i], 2], color=contextcolours[i])
            ax.plot(MDS_data[points[i], 0], MDS_data[points[i], 1], MDS_data[points[i], 2], color=contextcolours[i])

            for j in range(len(points[i])):
                if j==0 or j==len(points[i])-1:
                    label = str(int(numberlabel[points[i][j]]))
                    ax.text(MDS_data[points[i][j], 0], MDS_data[points[i][j], 1], MDS_data[points[i][j], 2], label, color='black', size=11, horizontalalignment='center', verticalalignment='center')

            # now plot the MDS of the model predictions
            if MDS_model is not None:
                ax.plot(MDS_model[points[i], 0], MDS_model[points[i], 1], MDS_model[points[i], 2], color=modelcolours[i])

        ax.set_xlabel('MDS dim 1')
        ax.set_ylabel('MDS dim 2')
        ax.set_zlabel('MDS dim 3')
        axislimits = (-0.7, 0.7)
        ax.set(xlim=axislimits, ylim=axislimits, zlim=axislimits)

        return fig,

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,

    # Animate.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)

    # save the animation as an mp4.
    if saveFig:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        strng = MDSplt.autoSaveFigure('animations/MDS_3Danim_linesmodelanddata_'+metric, args, True, False, 'compare', False)
        anim.save(strng+'.mp4', writer=writer)

# ---------------------------------------------------------------------------- #

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'.
       Source:  https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249 """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return angle_radians

# ---------------------------------------------------------------------------- #

def parallelness_test(model_system, allsubjects_params, fit_args):
    """The null hypothesis for line parallelness is that the distribution of angles
      between each bestfit line and each other bestfit line across all our models is uniform.
      We can then do a rayleigh test across subjects/models that tests the distribution of those cosine angles.
      Note that a rayleigh test will just reject the H0 that the angles are uniform around a circle
      (but wont say they are actually parallel, but it should be totally obvious from the distribution of angles plot)."""

    _, keep_parallel, div_norm, sub_norm, _, _, _ = fit_args
    parallel_fig_text = '_keptparallel' if keep_parallel else '_notkeptparallel'
    model_string = parallel_fig_text + '_divnorm' + div_norm + '_subnorm' + sub_norm

    angles = []
    for params in allsubjects_params:
        # the best fitting directions of each line in 3d space
        dir_lineA = np.asarray([1,0,0])
        dir_lineB = params[7:10]
        dir_lineC = params[10:]

        # compute the angles between the unit vectors in the bestfit lines model
        angle_AB = angle_between(dir_lineA, dir_lineB) # line A v B
        angle_BC = angle_between(dir_lineB, dir_lineC) # line B v C

        angles.append(angle_AB)
        angles.append(angle_BC)

    angles = np.asarray(angles)
    plt.figure()
    plt.hist(angles, bins=50)
    ax = plt.gca()
    ax.set_xlim(-np.pi, np.pi)
    plt.xlabel('Angle between lines (degrees)')
    plt.title('Distribution of angle between bestfit lines')
    plt.savefig('figures/hist_angles_between_lines_' + model_system + model_string + '.pdf', bbox_inches='tight')
    plt.close()

    # perform a rayleigh test that the angles come from a uniform circular distribution
    p,z = rayleigh(angles)

    print('-----')
    print('Rayleigh test (distr. angles between bestfit lines):')
    print('p = {:7f}'.format(p))
    print("Rayleigh's z: {:3f}".format(z))
    return p,z

# ---------------------------------------------------------------------------- #

def plot_divisive_normalisation(model_system, allsubjects_params, fit_args):

    _, keep_parallel, div_norm, sub_norm, _, _, _ = fit_args
    parallel_fig_text = '_keptparallel' if keep_parallel else '_notkeptparallel'
    model_string = parallel_fig_text + '_divnorm' + div_norm + '_subnorm' + sub_norm

    ratios = []
    for params in allsubjects_params:
        divisive_norm_ratio = params[6]
        ratios.append(divisive_norm_ratio)

    if len(ratios)>1:
        plt.figure()
        plt.hist(ratios, bins=50)
        ax = plt.gca()
        ax.set_xlim(0.9, 1.6)
        plt.xlabel('Divisive norm ratios (1: totally normalised -> 1.455: totally absolute)')
        plt.title('Bestfit divisive norm ratios: ' + model_system)
        plt.savefig('figures/hist_divnorm_ratios_' + model_system + model_string + '.pdf', bbox_inches='tight')
        plt.close()
    else:
        print('-----')
        print('Divisive norm ratio:   {:3f}'.format(ratios[0]))
        print('(totally normalised: 1 --> totally absolute: 1.45)')

# ---------------------------------------------------------------------------- #

def plot_subtractive_normalisation(model_system, allsubjects_params, fit_args):
    # hasn't been written properly yet HRS, and not necessary except for optional visualisation
    """
    ratios = []
    for params in allsubjects_params:
        subtractive_norm_ratio = params[6]
        ratios.append(divisive_norm_ratio)

    if len(ratios)>1:
        plt.figure()
        plt.hist(ratios, bins=50)
        ax = plt.gca()
        ax.set_xlim(0.9, 1.6)
        plt.xlabel('Divisive norm ratios (1: totally normalised -> 1.455: totally absolute)')
        plt.title('Bestfit divisive norm ratios: ' + model_system)
        plt.savefig('figures/hist_divnorm_ratios_' + model_system + '.pdf', bbox_inches='tight')
    else:
        print('Divisive norm ratio: {}'.format(ratios))
        print('Totally normalised: 1 ---> totally absolute: 1.455')
    """

# ---------------------------------------------------------------------------- #

def fit_and_plot_model(data, model_system, model_id, fit_args):
    """ Fit a euclidean distance lines model to the input data RDM."""

    cost_function, keep_parallel, div_norm, sub_norm, n_iter, metric, lenShort = fit_args

    original_data = copy.deepcopy(data)   # correlation data RDM, not z-scored

    # zscore the data RDM to make sure its on same scale as model for the fit
    zscored_data = stats.zscore(data, axis=None)
    opt_params, opt_SSE = bestfitRDM(zscored_data, fit_args, model_system, model_id)

    # Now visualize the geometry of the best fit lines in their euclidean space
    bestfit_lines = generate_lines(opt_params, fit_args)
    fig = plot_components(bestfit_lines)
    plt.savefig('figures/bestfit_lines_' + get_model_description(model_id, model_system, fit_args) + '.pdf', bbox_inches='tight')
    plt.close()

    # MDS of the bestfit model RDM (not z-scored)
    bestfit_RDM = pairwise_distances(bestfit_lines, metric='euclidean')
    np.fill_diagonal(np.asarray(bestfit_RDM), 0)
    MDS_model, evals = cmdscale(bestfit_RDM)
    MDS_model = MDS_model[:,:3] # just take the first 3 MDS components

    # MDS of the data
    MDS_data, evals = cmdscale(data)
    MDS_data = MDS_data[:,:3]   # just take the first 3 MDS components

    # Plot the procrustes-transformed MDS of the model and MDS of the data on top of each other
    procrustes_data, procrustes_model, disparity = procrustes(MDS_data, MDS_model)
    plot_bestmodel_data(procrustes_data, procrustes_model, model_system)

    # Generate an animation of the procrustes-transformed MDS of the model and MDS of the data on top of each other
    #animate3DMDS(procrustes_data, procrustes_model, RNN_args, metric, saveFig)

    return procrustes_data, procrustes_model, opt_SSE, opt_params

# ---------------------------------------------------------------------------- #

def get_model_description(model_id, model_system, fit_args):
    """Turn the model arguments into a descriptive string for figure and saved parameter file naming."""

    _, keep_parallel, div_norm, sub_norm, _, metric, _ = fit_args
    fitmean_text = '_meanfit_' if model_id==999 else ''
    keep_parallel_text = '_keptparallel_' if keep_parallel else '_notkeptparallel_'
    model_description = model_system + fitmean_text + keep_parallel_text + 'divnorm' + div_norm + '_subnorm' + sub_norm + '_datametric_' + metric + str(model_id)

    return model_description

# ---------------------------------------------------------------------------- #

def main():

    # fitting settings
    fit_to_mean = False
    model_system = 'RNN'       # fit to 'RNN' or 'EEG' data
    metric = 'correlation'  # the distance metric for the data (note that the lines model will be in euclidean space)
    cost_function = 'SSE'   # ***HRS obsolete, always uses SSE through scipy.optimize.minimize()
    keep_parallel = False
    div_norm = 'unconstrained' # 'unconstrained' 'normalised'  'absolute'
    sub_norm = 'unconstrained' # 'unconstrained' 'centred'  'offset'
    n_iter = 100               # number of random initialisations for each fit
    lenShort = 3               # somewhat arbitrary fixed parameter for the length of the short lines (we just fit a length ratio). Keep small ~3 otherwise true line centre params will move outside bounds
    fit_args = [cost_function, keep_parallel, div_norm, sub_norm, n_iter, metric, lenShort]

    # figure saving settings
    saveFig = True
    RNN_args = network_args('blocked', 'truecontext', 0.0)  # ***HRS beware this is no longer part of the saving string and will overwrite other figures/saved params
    EEG_args = network_args('blocked', 'truecontext', 0.0)

    # Load our data
    mean_RNN_RDM, mean_EEG_RDM, subjects_RNN_RDMs, subjects_EEG_RDMs = import_all_data(metric, RNN_args)

    # Plot low dimensional representations of the rnn and eeg data
    # generatePlots(mean_RNN_RDM, mean_EEG_RDM, RNN_args, EEG_args, saveFig)

    # Confirming fitting settings
    parallel_text = '' if keep_parallel else 'NOT '
    print('Fitting model:')
    print(' - fitting to ' + model_system + ' data')
    print(' - data RDM built from ' + metric + ' distance')
    print(' - model RDM built from euclidean distance')
    print(' - fitted lines are ' + parallel_text + 'constrained to be parallel')
    print(' - divisive normalisation is '+ div_norm + ' in fit')
    print(' - subtractive normalisation is '+ sub_norm + ' in fit')

    # Begin the fitting procedure
    allsubjects_params = []
    allsubjects_SSE = []

    if fit_to_mean:
        print('\nFitting to mean data from ' + model_system + '...')
        model_id = 999  # code id for fitting to mean
        data = mean_RNN_RDM if model_system == 'RNN' else mean_EEG_RDM
        transformed_data, transformed_model, SSE, params = fit_and_plot_model(data, model_system, model_id, fit_args)
        allsubjects_params = [params]
        allsubjects_SSE = [SSE]
    else:
        print('\nFitting to each subject/model from ' + model_system + '...')
        allsubjects_data = subjects_RNN_RDMs if model_system == 'RNN' else subjects_EEG_RDMs
        for model_id in range(allsubjects_data.shape[0]):
            data = allsubjects_data[model_id]
            print('\nFitting ' + model_system + ' subject ' + str(model_id+1) + '/' + str(allsubjects_data.shape[0]) + '...')
            transformed_data, transformed_model, SSE, params = fit_and_plot_model(data, model_system, model_id, fit_args)
            allsubjects_params.append(params)
            allsubjects_SSE.append(SSE)
    print('Fitting complete.')

    # Save the best fit parameters so we dont have to repeat this whole process
    model_description = get_model_description(model_id, model_system, fit_args)

    filename_parameters = 'bestfit_parameters_' + model_description
    filename_SSE = 'bestfit_SSE_' + model_description
    filepath_parameters = os.path.join(SAVE_FILELOC, filename_parameters)
    filepath_SSE = os.path.join(SAVE_FILELOC, filename_SSE)

    np.save(filepath_parameters, np.asarray(allsubjects_params))
    np.save(filepath_SSE, np.asarray(allsubjects_SSE))

    # Perform statistical tests on the best fit models
    parallelness_test(model_system, allsubjects_params, fit_args)
    plot_divisive_normalisation(model_system, allsubjects_params, fit_args)
    # plot_subtractive_normalisation(model_system, allsubjects_params) # not written properly yet and unnecessary

# ---------------------------------------------------------------------------- #

main()
