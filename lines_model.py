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
import multiprocessing as mp

from scipy.spatial import procrustes
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import correlation
from sklearn.manifold import MDS
#from astropy.stats import rayleightest
from pycircstat.tests import rayleigh
from sklearn.model_selection import KFold

import matplotlib.colors as mplcol
import sklearn.decomposition
from matplotlib import animation
from mpl_toolkits import mplot3d

RNN_BASEPATH = 'lines_model/rnn_RDMs/'
EEG_BASEPATH = 'lines_model/brain_RDMs/'
EEG_FILE_FABRICE = 'avRDM_magn.mat'
EEG_FILE_CHRIS = 'chris_eeg_data.mat'
SAVE_FILELOC = 'linesmodel_parameters/'
contextcolours = ['dodgerblue', 'orangered', np.asarray([1.0, 0.69803921, 0.0, 1.0]), 'black']   # 1-16, 1-11, 6-16 like fabrices colours
modelcolours = ['mediumblue','saddlebrown','darkgoldenrod']

condition_colours = [['teal','sienna'],['turquoise','darkorange']]


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
def import_EEG_data(basepath, filename, which_set):
    """ import the .mat eeg data for all subjects """

    filepath = os.path.join(basepath, filename)
    with open(filepath) as eeg_file:
        eeg_activations = loadmat(filepath)
        if which_set == 'fabrice':
            eeg_activations = np.asarray(eeg_activations['magndat'])
            eeg_activations = np.transpose(eeg_activations, (2,0,1))
        elif which_set == 'chris':
            eeg_activations = np.asarray(eeg_activations['data'])

    return eeg_activations          # subjects x condition x condition [low -> high -> full]
# ---------------------------------------------------------------------------- #

def import_RNN_data(basepath, args):
    """ import the rnn activations for each of ten models we care about"""
    rnn_files = os.listdir(basepath)
    rnn_files = [x for x in rnn_files if (args.blocking in x) and (args.context_label in x) ]
    rnn_files = [x for x in rnn_files if ('trlf' + str(args.train_lesion_freq) in x)]
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

def import_all_data(metric, RNN_args, which_set):
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
    EEG_FILE = EEG_FILE_CHRIS if which_set == 'chris' else EEG_FILE_FABRICE
    subjects_EEG_RDMs = import_EEG_data(EEG_BASEPATH, EEG_FILE, which_set)
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
    plotMDS(eeg_RDM, eeg_args, 'eeg_mds_chrisdata_',  -20, (-0.35,0.35), saveFig)

    # RDM matrices
    plotRDM(rnn_RDM, rnn_args, 'rnn_rdm_', saveFig)
    plotRDM(eeg_RDM, eeg_args, 'eeg_rdm_chrisdata_', saveFig)

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

    _, keep_parallel, div_norm, sub_norm, _, _, lenShort, _, line_centre_method = fit_args

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
    _, keep_parallel, div_norm, sub_norm, _, _, lenShort,_,line_centre_method = fit_args

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
        if line_centre_method == 'free':
            centre_A = [x0,y0,z0]
            centre_B = [x0,yB,zB]
            centre_C = [x0,yC,zC]
    elif sub_norm == 'offset':
        if line_centre_method == 'free':
            # constrain the centre of the three lines to be perfectly offset in the magnitude dimension
            mag_centre_low = -2.5 * (lenLong/const.N_POINTS_LONG)
            mag_centre_high = -mag_centre_low
            centre_A = [x0,y0,z0]
            centre_B = [mag_centre_low,yB,zB]
            centre_C = [mag_centre_high,yC,zC]
    elif sub_norm == 'unconstrained':
        if line_centre_method == 'free':
            centre_A = [x0,y0,z0]  # coordinate origin for centre of line A, long line
            centre_B = [xB,yB,zB]
            centre_C = [xC,yC,zC]
        #else:
        #    centre_A = cluster_centres[0] # set the line centres to the centre of each cluster
        #    centre_B = cluster_centres[1]
        #    centre_C = cluster_centres[2]

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
    fit_algorithm, keep_parallel, div_norm, sub_norm, n_iter, metric, lenShort, _, line_centre_method = fit_args

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
            lower_bound = [-2, -2, -2, -2, -2, -2, 0, -1, -1, -1, -1, -1, -1]
            upper_bound = [2,   2,  2,  2,  2,  2, 2,  1,  1,  1,  1,  1,  1]

            # Randomise the initial parameter starting point for those we are fitting (first iteration will just use our guesses)
            init_params = [random.uniform(lower_bound[i],upper_bound[i]) for i in range(len(lower_bound))]

            # select a loss function for the fit: either SSE (for euclidean fits) or summed correlation distance
            bounds = [(lower_bound[i],upper_bound[i]) for i in range(len(init_params))]
            if fit_algorithm == 'default':
                result = optimize.minimize(fitfunc, init_params, args=(fit_args, uppertri_data), bounds=bounds )
            elif fit_algorithm == 'Nelder-Mead':
                result = optimize.minimize(fitfunc, init_params, args=(fit_args, uppertri_data), method='Nelder-Mead' ) # trying a different solver method
            else:
                print('Warning: please specify which algorithm you want to use for the SSE optimisation.')
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

def bestfitRDM(data, fit_args, model_description, model_id=0):
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
    plt.savefig('figures/bestfit_rdm_' + model_description + '.pdf',bbox_inches='tight')
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

def plot_bestmodel_data(mtx1, mtx2, model_description):
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

    plt.savefig('figures/bestfit_procrustes_' + model_description + '.pdf', bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------- #

def animate3DMDS(MDS_data, MDS_model, model_id, model_system, fit_args, fit_method, RNN_args, saveFig=True):
    """ This function will plot the numerosity labeled, context-marked MDS projections
     of the hidden unit activations on a 3D plot, animate/rotate that plot to view it
     from different angles and optionally save it as a mp4 file.
    """
    model_description = get_model_description(model_id, model_system, fit_args, fit_method, RNN_args)
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
        if model_system == 'EEG':
            axislimits = (-0.3, 0.3)
        else:
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
        anim.save('animations/MDS_3Danim_linesmodelanddata_'+model_description+'.mp4', writer=writer)

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

    _, keep_parallel, div_norm, sub_norm, _, _, _, _, _ = fit_args
    parallel_fig_text = '_keptparallel' if keep_parallel else '_notkeptparallel'
    model_string = parallel_fig_text + '_divnorm' + div_norm + '_subnorm' + sub_norm
    model_string = model_string + '_meanfit' if mean_fits else model_string

    angles = []
    for params in allsubjects_params:
        # the best fitting directions of each line in 3d space
        dir_lineA = np.asarray([1,0,0])
        dir_lineB = params[7:10]
        dir_lineC = params[10:]

        # compute the angles between the unit vectors in the bestfit lines model
        angle_AB = angle_between(dir_lineA, dir_lineB) # line A v B
        angle_BC = angle_between(dir_lineB, dir_lineC) # line B v C
        angle_AC = angle_between(dir_lineA, dir_lineC) # line A v C

        angles.append(angle_AB)
        angles.append(angle_BC)
        angles.append(angle_AC)

    angles = np.asarray(angles)
    plt.figure()
    plt.hist([angle*(180/np.pi) for angle in angles], bins=40)
    ax = plt.gca()
    ax.set_xlim(-180, 180)
    plt.xlabel('Angle between lines (degrees)')
    plt.title('Distribution of angle between bestfit lines')
    plt.savefig('figures/hist_angles_between_lines_' + model_system + model_string + '.pdf', bbox_inches='tight')
    plt.close()

    # perform a rayleigh test that the angles come from a uniform circular distribution
    p,z = rayleigh(angles)

    print('-----')
    print('Rayleigh test (distr. angles between bestfit lines):')
    print('p = {:.3e}'.format(p))
    print("Rayleigh's z: {:.3e}".format(z))
    return p,z

# ---------------------------------------------------------------------------- #

def plot_divisive_normalisation(model_system, allsubjects_params, fit_args):

    _, keep_parallel, div_norm, sub_norm, _, _, _, which_set, _ = fit_args
    parallel_fig_text = '_keptparallel' if keep_parallel else '_notkeptparallel'
    model_string = parallel_fig_text + '_divnorm' + div_norm + '_subnorm' + sub_norm

    ratios = []
    for params in allsubjects_params:
        divisive_norm_ratio = params[6]
        ratios.append(divisive_norm_ratio)

    if len(ratios)>1:
        plt.figure()
        plt.hist(ratios, bins=30)
        ax = plt.gca()
        ax.set_xlim(0.95, 1.5)
        plt.xlabel('Divisive norm ratios (1: totally normalised -> 1.455: totally absolute)')
        plt.title('Bestfit divisive norm ratios: ' + model_system)
        plt.savefig('figures/hist_divnorm_ratios_' + model_system + model_string + which_set + '.pdf', bbox_inches='tight')
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

def fit_and_plot_model(train_data, test_data, model_system, model_id, fit_args, fit_method, RNN_args):
    """ Fit a euclidean distance lines model to the input data RDM."""

    fit_algorithm, keep_parallel, div_norm, sub_norm, n_iter, metric, lenShort, which_set, line_centre_method = fit_args
    model_description = get_model_description(model_id, model_system, fit_args, fit_method, RNN_args)
    model_description = model_description + '_' + fit_method if 'cross' in fit_method else model_description

    # zscore the data RDM to make sure its on same scale as model for the fit
    zscored_train_data = stats.zscore(train_data, axis=None)
    opt_params, opt_SSE = bestfitRDM(zscored_train_data, fit_args, model_description, model_id)

    # Now visualize the geometry of the best fit lines in their euclidean space
    bestfit_lines = generate_lines(opt_params, fit_args)
    fig = plot_components(bestfit_lines)
    plt.savefig('figures/bestfit_lines_' + model_description + '.pdf', bbox_inches='tight')
    plt.close()

    # MDS of the bestfit model RDM (not z-scored)
    bestfit_RDM = pairwise_distances(bestfit_lines, metric='euclidean')
    np.fill_diagonal(np.asarray(bestfit_RDM), 0)
    MDS_model, evals = cmdscale(bestfit_RDM)
    MDS_model = MDS_model[:,:3] # just take the first 3 MDS components

    # MDS of the data
    # note that when we are doing cross-val this will be the left out subject,
    # and when fitting to individual subjects of the global mean, its just the same as training data
    MDS_data, evals = cmdscale(test_data)
    MDS_data = MDS_data[:,:3]   # just take the first 3 MDS components

    # Plot the procrustes-transformed MDS of the model and MDS of the data on top of each other
    procrustes_data, procrustes_model, disparity = procrustes(MDS_data, MDS_model)
    plot_bestmodel_data(procrustes_data, procrustes_model, model_description)

    # Generate an animation of the procrustes-transformed MDS of the model and MDS of the data on top of each other
    #animate3DMDS(procrustes_data, procrustes_model, model_id, model_system, fit_args, fit_method, lesionFrequency)

    # now compute the final SSE by testing the model against the test set
    zscored_test_data = stats.zscore(test_data, axis=None)
    uppertri_data, ind = matrix_to_uppertri(zscored_test_data)
    test_SSE = sum((makelinesmodel(fit_args, *opt_params) - uppertri_data)**2)

    # do a sanity check (these should be the same unless doing cross-validation)
    print('train SSE: {}'.format(opt_SSE))
    print('test SSE: {}'.format(test_SSE))

    return procrustes_data, procrustes_model, test_SSE, opt_params

# ---------------------------------------------------------------------------- #

def get_model_description(model_id, model_system, fit_args, fit_method, RNN_args):
    """Turn the model arguments into a descriptive string for figure and saved parameter file naming."""

    fit_algorithm, keep_parallel, div_norm, sub_norm, _, metric, _, which_set, line_centre_method = fit_args
    keep_parallel_text = '_keptparallel_' if keep_parallel else '_notkeptparallel_'
    model_description = model_system + '_' + fit_method +  '_' + RNN_args.blocking + '_' + RNN_args.context_label + 'contextlabel' + keep_parallel_text + 'divnorm' + div_norm + '_subnorm' + sub_norm + '_linecentres_' + line_centre_method + '_datametric_' + metric + str(model_id) + '_' + fit_algorithm + '_' + which_set + '_trlf' + str(RNN_args.train_lesion_freq)

    return model_description

# ---------------------------------------------------------------------------- #

def fit_models(datasets, fit_method, model_system, fit_args, RNN_args):
    """Fit model defined by fit_args to all the subjects (or to mean, if instructed),
    and plot and save the resulting parameters/SSE/MDS etc results."""

    mean_RNN_RDM, mean_EEG_RDM, subjects_RNN_RDMs, subjects_EEG_RDMs = datasets
    fit_algorithm, keep_parallel, div_norm, sub_norm, n_iter, metric, lenShort, which_set, line_centre_method = fit_args

    # Confirming fitting settings
    parallel_text = '' if keep_parallel else 'NOT '
    fit_method_text = 'with cross-validation' if fit_method == 'cross_val' else ''
    print('Fitting model:')
    print(' - fitting to ' + model_system + ' data' + fit_method_text)
    print(' - data RDM built from ' + metric + ' distance')
    print(' - fitting with algorithm ' + fit_algorithm)
    print(' - model RDM built from euclidean distance')
    print(' - fitted lines are ' + parallel_text + 'constrained to be parallel')
    print(' - divisive normalisation is '+ div_norm + ' in fit')
    print(' - subtractive normalisation is '+ sub_norm + ' in fit')

    # Begin the fitting procedure
    allsubjects_params = []
    allsubjects_SSE = []

    if fit_method == 'fit_to_mean':
        # same train and test data
        print('\nFitting to mean data from ' + model_system + '...')
        model_id = 999  # code id for fitting to mean
        train_data = mean_RNN_RDM if model_system == 'RNN' else mean_EEG_RDM
        test_data = copy.deepcopy(train_data)

        transformed_data, transformed_model, SSE, params = fit_and_plot_model(train_data, test_data, model_system, model_id, fit_args, fit_method, RNN_args)
        allsubjects_params = [params]
        allsubjects_SSE = [SSE]

    elif fit_method == 'individual':
        # same train and test data
        print('\nFitting to each subject/model from ' + model_system + '...')
        allsubjects_data = subjects_RNN_RDMs if model_system == 'RNN' else subjects_EEG_RDMs
        for model_id in range(allsubjects_data.shape[0]):
            train_data = allsubjects_data[model_id]
            test_data = copy.deepcopy(train_data)
            print('\nFitting ' + model_system + ' subject ' + str(model_id+1) + '/' + str(allsubjects_data.shape[0]) + '...')
            transformed_data, transformed_model, SSE, params = fit_and_plot_model(train_data, test_data, model_system, model_id, fit_args, fit_method, RNN_args)
            allsubjects_params.append(params)
            allsubjects_SSE.append(SSE)

    elif fit_method == 'cross_val':
        # train on k-1 dataset and test on the left-out subject's data
        print('\nFitting to k-1 subjects/models (crossval) from ' + model_system + '...')
        allsubjects_data = subjects_RNN_RDMs if model_system == 'RNN' else subjects_EEG_RDMs

        n_folds = allsubjects_data.shape[0]  # just leave out one subject or model in each fold
        kf = KFold(n_splits=n_folds)
        fold = 0
        for train_index, test_index in kf.split(allsubjects_data):
            print("TRAIN SET:", train_index, "TEST SET:", test_index)
            # assemble the k-1 dataset to fit to
            train_data = np.mean(allsubjects_data[train_index,:,:], axis=0)
            test_data = np.mean(allsubjects_data[test_index,:,:], axis=0)

            # fit a model to the mean of the k-1 data subset (and test it on the left-out test data)
            print('\nFitting ' + model_system + ', leaving out subject ' + str(fold+1) + '/' + str(allsubjects_data.shape[0]) + '...')
            transformed_data, transformed_model, SSE, params = fit_and_plot_model(train_data, test_data, model_system, fold, fit_args, fit_method, RNN_args)
            allsubjects_params.append(params)
            allsubjects_SSE.append(SSE)
            fold += 1
        model_id = fold

    elif fit_method == 'cross_val_5fold':

        # train on k-1 folds dataset and test on the left-out portion of data
        print('\nFitting 5-fold cross-validation to subset of subjects/models from ' + model_system + '...')
        allsubjects_data = subjects_RNN_RDMs if model_system == 'RNN' else subjects_EEG_RDMs

        n_folds = 5
        kf = KFold(n_splits=n_folds)
        fold = 0
        for train_index, test_index in kf.split(allsubjects_data):
            print("TRAIN SET:", train_index, "TEST SET:", test_index)
            # assemble the k-1 dataset to fit to
            train_data = np.mean(allsubjects_data[train_index,:,:], axis=0)
            test_data = np.mean(allsubjects_data[test_index,:,:], axis=0)

            # fit a model to the mean of the k-1 data subset (and test it on the left-out test data)
            print('\nFitting ' + model_system + ', leaving out fold ' + str(fold+1) + '/' + str(n_folds) + '...')
            transformed_data, transformed_model, SSE, params = fit_and_plot_model(train_data, test_data, model_system, fold, fit_args, fit_method, RNN_args)
            allsubjects_params.append(params)
            allsubjects_SSE.append(SSE)
            fold += 1
        model_id = fold

    print('Fitting complete.')

    # Save the best fit parameters so we dont have to repeat this whole process
    model_description = get_model_description(model_id, model_system, fit_args, fit_method, RNN_args)
    model_description = model_description + '_' + fit_method if 'cross' in fit_method else model_description

    filename_parameters = 'bestfit_parameters_' + model_description
    filename_SSE = 'bestfit_SSE_' + model_description
    filepath_parameters = os.path.join(SAVE_FILELOC, filename_parameters)
    filepath_SSE = os.path.join(SAVE_FILELOC, filename_SSE)

    np.save(filepath_parameters, np.asarray(allsubjects_params))
    np.save(filepath_SSE, np.asarray(allsubjects_SSE))


    # Perform statistical tests on the best fit models
    #parallelness_test(model_system, allsubjects_params, fit_args)
    #plot_divisive_normalisation(model_system, allsubjects_params, fit_args)
    # plot_subtractive_normalisation(model_system, allsubjects_params) # not written properly yet and unnecessary
    return None
# ---------------------------------------------------------------------------- #

def import_fit_parameters(ax, args, fit_args, model_system, fit_method):

    rnn_files = os.listdir(const.PARAMETER_DIRECTORY)
    fit_algorithm, keep_parallel, div_norm, sub_norm, n_iter, metric, lenShort, which_set, line_centre_method = fit_args

    # restrict by RNN training conditions
    rnn_files = [x for x in rnn_files if (args.blocking in x) and (args.context_label in x) ]
    rnn_files = [x for x in rnn_files if ('trlf' + str(args.train_lesion_freq) in x)]
    rnn_files = [x for x in rnn_files if model_system in x]

    # restrict by lines model fitting conditions
    parallel_fig_text = '_keptparallel' if keep_parallel else '_notkeptparallel'
    rnn_files = [x for x in rnn_files if (parallel_fig_text in x) and ('_divnorm' + div_norm in x)]
    rnn_files = [x for x in rnn_files if ('_subnorm' + sub_norm in x) and (fit_method in x)]

    SSE_files = [x for x in rnn_files if ('_SSE_' in x)]
    params_files = [x for x in rnn_files if ('_parameters_' in x)]

    # data for each rnn instance
    for filename in params_files:
        filepath = os.path.join(const.PARAMETER_DIRECTORY, filename)

        with open(filepath) as rnn_file:
            params_data = np.load(filepath)

    params_data = np.asarray(params_data)
    n_models = params_data.shape[0]
    context_distance = []
    subnorm_ratios = []
    divnorm_ratios = []

    for i in range(n_models):
        params = params_data[i]
        xB,yB,zB, xC,yC,zC, lenRatio, dirxB,diryB,dirzB, dirxC,diryC,dirzC = params

        # evaluate context separation
        centre_A = np.asarray([0,0,0])
        centre_B = np.asarray([xB,yB,zB])
        centre_C = np.asarray([xC,yC,zC])
        dist_AB = np.absolute(np.linalg.norm(centre_A - centre_B))
        dist_AC = np.absolute(np.linalg.norm(centre_A - centre_C))
        dist_BC = np.absolute(np.linalg.norm(centre_B - centre_C))
        context_distance.append(np.mean([dist_AB, dist_AC, dist_BC]))

        # evaluate subtractive normalisation
        # full centre is at 0 in magnitude dimension by definition
        low_offset = centre_B[0]
        high_offset = -centre_C[0]

        #theoretical full offset values
        lenLong = lenShort*lenRatio
        full_offset_centre_low = -2.5 * (lenLong/const.N_POINTS_LONG)
        full_offset_centre_high = -full_offset_centre_low

        low_subratio = np.absolute(low_offset/full_offset_centre_low)
        high_subratio = np.absolute(high_offset/full_offset_centre_high)
        mean_offset = np.mean([low_subratio, high_subratio])
        subnorm_ratios.append(1-mean_offset)  # make so that higher value means more normalised

        lenRatio = 1.0/lenRatio
        divnorm_ratios.append(lenRatio)

    mean_divnorm_ratios = np.mean(divnorm_ratios)
    std_divnorm_ratios = np.std(divnorm_ratios)

    mean_context_distance = np.mean(context_distance)
    std_context_distance = np.std(context_distance)

    mean_subnorm_ratios = np.mean(subnorm_ratios)
    std_subnorm_ratios = np.std(subnorm_ratios)

    lenRatio = lenRatio # make so that higher value means more normalised

    print('--------')
    print('length ratio (short/long): {:.4f} +- {:.4f}'.format(mean_divnorm_ratios, std_divnorm_ratios))
    print('subtractive norm ratio: {:.4f} +- {:.4f}'.format(mean_subnorm_ratios, std_subnorm_ratios))
    print('context distance: {:.4f} +- {:.4f}'.format(mean_context_distance, std_context_distance))

    summary_params = [mean_context_distance, std_context_distance], [mean_divnorm_ratios, std_divnorm_ratios], [mean_subnorm_ratios, std_subnorm_ratios]

    return summary_params

# ---------------------------------------------------------------------------- #

def plot_fit_parameters(ax, summary_parameters, lesioning, block, contextlabelling):

    truecontext_lesioned_divnorm = []
    truecontext_unlesioned_divnorm = []
    constcontext_lesioned_divnorm = []
    constcontext_unlesioned_divnorm = []

    truecontext_lesioned_contextdist = []
    truecontext_unlesioned_contextdist = []
    constcontext_lesioned_contextdist = []
    constcontext_unlesioned_contextdist = []

    truecontext_lesioned_subnorm = []
    truecontext_unlesioned_subnorm = []
    constcontext_lesioned_subnorm = []
    constcontext_unlesioned_subnorm = []

    for i, params in enumerate(summary_parameters):

        context_dist, div_norm, sub_norm = params
        mean_context_distance, std_context_distance = context_dist
        lenRatio, lenRatio_std = div_norm
        subRatio, subRatio_std = sub_norm
        lesion_state = lesioning[i]
        block_state = block[i]
        contextlabel = contextlabelling[i]

        xloc = 1 if block_state == 'blocked' else 0
        cue = 0 if contextlabel == 'truecontext' else 1
        lesioned = 0 if lesion_state == False else 1
        colour = condition_colours[cue][lesioned]

        ax[0].errorbar(xloc, mean_context_distance, std_context_distance, color=colour)
        ax[0].scatter(xloc, mean_context_distance, color=colour)
        ax[0].set_xticks([0,1])
        ax[0].set_xticklabels(['interleaved','blocked'], rotation=45)
        ax[0].set_title('context distance')

        ax[1].axhline(1.0, color='lightgrey', linestyle=':', linewidth=1)
        ax[1].axhline(11.0/16.0, color='lightgrey', linestyle=':', linewidth=1)
        ax[1].errorbar(xloc, lenRatio, lenRatio_std, color=colour)
        h = ax[1].scatter(xloc, lenRatio, color=colour)
        ax[1].set_xticks([0,1])
        ax[1].set_xticklabels(['interleaved','blocked'], rotation=45)
        ax[1].set_title('divisive norm')
        #ax[1].set_ylim([0.95, 1.7])

        ax[2].axhline(1.0, color='lightgrey', linestyle=':', linewidth=1)
        ax[2].axhline(0.0, color='lightgrey', linestyle=':', linewidth=1)
        ax[2].errorbar(xloc, subRatio, subRatio_std, color=colour)
        h = ax[2].scatter(xloc, subRatio, color=colour)
        ax[2].set_xticks([0,1])
        ax[2].set_xticklabels(['interleaved','blocked'], rotation=45)
        ax[2].set_title('subtractive norm')
        #ax[2].set_ylim([0.9, 1.7])

        if contextlabel =='truecontext':
            if lesion_state == False:
                truecontext_unlesioned_divnorm.append(lenRatio)
                truecontext_unlesioned_contextdist.append(mean_context_distance)
                truecontext_unlesioned_subnorm.append(subRatio)
            else:
                truecontext_lesioned_divnorm.append(lenRatio)
                truecontext_lesioned_contextdist.append(mean_context_distance)
                truecontext_lesioned_subnorm.append(subRatio)
        else:
            if lesion_state == False:
                constcontext_unlesioned_divnorm.append(lenRatio)
                constcontext_unlesioned_contextdist.append(mean_context_distance)
                constcontext_unlesioned_subnorm.append(subRatio)
            else:
                constcontext_lesioned_divnorm.append(lenRatio)
                constcontext_lesioned_contextdist.append(mean_context_distance)
                constcontext_lesioned_subnorm.append(subRatio)

    ax[0].plot([0,1], truecontext_unlesioned_contextdist, color=condition_colours[0][0])
    ax[0].plot([0,1], constcontext_unlesioned_contextdist, color=condition_colours[1][0])
    ax[0].plot([0,1], truecontext_lesioned_contextdist, color=condition_colours[0][1])
    ax[0].plot([0,1], constcontext_lesioned_contextdist, color=condition_colours[1][1])

    ax[1].plot([0,1], truecontext_unlesioned_divnorm, color=condition_colours[0][0])
    ax[1].plot([0,1], constcontext_unlesioned_divnorm, color=condition_colours[1][0])
    ax[1].plot([0,1], truecontext_lesioned_divnorm, color=condition_colours[0][1])
    ax[1].plot([0,1], constcontext_lesioned_divnorm, color=condition_colours[1][1])

    h1, = ax[2].plot([0,1], truecontext_unlesioned_subnorm, color=condition_colours[0][0])
    h2, = ax[2].plot([0,1], constcontext_unlesioned_subnorm, color=condition_colours[1][0])
    h3, = ax[2].plot([0,1], truecontext_lesioned_subnorm, color=condition_colours[0][1])
    h4, = ax[2].plot([0,1], constcontext_lesioned_subnorm, color=condition_colours[1][1])

    plt.legend((h1,h2,h3,h4), ('not lesioned, context labelled','not lesioned, context not labelled','lesioned, context labelled', 'lesioned, context not labelled'))

# ---------------------------------------------------------------------------- #

def main():

    # fitting settings
    fit_models = False
    plot_parameters = True
    fit_method = 'individual'  # 'fit_to_mean'  'individual' 'cross_val' 'cross_val_5fold'
    model_system = 'RNN'       # fit to 'RNN' or 'EEG' data
    lesioned = True            # True/False selects 0.0f during training or 0.1f during training
    parallelize = False
    fig, ax = plt.subplots(1,3, figsize=(8,4))
    summary_parameters = []
    lesioning = []
    contextlabelling = []
    block = []

    for i, lesioned in enumerate([True, False]):
        for j, contextlabel in enumerate(['truecontext','constantcontext']):
            for k, blocking in enumerate(['intermingled', 'blocked']):

                metric = 'correlation'  # the distance metric for the data (note that the lines model will be in euclidean space)
                fit_algorithm = 'default'  # 'default'  'Nelder-Mead'
                keep_parallel = True
                line_centre_method = 'free' # 'free' or 'cluster_centres'
                div_norm = 'unconstrained' # 'unconstrained' 'normalised'  'absolute'
                sub_norm = 'unconstrained' # 'unconstrained' 'centred'  'offset'
                n_iter = 100               # number of random initialisations for each fit
                lenShort = 3               # somewhat arbitrary fixed parameter for the length of the short lines (we just fit a length ratio). Keep small ~3 otherwise true line centre params will move outside bounds
                which_set = 'chris'   #'fabrice' 'chris'

                # figure saving settings
                saveFig = True
                lesion_freq = 0.1 if lesioned else 0.0
                RNN_args = network_args(blocking, contextlabel, lesion_freq)  # ***HRS beware this is no longer part of the saving string and will overwrite other figures/saved params
                EEG_args = network_args('blocked', 'truecontext', 0.0)

                if fit_models:
                    # Load our data
                    mean_RNN_RDM, mean_EEG_RDM, subjects_RNN_RDMs, subjects_EEG_RDMs = import_all_data(metric, RNN_args, which_set)
                    datasets = [mean_RNN_RDM, mean_EEG_RDM, subjects_RNN_RDMs, subjects_EEG_RDMs]

                # Plot low dimensional representations of the rnn and eeg data
                #generatePlots(mean_RNN_RDM, mean_EEG_RDM, RNN_args, EEG_args, saveFig)

                # Sets of parameter settings for looping
                parameter_set_1 = [False,  'unconstrained', 'unconstrained']  # totally free model: 1
                parameter_set_2 = [True,  'unconstrained', 'unconstrained']   # free but parallel model: 2
                parameter_set_3 = [True,  'normalised', 'unconstrained']   # parallel and div normalised model: 3
                parameter_set_4 = [True,  'absolute', 'unconstrained']     # parallel and div absolute model: 4
                parameter_set_5 = [True,  'unconstrained', 'centred']      # parallel and sub centred model: 5
                parameter_set_6 = [True,  'unconstrained', 'offset']       # parallel and sub offset model: 6
                #parameter_combinations = [parameter_set_1, parameter_set_2, parameter_set_3, parameter_set_4, parameter_set_5, parameter_set_6]

                # actually when we dont restrict it to be parallel it seems that there is more divisive normalisation in the EEG subjects so lets do that model
                parameter_set_7 = [False,  'normalised', 'unconstrained']   # parallel and div normalised model: 3
                parameter_set_8 = [False,  'absolute', 'unconstrained']     # parallel and div absolute model: 4
                parameter_set_9 = [False,  'unconstrained', 'centred']      # parallel and sub centred model: 5
                parameter_set_10 = [False,  'unconstrained', 'offset']       # parallel and sub offset model: 6
                #parameter_combinations = [parameter_set_1, parameter_set_2, parameter_set_3, parameter_set_4, parameter_set_5,
                #                          parameter_set_6, parameter_set_7, parameter_set_8, parameter_set_9, parameter_set_10]
                parameter_combinations = [parameter_set_2]

                if fit_models:
                    if parallelize:
                        # Parallelize fitting of different models to different cores
                        n_processors = mp.cpu_count()
                        print("\nParallelising fitting over all {} available CPU cores.\n".format(n_processors))
                        # create a multiprocessing pool
                        pool = mp.Pool(n_processors)
                        parameter_args = [[fit_algorithm, keep_parallel, div_norm, sub_norm, n_iter, metric, lenShort, which_set] for keep_parallel, div_norm, sub_norm in parameter_combinations]
                        results = [ pool.apply(fit_models, args=(datasets, fit_method, model_system, fit_args, lesion_freq)) for fit_args in parameter_args]
                        pool.close()
                    else:
                        # Serial option
                        for keep_parallel, div_norm, sub_norm in parameter_combinations:
                            fit_args = [fit_algorithm, keep_parallel, div_norm, sub_norm, n_iter, metric, lenShort, which_set, line_centre_method]
                            fit_models(datasets, fit_method, model_system, fit_args, RNN_args)

                # compare fit model parameters
                if plot_parameters:
                    keep_parallel, div_norm, sub_norm = parameter_set_2
                    fit_args = [fit_algorithm, keep_parallel, div_norm, sub_norm, n_iter, metric, lenShort, which_set, line_centre_method]
                    params = import_fit_parameters(ax, RNN_args, fit_args, model_system, fit_method)
                    summary_parameters.append(params)
                    lesioning.append(lesioned)
                    block.append(blocking)
                    contextlabelling.append(contextlabel)

    #plt.legend(handles, ['lesioned','lesioned','','','','','',''])
    plot_fit_parameters(ax, summary_parameters, lesioning, block, contextlabelling)
    plt.savefig(os.path.join(const.FIGURE_DIRECTORY, 'parameters_plot.pdf'), bbox_inches='tight')

# ---------------------------------------------------------------------------- #

#if __name__ is '__main__':
main()
