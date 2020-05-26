# Author: Hannah Sheahan, sheahan.hannah@gmail.com
# Date: 17/05/2020
# Issues: N/A
# Notes: - individual EEG subject lines model fits are terrible because the data
#          is so noisy at an individual subject level (no surprises there)
# ---------------------------------------------------------------------------- #

import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
from pycircstat.tests import rayleigh

#import lines_model as lines

# ---------------------------------------------------------------------------- #

FILE_LOC = 'linesmodel_parameters/'
FIGURE_LOC = 'figures/'

# ---------------------------------------------------------------------------- #
def load_fits(fit_args):

    fit_algorithm, fit_method, model_system, keep_parallel, div_norm, sub_norm, which_set = fit_args

    # extract the saved parameters and SSE for the fitted models specified in fit_args
    parallel_searchtext = '_keptparallel' if keep_parallel else '_notkeptparallel'
    all_files = os.listdir(FILE_LOC)
    all_files = [x for x in all_files if (model_system in x)]
    if fit_method == 'fit_to_mean':
        all_files = [x for x in all_files if ('mean' in x)]
    elif fit_method == 'individual':
        all_files = [x for x in all_files if  ('mean' not in x)]
    elif fit_method == 'cross_val':
        all_files = [x for x in all_files if  ('cross_val' in x) and ('cross_val_5fold' not in x)]
    elif fit_method == 'cross_val_5fold':
        all_files = [x for x in all_files if ('cross_val_5fold' in x)]

    all_files = [x for x in all_files if ('divnorm'+div_norm in x) and ('subnorm'+sub_norm in x)]
    all_files = [x for x in all_files if (parallel_searchtext in x)]
    all_files = [x for x in all_files if (which_set in x)]


    if fit_algorithm == "Nelder-Mead":
        all_files = [x for x in all_files if (fit_algorithm in x)]

    sse_files = [x for x in all_files if ('SSE' in x)]
    params_files = [x for x in all_files if ('parameters' in x)]

    if len(sse_files) > 1 or len(params_files) > 1:
        print('Warning: multiple fitting records found for model specified in fit_args.')
        print(all_files)
    elif len(sse_files)==0 or len(params_files)==0:
        print('Warning: no fitting record found for model specified in fit_args.')
    else:
        SSE = np.load(os.path.join(FILE_LOC, sse_files[0]))
        parameters = np.load(os.path.join(FILE_LOC, params_files[0]))

    return SSE, parameters

# ---------------------------------------------------------------------------- #

def view_fit(sse, params):
    print('-------------------')
    print('SSE: {:.3f}'.format(sse))
    print('Line length ratio (1.0=norm -> 1.45=abs): {:.4f}'.format(params[6]))
    print('Line B centre, (x,y,z): {:.3f},{:.3f},{:.3f}'.format(*params[0:3]))
    print('Line C centre, (x,y,z): {:.3f},{:.3f},{:.3f}'.format(*params[3:6]))
    print('Line B direction, (x,y,z): {:.3f},{:.3f},{:.3f}'.format(*params[7:10]))
    print('Line C direction, (x,y,z): {:.3f},{:.3f},{:.3f}'.format(*params[10:]))

# ---------------------------------------------------------------------------- #

def unit_vector(x):
    # create a unit vector from the input array
    x = np.asarray(x)
    normed_x = x / np.linalg.norm(x)
    normed_x = list(normed_x)
    return normed_x

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

    fit_algorithm, mean_fits, _, keep_parallel, div_norm, sub_norm, _ = fit_args
    parallel_fig_text = '_keptparallel' if keep_parallel else '_notkeptparallel'
    model_string = parallel_fig_text + '_divnorm' + div_norm + '_subnorm' + sub_norm + '_' + mean_fits + '_' + fit_algorithm


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
    ax.set_xlim(-10, 180)
    plt.xlabel('Angle between lines (degrees)')
    plt.title('Distribution of angle between bestfit lines')
    plt.savefig('figures/hist_angles_between_lines_' + model_system + model_string + '.pdf', bbox_inches='tight')
    plt.close()

    # perform a rayleigh test that the angles come from a uniform circular distribution
    p,z = rayleigh(angles)

    print('-----')
    print('Rayleigh test (distr. angles between bestfit lines):')
    print('p = {:.3e}'.format(p))
    print("Rayleigh's z: {:.3f}".format(z))
    return p,z

# ---------------------------------------------------------------------------- #

def plot_divisive_normalisation(model_system, allsubjects_params, fit_args):

    fit_algorithm, mean_fits, _, keep_parallel, div_norm, sub_norm, _ = fit_args
    parallel_fig_text = '_keptparallel' if keep_parallel else '_notkeptparallel'
    model_string = parallel_fig_text + '_divnorm' + div_norm + '_subnorm' + sub_norm
    model_string = model_string + '_meanfit' if mean_fits else model_string

    ratios = []
    for params in allsubjects_params:
        divisive_norm_ratio = params[6]
        ratios.append(divisive_norm_ratio)

    #if len(ratios)>1:
    plt.figure()
    plt.hist(ratios, bins=15)
    ax = plt.gca()
    ax.set_xlim(0.95, 1.5)
    plt.xlabel('Divisive norm ratios (1: totally normalised -> 1.455: totally absolute)')
    plt.title('Bestfit divisive norm ratios: ' + model_system)
    plt.savefig('figures/hist_divnorm_ratios_' + model_system + model_string + '.pdf', bbox_inches='tight')
    plt.close()
    if len(ratios)==1:
        print('-----')
        print('Divisive norm ratio:   {:3f}'.format(ratios[0]))
        print('(totally normalised: 1 --> totally absolute: 1.45)')

# ---------------------------------------------------------------------------- #

def plot_SSE_distribution(compare_SSEs, p, model_strings, figure_description):
    """Take a look at the histogram (and save it) of the model SSEs for each hypothesis"""

    plt.figure()
    model_colors = ['dodgerblue', 'red']  # blue for normalised/desired model, pink for alternate
    for which_hypothesis in range(len(compare_SSEs)):
        SSE = compare_SSEs[which_hypothesis]
        plt.hist(SSE, color=model_colors[which_hypothesis], alpha=0.6)

    ax = plt.gca()
    legend = ax.legend(model_strings)
    p_string = 'p = {:.3e}'.format(p)
    plt.text(np.median(np.asarray(compare_SSEs).flatten()), 3, p_string)
    plt.xlabel('SSE')
    plt.ylabel('Model instances/subjects')
    plt.title(figure_description)
    plt.savefig(os.path.join(FIGURE_LOC, figure_description) + '.jpg', bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------- #

def get_figure_description(caption, fit_args):

    fit_algorithm, fit_method, model_system, keep_parallel, div_norm, sub_norm, which_set = fit_args
    kept_parallel_text = '_keptparallel' if keep_parallel else '_notkeptparallel'
    return  'SSE_distribution_' + model_system + '_' + caption + '_' + fit_method + kept_parallel_text + '_' + fit_algorithm + '_' + which_set

# ---------------------------------------------------------------------------- #

def main():

    fit_method = 'cross_val'     # 'fit_to_mean'  'individual' 'cross_val' 'cross_val_5fold'
    model_system = 'EEG' # 'RNN' or 'EEG'
    which_set = 'chris'  # the distance which_set for the data (note that the lines model will be in euclidean space)
    fit_algorithm = 'default'  # 'default'  'Nelder-Mead'

    #for subject in range(len(SSE)):
    #    view_fit(SSE[subject], parameters[subject])

    # Assess parallelness of unconstrained lines
    keep_parallel = False
    div_norm = 'unconstrained' # 'unconstrained' 'normalised'  'absolute'
    sub_norm = 'unconstrained' # 'unconstrained' 'centred'  'offset'
    fit_args = [fit_algorithm, fit_method, model_system, keep_parallel, div_norm, sub_norm, which_set]
    SSE, parameters = load_fits(fit_args)
    parallelness_test(model_system, parameters, fit_args)
    plot_divisive_normalisation(model_system, parameters, fit_args)

    # Assess SSE of parallel vs not parallel fits
    div_norm = 'unconstrained' # 'unconstrained' 'normalised'  'absolute'
    sub_norm = 'unconstrained'
    compare_SSEs = []
    for keep_parallel in [True, False]:
        fit_args = [fit_algorithm, fit_method, model_system, keep_parallel, div_norm, sub_norm, which_set]
        SSE, parameters = load_fits(fit_args)
        compare_SSEs.append(SSE)

    tstat, p = stats.ttest_rel(compare_SSEs[0], compare_SSEs[1])
    plot_SSE_distribution(compare_SSEs, p, ['parallel','unconstrained lines'], get_figure_description('parallelness', fit_args))

    print('-------------------')
    print('Parallelness SSE test:')
    print(np.mean(compare_SSEs[0]), np.mean(compare_SSEs[1]))
    print('t-statistic (2-sided t-test)= {:.3f}'.format(tstat))
    print('p =  {:.3e}'.format(p))


    # Assess divisive normalisation vs absolute line lengths
    keep_parallel = True
    sub_norm = 'unconstrained' # 'unconstrained' 'centred'  'offset'
    compare_SSEs = []
    for div_norm in ['normalised', 'absolute']:
        fit_args = [fit_algorithm, fit_method, model_system, keep_parallel, div_norm, sub_norm, which_set]
        SSE, parameters = load_fits(fit_args)
        compare_SSEs.append(SSE)

    tstat, p = stats.ttest_rel(compare_SSEs[0], compare_SSEs[1])
    plot_SSE_distribution(compare_SSEs, p, ['divisive norm','absolute length'],  get_figure_description('divisive_norm', fit_args))

    print('-------------------')
    print('Divisive normalisation test:')
    print(np.mean(compare_SSEs[0]), np.mean(compare_SSEs[1]))
    print('t-statistic (2-sided t-test)= {:.3f}'.format(tstat))
    print('p =  {:.3e}'.format(p))

    # Assess subtractive normalisation (line centring) vs totally offset lines
    keep_parallel = True
    div_norm = 'unconstrained' # 'unconstrained' 'normalised'  'absolute'
    compare_SSEs = []
    for sub_norm in ['centred', 'offset']:
        fit_args = [fit_algorithm, fit_method, model_system, keep_parallel, div_norm, sub_norm, which_set]
        SSE, parameters = load_fits(fit_args)
        compare_SSEs.append(SSE)

    tstat, p = stats.ttest_rel(compare_SSEs[0], compare_SSEs[1])
    plot_SSE_distribution(compare_SSEs, p, ['centred lines','offset lines'], get_figure_description('subtractive_norm', fit_args))

    print('-------------------')
    print('Subtractive normalisation test:')
    print(np.mean(compare_SSEs[0]), np.mean(compare_SSEs[1]))
    print('t-statistic (2-sided t-test)= {:.3f}'.format(tstat))
    print('p =  {:.3e}'.format(p))

# ---------------------------------------------------------------------------- #

main()
