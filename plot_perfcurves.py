# A quick plotter of training performance curves when context is labelled for the different number ranges or shuffled across examples
# Author: Hannah Sheahan, sheahan.hannah@gmail.com
# Date: 21/01/2020
# Issues: N/A
# Notes: N/A


import json
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


# Load all our training records
fileloc = 'trainingrecords/'
files = [f for f in listdir(fileloc) if isfile(join(fileloc, f)) and f != '.DS_Store']
constantcontext, truecontext, randcontext = [[] for i in range(3)]

# Sort training records into those for which context was marked or not
for file in files:
    dict = json.load(open(fileloc+file))

    if dict['model']=='recurrent_constantcontext':
        constantcontext.append(dict['trainingPerformance'])
    elif dict['model']=='recurrent_truecontext':
        truecontext.append(dict['trainingPerformance'])
    elif dict['model']=='recurrent_randcontext':
        randcontext.append(dict['trainingPerformance'])

constc_perf = np.asarray(constantcontext)
truec_perf = np.asarray(truecontext)
randc_perf = np.asarray(randcontext)

# Plot the training curves mean +- std results
conditions_data = [ truec_perf, randc_perf, constc_perf]

linelabels = [ 'true (1-3) context markers', 'random (1-3) context markers', 'constant (1) context markers']
colours = ['blue', 'orange', 'purple']
plt.figure()

for condition in range(len(conditions_data)):
    means = np.mean(conditions_data[condition],0)
    ses = np.std(conditions_data[condition],0) / np.sqrt(len(constantcontext))
    epochs = range(len(means))
    plt.plot(epochs, means, color=colours[condition], label=linelabels[condition])
    plt.fill_between(epochs, means-ses, means+ses, color=colours[condition], alpha=0.25)
#plt.plot(epochs, np.ones(means.shape)*50, color='grey', linestyle=':')

plt.xlabel('Epochs')
plt.ylabel('Training performance %')
plt.legend()
plt.xlim((-0.5,8))
plt.savefig('context_training_comparison_shuffleddataset.pdf')
