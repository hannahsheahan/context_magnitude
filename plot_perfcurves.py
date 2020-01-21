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
nocontext, withcontext = [[] for i in range(2)]

# Sort training records into those for which context was marked or not
for file in files:
    dict = json.load(open(fileloc+file))

    if dict['model']=='recurrent_nocontext':
        nocontext.append(dict['trainingPerformance'])
    else:
        withcontext.append(dict['trainingPerformance'])

nc_perf = np.asarray(nocontext)
wc_perf = np.asarray(withcontext)


# Plot the training curves mean +- std results
conditions_data = [nc_perf, wc_perf]
linelabels = ['random (1-3) context markers', 'correct (1-3) context markers']
colours = ['blue', 'orange']
plt.figure()

for condition in range(len(conditions_data)):
    means = np.mean(conditions_data[condition],0)
    stds = np.std(conditions_data[condition],0)
    epochs = range(len(means))
    plt.plot(epochs, means, color=colours[condition], label=linelabels[condition])
    plt.fill_between(epochs, means-stds, means+stds, color=colours[condition], alpha=0.25)

plt.xlabel('Epochs')
plt.ylabel('Training performance')
plt.legend()
plt.savefig('figure.pdf')
