# Script for calculating optimal performance under two different policies post-lesion:
# 1. a local policy that uses the local context median when responding whether number A>B
# 2. a global policy that uses the global number median across all contexts when responding whether A>B

# Author: Hannah Sheahan
# Date: 07/04/2020
# Notes: N/A
# Issues: N/A
# ---------------------------------------------------------------------------- #
import constants as const
import statistics as stats
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------- #

# Define the ranges of primary targets displayed in each context
localxranges = [[const.FULLR_LLIM,const.FULLR_ULIM], [const.LOWR_LLIM,const.LOWR_ULIM], [const.HIGHR_LLIM,const.HIGHR_ULIM]]
globalxrange = [i for contextrange in localxranges for i in range(contextrange[0], contextrange[1]+1)]
globalmedian = stats.median(globalxrange)

# record performance as a function of distance between current number and context (or global) median
policies = ['global','local']
absDistToMedianPerformance = np.zeros((len(policies),const.NCONTEXTS,16)) # (each element in dim3 encodes 0.5 numerosity difference)
absDistToGlobalMedianPerformance = np.zeros((len(policies),const.NCONTEXTS,16))
absDistToMedianPerformance[:] = np.nan
absDistToGlobalMedianPerformance[:] = np.nan

for whichpolicy, policy in enumerate(policies):
    print('Testing policy: '+ policy)
    Ptotal = 0
    for whichrange in range(len(localxranges)):
        xmin = localxranges[whichrange][0]
        xmax = localxranges[whichrange][1]

        # Possible values for xA and xB
        xvalues = list(range(xmin, xmax+1))
        if policy == 'local':
            xmedian = stats.median(xvalues)
        else:
            xmedian = globalmedian
        Na = len(xvalues)
        Nb = Na-1         # xA never equals xB
        P_a = 1/Na        # uniform distribution for sampling a from xA

        print('Range {}, x values:'.format(whichrange))
        print(xvalues)
        print('median for policy evaluation: {}'.format(xmedian))

        Pcorrect = 0
        for a in xvalues:
            P_agreaterB = (a-xmin)/Nb
            if (a<=xmedian):
                Pcorrect_a = (1 - P_agreaterB)
            else:
                Pcorrect_a = P_agreaterB
            Pcorrect += P_a*Pcorrect_a

            # distance of current number a to median (local or global)
            absDistToMed = int(abs(a-stats.median(xvalues))*2)
            absDistToGlobalMed = int(abs(a-globalmedian)*2)
            absDistToMedianPerformance[whichpolicy,whichrange,absDistToMed] = Pcorrect_a
            absDistToGlobalMedianPerformance[whichpolicy,whichrange,absDistToGlobalMed] = Pcorrect_a


        print(('{:.2f}% correct for range {}, under policy '+policy).format(Pcorrect*100, whichrange))
        print('')
        Ptotal += Pcorrect

    Ptotal /= const.NCONTEXTS
    print('Mean performance across all 3 ranges with ' + policy + ' policy: {:.2f}%'.format(Ptotal*100))
    print('---\n')

# ---------------------------------------------------------------------------- #

# Now evaluate performance under each policy as a function of numerical each distance to median
contextcolours = [['darkgoldenrod','navy','darkred'],['gold', 'dodgerblue', 'orangered'] ]  # 1-16, 1-11, 6-16 like fabrices colours
titles = ['global policy model', 'local policy model']
fig,ax = plt.subplots(1,2)

for whichpolicy, policy in enumerate([ 'global','local']):

    handles = []
    #  mean vs global median
    tmp = list(np.nanmean(absDistToGlobalMedianPerformance[whichpolicy],0))
    distvalues = np.linspace(0,7.5,len(tmp))
    perf = [i for i in tmp if not np.isnan(i)]
    distmedian = [distvalues[i] for i in range(len(tmp)) if not np.isnan(tmp[i])]
    h, = ax[whichpolicy].plot(distmedian, perf, color='black')
    handles.append(h)

    #  mean vs local median
    tmp = list(np.nanmean(absDistToMedianPerformance[whichpolicy],0))
    distvalues = np.linspace(0,7.5,len(tmp))
    perf = [i for i in tmp if not np.isnan(i)]
    distmedian = [distvalues[i] for i in range(len(tmp)) if not np.isnan(tmp[i])]
    h, = ax[whichpolicy].plot(distmedian, perf, color='grey')
    handles.append(h)

    for whichrange in range(len(localxranges)):
        tmp = list(absDistToMedianPerformance[whichpolicy][whichrange])
        distvalues = np.linspace(0,7.5,len(tmp))
        perf = [i for i in tmp if not np.isnan(i)]
        distmedian = [distvalues[i] for i in range(len(tmp)) if not np.isnan(tmp[i])]
        h, = ax[whichpolicy].plot(distmedian, perf, color=contextcolours[whichpolicy][whichrange])
        handles.append(h)
        ax[whichpolicy].set_ylim([0.27, 1.03])
        ax[whichpolicy].set_title(titles[whichpolicy])
        ax[0].set_ylabel('Theoretical performance post-lesion')
        ax[whichpolicy].set_xlabel('|current# - '+r'$\~x|$')
        #mylist = [i for i in tmp if i!=0]
        ax[whichpolicy].legend(handles,['mean vs global '+r'$\~x$','mean vs local '+r'$\~x$','full range vs local '+r'$\~x$','low range vs '+r'$\~x$','high range vs '+r'$\~x$'])
plt.savefig('model_disttomedian.pdf',bbox_inches='tight')
