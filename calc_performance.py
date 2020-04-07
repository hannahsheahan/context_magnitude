# Script for calculating optimal performance under two different policies post-lesion:
# 1. a local policy that uses the local context median when responding whether number A>B
# 2. a global policy that uses the global number median across all contexts when responding whether A>B

# Author: Hannah Sheahan
# Date: 07/04/2020
# Notes: N/A
# Issues: N/A
# ---------------------------------------------------------------------------- #

import math
import statistics as stats

# Define the ranges of primary targets displayed in each context
# full range
FULL_LLIM = 1
FULL_ULIM = 16

# low range
LOW_LLIM = 1
LOW_ULIM = 11

# high range
HIGH_LLIM = 5
HIGH_ULIM = 16

localxranges = [[FULL_LLIM,FULL_ULIM], [LOW_LLIM,LOW_ULIM], [HIGH_LLIM,HIGH_ULIM]]
globalxrange = [i for contextrange in localxranges for i in range(contextrange[0], contextrange[1]+1)]
globalmedian = stats.median(globalxrange)

for policy in ['local', 'global']:
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
                Pcorrect += P_a*(1 - P_agreaterB)
            else:
                Pcorrect += P_a*P_agreaterB

        print(('{:.2f}% correct for range {}, under policy '+policy).format(Pcorrect*100, whichrange))
        print('')
        Ptotal += Pcorrect

    Ptotal /= 3
    print('Mean performance across all 3 ranges with ' + policy + ' policy: {:.2f}%'.format(Ptotal*100))
    print('---\n')
