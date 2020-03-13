# Constant values for importing into magnitude/context remapping project
# Author: Hannah Sheahan, sheahan.hannah@gmail.com
# Date: 13/03/2020

# Total maximum numbers for one-hot coding
TOTALMAXNUM = 16    # max numerosity
NCONTEXTS = 3       # max number of contexts for one-hot coding

# define upper and lower limits for each # range)
FULLR_LLIM = 1      # full # range, lower limit
FULLR_ULIM = 16     # full # range, upper limit
LOWR_LLIM = 1       # low # range, lower limit
LOWR_ULIM = 11      # low # range, upper limit
HIGHR_LLIM = 6      # high # range, lower limit
HIGHR_ULIM = 16     # high # range, upper limit

# the resulting range spans
FULLR_SPAN = FULLR_ULIM - FULLR_LLIM +1
LOWR_SPAN = LOWR_ULIM - LOWR_LLIM +1
HIGHR_SPAN = HIGHR_ULIM - HIGHR_LLIM +1

# trial types
TRIAL_FILLER  = 0
TRIAL_COMPARE = 1
