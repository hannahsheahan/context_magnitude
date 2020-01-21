"""
This script contains model RDM matrices for comparing the emergent representations
produced by a RNN trained on a relative magnitude task.
Author: Hannah Sheahan, sheahan.hannah@gmail.com
Date: 22/12/2019
Notes: N/A
Issues: N/A
"""
# ---------------------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import animation
from sklearn.metrics import pairwise_distances

N = 25 # the number of items to compare (1-15, context A; 1-10 context B; 5-15 context C)
