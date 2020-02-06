import numpy as np
import scipy.io

arr = np.load('constantcontextlabel_recurrentnet_meanactivations.npy')

scipy.io.savemat('matlab_constantcontextlabel_recurrentnet_meanactivations.mat', mdict={'arr': arr})
