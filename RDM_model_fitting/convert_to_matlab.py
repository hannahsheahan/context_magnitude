import numpy as np
import scipy.io
import os

print(os.getcwd())
arr = np.load('RDM_fillers_RNN_constcontextlabel_blck_seq_retainstate_n0.0_bs1_lr0.001_ep3_r200_h200_bpl120.npy')

scipy.io.savemat('matlab_RDM_fillers_RNN_constcontextlabel_blck_seq_retainstate_n0.0_bs1_lr0.001_ep3_r200_h200_bpl120.mat', mdict={'arr': arr})
