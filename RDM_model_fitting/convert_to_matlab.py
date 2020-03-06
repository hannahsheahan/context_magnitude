import numpy as np
import scipy.io
import os

print(os.getcwd())
arr = np.load('RDM_compare_RNN_constcontextlabel_blck_seq_numrangeintermingled_retainstate_n0.0_bs1_lr5e-05_ep6_r200_h200_bpl120.npy')

scipy.io.savemat('RDM_compare_RNN_constcontextlabel_blck_seq_numrangeintermingled_retainstate_n0.0_bs1_lr5e-05_ep6_r200_h200_bpl120.mat', mdict={'arr': arr})
