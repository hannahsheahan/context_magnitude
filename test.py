import numpy as np


def fullmatrix_touppertri(x):
    # return the flattened elements and indices of the upper triangular portion of the matrix
    ind = np.triu_indices(x.shape[0])
    flatx = [x[ind[0][i], ind[1][i]] for i in range(len(ind[0]))]
    flatx = np.asarray(flatx)
    return flatx, ind

# ---------------------------------------------------------------------------- #

def flatuppertri_tofullmatrix(x, ind, n):
    uppertri = np.zeros((n,n))
    uppertri[ind] = x
    zeroed_diag = np.zeros((n,n)) + uppertri
    np.fill_diagonal(zeroed_diag, 0)
    fullmatrix = uppertri + np.transpose(zeroed_diag)
    return fullmatrix



test = np.arange(16).reshape(4, 4)
print(test)
print('-----')
a, ind = fullmatrix_touppertri(test)
print(a)
print('-----')

b = flatuppertri_tofullmatrix(a, ind, test.shape[0])
print(b)
