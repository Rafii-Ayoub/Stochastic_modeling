import numpy as np

def to_column(matrix,ti):
    n = int(matrix.size / matrix[0].size)
    result = np.zeros(n)
    for k in range(n):
        result[k] = matrix[k][ti-1]
    return result

def moving_average(dataset, n=4) :
    ret = np.cumsum(dataset, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def empirique_mean(matrix,t0,tf):
    n = int(matrix.size / matrix[0].size)
    result = np.zeros(n)
    N=tf-t0
    for k in range(t0,tf):
        result += to_column(matrix,k)
    return result/N



def empirique_covariance(matrix,t0,tf):
    n = int(matrix.size / matrix[0].size)
    result = np.zeros([n,n])
    N = tf - t0
    emean = empirique_mean(matrix, t0, tf)
    for k in range(t0, tf):
        result+= np.outer((to_column(matrix,k) - emean) ,(to_column(matrix,k) - emean))
    return result/N

def entropy_local(r, matrix) :
    l=matrix.size
    entropy_matrix= []
    values=[]
    for k in range(r,l-r):
        U1 = matrix[k-r:k-1]
        U2 = matrix[k+1:k+r]
        m1 = U1.mean()
        m2 = U2.mean()
        S1 = np.var(U1)
        S2 = np.var(U2)
        entropy = 0.5*(m1-m2)**2 * (1/S1**2 + 1/S2**2) + 0.5 * ((S1 / S2)** 2 + (S2 / S1)**2) -1
        values.append(k)
        entropy_matrix.append(entropy)

    return np.array(entropy_matrix)


def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)