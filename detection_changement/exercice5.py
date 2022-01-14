from exercice4 import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import scipy.io
from numpy import linalg as la

mat = scipy.io.loadmat('X_pluv.mat')
matrix = mat["X_pluv"]
plt.style.use('seaborn-white')


def L_local(matrix,p,t0,tf):
    result=np.empty(tf-t0)
    for tm in range (t0,tf):
        L = L_log(matrix, tm-p, tm-1) + L_log(matrix, tm+1, tm+p)

        result[tm - t0 - 2] = la.norm(L)
    return result

def test_exo5():
    result = np.zeros(1000)
    for k in range(9):
        p = k * 100
        result[p:100 + p] = L_local(matrix, 10 ,p, 100 + p)
    result[901:1000] = L_local(matrix, 10 ,901, 1000)
    x = result
    y = np.arange(0, x.size)
    plt.title("Approche vraisamblance locale")
    plt.xlabel("date")
    plt.ylabel("vraisamblance")
    plt.plot(y, x)
    plt.show()


test_exo5()





#print(L(matrix,2,5))
