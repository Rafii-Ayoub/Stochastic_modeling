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

def L_log_city(city,tm,tn):
    city = city[tm:tn]
    mean = city.mean()
    var = city.var()
    N=tn-tm
    denominator = var ** (-N)
    exposant = 0
    for tk in range(tn,tm):
        exposant+= (0.5*(city[tk] - mean)**2)/var
    # En analyse monovarié on a une seule ville dpnc K=1
    K=1
    return np.log(denominator) + np.log(2*np.pi) + N + K + exposant

def L_global_city(city,t0,tf):
    N=(tf-t0-4)
    result = np.empty(N)
    for k in range(t0+2,tf-2):
        result[k-t0-2]= L_log_city(city, t0, k) + L_log_city(city, k+1, tf)

    return result

def vraissamblance_max(city,t0,tf):
    result=np.zeros(2)
    l=0
    date=0
    matrix=L_global_city(city,t0,tf)
    n = len(matrix)
    for k in range(n):
        if l<matrix[k]:
            l=matrix[k]
            date=k
    result[0]=l
    result[1]=int(date)
    return result

def fusion_vraissamblance(matrix):
    L=np.zeros(len(matrix))
    i=0
    for city in matrix:
        L[i]=vraissamblance_max(city,10,1000)[1]
        i+=1
    return L

def analyses():
    print('Moyenne  de la date de changement : ',fusion_vraissamblance(matrix).mean(), '\n Variance de la date de changement : ',fusion_vraissamblance(matrix).var())


#print(L_global_city(matrix[0],10,10000))
#print(fusion_vraissamblance(matrix))
