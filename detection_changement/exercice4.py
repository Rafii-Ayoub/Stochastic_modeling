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


def to_column(matrix,ti):
    n = int(matrix.size / matrix[0].size)
    result = np.zeros(n)
    for k in range(n):
        result[k] = matrix[k][ti-1]
    return result

#print(to_column(matrix,2) ,to_column(matrix,3) )

def stations_matrix(matrix,t0,tf):
    p = int(matrix.size/matrix[0].size)
    n = tf-t0
    result = np.zeros([n,p])
    for k in range(t0,tf):
        result[k-t0] = to_column(matrix,k)
    return result


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


def L_log(matrix,tm,tn):
    K=int(matrix.size/matrix[0].size)
    ecov = empirique_covariance(matrix,tm,tn)
    N=tn-tm
    denominator = la.norm(ecov) ** (-N)
    exposant = np.empty(K)
    for tk in range(tn,tm):
        A = to_column(matrix, 7) - empirique_mean(matrix, 2, 9)
        B = np.linalg.inv(np.linalg.inv(empirique_covariance(matrix, 2, 9)))
        C = np.diag(A)
        exposant+= 0.5* np.dot(np.dot(B, C), A)

    return np.log(denominator) + exposant

def L_global(matrix,t0,tf):
    N=(tf-t0-4)
    result = np.empty(N)
    L=0
    # Creer une matrice colonne qui contient les normes de chaque matrice colonne calculé
    # Chaque matrice colonne représente la matrice de vraisamblance eu point m
    for k in range(t0+2,tf-2):
        L= L_log(matrix, t0, k) + L_log(matrix, k+1, tf)
        result[k-t0-2]=la.norm(L)

    return result


def demo_exo4_matrice_des_vraisamblances():
    result = np.zeros(9900)
    for k in range(998):
        p =k*100
        result[p:100+p]= L_global(matrix,p,104+p)
    result[9801:9900] =L_global(matrix,9801,9904)
    print(result)

def demo_exo4_tracage_courbe():
    result = np.zeros(9900)
    for k in range(998):
        p = k * 100
        result[p:100 + p] = L_global(matrix, p, 104 + p)
    result[9801:9900] = L_global(matrix, 9801, 9904)
    x = result
    y = np.arange(0, x.size )

    plt.title("Approche vraisamblance globale")
    plt.xlabel("date")
    plt.ylabel("max entropy value")
    plt.plot(y, x)
    plt.show()


def demo_exo4_tracage_courbe2():
    result = np.zeros(1000)
    for k in range(9):
        p = k * 100
        result[p:100 + p] = L_global(matrix, p, 104 + p)
    result[901:1000] = L_global(matrix, 901, 1004)
    x= result
    y = np.arange(0, x.size)
    plt.title("Approche vraisamblance locale")
    plt.xlabel("date")
    plt.ylabel("vraisamblance")
    plt.plot(y, x)
    plt.show()

#demo_exo4_tracage_courbe2()

