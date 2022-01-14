# Importing the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


mat = scipy.io.loadmat('X_pluv.mat')
matrix = mat["X_pluv"]

""" --------- Approche globale -------------------------"""

def to_column(matrix,ti):
    n = int(matrix.size / matrix[0].size)
    result = np.zeros(n)
    for k in range(n):
        result[k] = matrix[k][ti-1]
    return result

#print(to_column(matrix,2) ,to_column(matrix,3) )



def empirique_mean(matrix,t0,tf):
    n = int(matrix.size / matrix[0].size)
    result = np.zeros(n)
    N=tf-t0
    for k in range(t0,tf):
        result += to_column(matrix,k)
    return result/N
#print(empirique_mean(matrix,2,5))


def empirique_covariance(matrix,t0,tf):
    n = int(matrix.size / matrix[0].size)
    result = np.zeros([n,n])
    N = tf - t0
    emean = empirique_mean(matrix, t0, tf)
    for k in range(t0, tf):
        result+= np.outer((to_column(matrix,k) - emean) ,(to_column(matrix,k) - emean))
    return result/N
#empirique_covariance(matrix,2,5)


def D(emean1,evar1,emean2,evar2):
    A = 0.5*(emean1-emean2)
    B = np.linalg.inv(evar1-evar2)
    C = np.diag((emean1-emean2))

    return np.log10(np.absolute(  np.dot(np.dot(B, C), A)+ 0.5*np.trace(np.linalg.inv(evar1)*evar2 + np.linalg.inv(evar2)*evar1)))




def A(tm,t0,tf):
    emean1= empirique_mean(matrix,t0,tm)
    emean2= empirique_mean(matrix,tm+1,tf)
    evar1 = empirique_covariance(matrix,t0,tm)
    evar2 = empirique_covariance(matrix,tm+1,tf)
    return D(emean1,evar1,emean2,evar2)

#print(A(50,1,100))

def KL_global(matrix):
    n=matrix.size
    k = int(matrix.size / matrix[0].size)
    result = np.zeros([100-3,k])
    for k in range(2,100-1):
        result[k]= A(k,1,100)
    return result



"-------------- Approche monovarié --------------------"


def demo_entropie_globale():
    x = entropy_mono(100, matrix[1])
    y = np.arange(100, x.size + 100)

    plt.title("Approche entropie globale")
    plt.xlabel("date")
    plt.ylabel("max entropy value")
    plt.plot(y, x)
    plt.show()

""" Calcul de la divergence KL pour detecter le changement """

def entropy_mono(n, matrix) :
    l=matrix.size
    entropy_matrix= []
    values=[]
    for k in range(n,l-n):
        U1 = matrix[n:n+k]
        U2 = matrix[k+1-n:l-n]
        m1 = U1.mean()
        m2 = U2.mean()
        S1 = np.var(U1)
        S2 = np.var(U2)
        entropy = 0.5*(m1-m2)**2 * (1/S1**2 + 1/S2**2) + 0.5 * ((S1 / S2)** 2 + (S2 / S1)**2) -1
        values.append(k)
        entropy_matrix.append(entropy)

    return np.array(entropy_matrix)

""" Calcul de la valeur max de divergence et sa date """

def max_entropy_mono(n, matrix):
    l=matrix.size
    entropy_matrix= []
    max = 0
    for k in range(n,l-n):
        U1= matrix[0:k-1]
        U2 = matrix[k-1:l-n]
        m1 = U1.mean()
        m2 = U2.mean()
        S1 = np.var(U1)
        S2 = np.var(U2)
        entropy = 0.5*(m1-m2)**2 * (1/S1**2 + 1/S2**2) + 0.5 * ((S1 / S2)** 2 + (S2 / S1)**2) -1

        if max<entropy:
            max =entropy
    return max

""" Calcul de la valeur max de divergence et sa date pour chaque ville """

def max_entropy_by_city(matrix):
    l=[]
    n =matrix.size
    for k in range(n):
       x=entropy_mono(10, matrix[k])
       m = np.argmax(x)
       print(m)
       l.append(m)
    return np.array(entropy_matrix)






