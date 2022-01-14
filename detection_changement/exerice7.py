# Importing the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


mat = scipy.io.loadmat('X_pluv.mat')
matrix = mat["X_pluv"]

def demo_entropie_locale():
    x = entropy_local(100, matrix[1])
    y = np.arange(100, x.size + 100)

    plt.title("Approche entropie locale")
    plt.xlabel("date")
    plt.ylabel("max entropy value")
    plt.plot(y, x)
    plt.show()


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



