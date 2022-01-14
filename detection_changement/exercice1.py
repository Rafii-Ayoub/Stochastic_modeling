
# Importing the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline



mat = scipy.io.loadmat('X_pluv.mat')
matrix = mat["X_pluv"]
def show_matrix():
    print(matrix)

def calcul_covariance():
    c1=np.cov(matrix[0],matrix[1])
    c2=np.cov(matrix[1],matrix[2])
    c3=np.cov(matrix[2],matrix[0])
    print("Covriance Ville 1/ Ville 2", c1)
    print("Covriance Ville 2/ Ville 3", c2)
    print("Covriance Ville 3/ Ville 1", c3)


def dessiner_histo():
     plt.hist2d(matrix[2],matrix[0])
     plt.show()


def dessiner_ddp_echantillon():
   N = matrix.size
   n = N//10
   p, x = np.histogram(matrix, bins=n) # bin it into n = N//10 bins
   x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
   f = UnivariateSpline(x, p, s=n)
   plt.plot(x, f(x))
   plt.show()