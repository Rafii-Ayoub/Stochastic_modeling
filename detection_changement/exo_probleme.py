import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


mat = scipy.io.loadmat('Pixel_3Series.mat')
matrix = mat['Pixel']


#print("la taille de notre image est: ", mat['Pixel'].size,"*", mat['Pixel'][0].size )

def diviser_par_instant(matrix):
    return matrix[0],matrix[1],matrix[2]

def diviser_image_en4(matrix):
    max = matrix.size
    if ( max%4==0):
        shape = int(max/4)
        return  matrix[0:shape],matrix[shape:2*shape],matrix[2*shape:3*shape],matrix[3*shape:4*shape]

#print(diviser_image_en4(matrix[0]))
#print(matrix.size)




def L_log_picture(picture,tm,tn):
    picture = picture[tm:tn]
    mean = picture.mean()
    var = picture.var()
    N=tn-tm
    denominator = var ** (-N)
    exposant = 0
    for tk in range(tn,tm):
        exposant+= (0.5*(picture[tk] - mean)**2)/var
    K=1
    return np.log(denominator) + np.log(2*np.pi) + N + K + exposant

#print(L_log_picture(matrix[0],0,100))

def L_global_picture(picture,t0,tf):
    N=(tf-t0-4)
    result = np.empty(N)
    for k in range(t0+2,tf-2):
        result[k-t0-2]= L_log_picture(picture, t0, k) + L_log_picture(picture, k+1, tf)

    return result
print(L_global_picture(matrix[0],0,256))


def vraissamblance_max(picture,t0,tf):
    result=np.zeros(2)
    l=0
    date=0
    matrix=L_global_picture(picture,t0,tf)
    n = len(matrix)
    for k in range(n):
        if l<matrix[k]:
            l=matrix[k]
            date=k
    result[0]=l
    result[1]=int(date)
    return result

#print(vraissamblance_max(matrix[0],0,256))


def fusion_vraissamblance(matrix):
    L=np.zeros(len(matrix))
    i=0
    for picture in matrix:
        L[i]=vraissamblance_max(picture,0,256)[1]
        i+=1
    return L

def demo_detection_changement_image():
         print(fusion_vraissamblance(matrix))
