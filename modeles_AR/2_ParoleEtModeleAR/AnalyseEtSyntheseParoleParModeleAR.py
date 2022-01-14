# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:14:14 2020
@contact: atto / abatt@univ-smb.fr 
"""
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import time
import random as aleas
from scipy.signal import lfilter
import sounddevice as sd
from statsmodels.tsa.ar_model import AutoReg


""" ------------------ Importer le fichier mat et extraire les données-----------------"""

DataParole = loadmat('DataParole.mat')
DataParole = DataParole['DataParole']
wait = input("Appuyer sur une touche du clavier ")
z = DataParole


""" ------------------------------- Tracer le signal ----------------------------------"""

def tracer_signal(data):
   plt.plot(data)
   plt.title("Signal DataParole")
   plt.ylabel('Intensité en DB')
   plt.show()


""" ----------- Définir n1 et n2 le début et la fin de la série à analyser ---------------"""

n1 = 200
n2 = len(z)
n3 = n2-n1+1
y=z[n1:n2]


""" ----------- définir la longueur de la première trame et la dessiner ---------------"""
m=150
np.floor([n3/m])
ordreAR =8
y1=y[1:m]

def tracer_trame(data,index):
    plt.plot(data)
    plt.ylabel('Data Parole')
    plt.title("Trame %i" %index)
    plt.show()
    
#tracer_trame(y1,1)

""" ----------- Implementer un modele statistique autoregressive  ---------------"""

model = AutoReg(y,lags=4)
model_fitted = model.fit()


def tracer_serie_stationnaire(data,index):
    a = data.params
    z=[k*0 for k in range(150)]
    for k in range(10,150):
         z[k]=-a[0]*y1[k-1]-a[1]*y1[k-2]-a[2]*y1[k-3]-a[3]*y1[k-4]-a[4]*y1[k-5]
    plt.plot(range(len(y1[3 :])),z[4:],label='Data =series stationnaires %i' %index)
    plt.title("Serie stationnaire 1")
    plt.show()

#tracer_serie_stationnaire(model_fitted,1)


""" ----------- Dessiner la parole à l'aide du modèle ajusté -----------------"""

def tracer_parole_ajuste(fitted_model):
    plt.plot(y1[4:], 'b-', label='data')
    plt.plot(fitted_model.fittedvalues[4:], 'r-', label='data')
    plt.title("Parole ajusté")
    plt.show()

#tracer_parole_ajuste(model_fitted)

"""------------------------- Dessiner les trames réels vs estimés -----------------------------"""

a = model_fitted.params
yf1=lfilter(a[0:9],1,y1)
n = 150
res = y1- yf1
NbTramesAffichees = 10;  
m1=ordreAR+1
k=1
residuel = y1-yf1
NbTrames = int((n2-n1+1)/m)

def tracer_trames(NbTrames,data):
    
    for k in range(1,NbTrames-1):
        y2 = data[k*m -m1 + 1 : (k+1)*m]
        model = AutoReg(y2,lags=4)
        model_fitted = model.fit()
        coeffsAR = model_fitted.params
        yf2 = lfilter(coeffsAR[1:8],1,y2)
        
        if k< 10:
            plt.plot(yf2[m1:m1+m-1], 'g-', label='data')
            plt.plot(y2[m1:m1+m-1], 'b-', label='data')
            plt.title("Trame %d: Estimée vs Réalité "%k)
            plt.show()

#tracer_trames(NbTrames,y)

"""------------------- Tracer le signal reel et estimé ------------------------"""

def generer_residuel(data,residuel_data):
    residuel = residuel_data
    for k in range(1,NbTrames-1):
        y2 = data[k*m -m1 + 1 : (k+1)*m]
        model = AutoReg(y2,lags=4)
        model_fitted = model.fit()
        coeffsAR = model_fitted.params
        yf2 = lfilter(coeffsAR[1:8],1,y2)
        residuel2 = y2[m1:m1+m-1]-yf2[m1:m1+m-1]
        residuel = np.concatenate((residuel,residuel2), axis=0)
    return residuel


def tracer_Parole_reelVSestime(data_reel,data_estime):
    plt.plot(data_estime) 
    plt.title("Parole estimée")
    plt.show()
    plt.plot(data_reel) 
    plt.title("Vraie Paroles")
    plt.show()

#tracer_residuel(y,generer_residuel(y,residuel))

def synteses():
    # tracer le signal
    tracer_signal(z)
    #Définir n1 et n2 le début et la fin de la série à analyser
    n1 = 200
    n2 = len(z)
    n3 = n2-n1+1
    y=z[n1:n2]
    # Définir les parametres pour tracer la 1ère trame
    m=150
    np.floor([n3/m])
    ordreAR =8
    y1=y[1:m]
    # Tracer la première trame
    tracer_trame(y1,1)
    # Implementer un modele statistique autoregressive 
    model = AutoReg(y,lags=4)
    model_fitted = model.fit()
    tracer_serie_stationnaire(model_fitted,1)
    a = model_fitted.params
    yf1=lfilter(a[0:9],1,y1)
    # Définir les paramètres pour tracer les trames 
    m1=ordreAR+1
    residuel = y1-yf1
    NbTrames = int((n2-n1+1)/m)
    # tracer les trames
    tracer_trames(NbTrames,y)
    #tracer la parole reel vs estimée
    tracer_Parole_reelVSestime(y,generer_residuel(y,residuel))

    

if __name__ == '__main__':
    
    # Entendre le signal reel 
    sd.play(DataParole, 8192) 
    
    
    #tracer les courbes et lancer les analyses
    synteses()


    # Entendre le signal estimée 
    sd.play(residuel, 8192)
    time.sleep(3)
    sd.play(residuel, 8192)