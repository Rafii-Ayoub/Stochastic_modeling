import random as aleas  
import matplotlib.pyplot as plt 
from scipy.signal import freqz   
import numpy as np 
import pandas as pd


[a1,a2]=[ -0.0707, 0.2500]
[b1,b2]=[ -1.6674, 0.9025] 
[c1,c2]= [1.7820, 0.8100]

def creer_series(n):
    t=range(0,n)
    z1=[k*0 for k in t]
    z2=[k*0 for k in t]
    z3=[k*0 for k in t]
    y=[k*0 for k in t]
    
    for k in range(2,n):
        z1[k]= y[k] + a1*y[k-1] + a2*y[k-2]+aleas.gauss(0,1)
    for k in range(1,n):
        z2[k]=y[k] + b1*y[k-1] + b2*y[k-2]+aleas.gauss(0,1)
    for k in range(1,n):
        z3[k]= y[k] + c1*y[k-1] + c2*y[k-2]+aleas.gauss(0,1)
    z1=z1[3:] 
    z2=z2[3:] 
    z3=z3[3:] 
    t=t[5:]
    return z1,z2,z3,t
   
#print(creer_series(100))


def tracer_densite_spectrale(data,index):
    ps = np.abs(np.fft.fft(data))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(len(data), time_step)
    idx = np.argsort(freqs)
    ps = np.abs(np.fft.fft(data))**2
    plt.title("Densité spectrale %s" %index )
    plt.plot(freqs[idx], ps[idx])
    plt.show()

def tracer_serie(z,t,index):
    plt.plot(t[0:len(t)], z[0:len(t)])
    plt.title("serie %s" %index )
    plt.show()

def tracer_spectre(data,index):
    p = 20*np.log10(np.abs(np.fft.rfft(data)))
    f = np.linspace(0, len(data)/2, len(p))
    plt.plot(f, p) 
    plt.title("Spectre de la série %s" %index ) 
    plt.show() 


def tracer_autocorrelation(data,index):

    plt.title("fonction d'autocorrelation %i" %index)
    plt.xlabel("Lags")
    plt.acorr(data, maxlags = 20)
    plt.show()


def tracer_courbes(n):
    z1,z2,z3,t= creer_series(n)
    i=0
    for z in [z1,z2,z3]:
        i+=1
        tracer_serie(z,t,i)
        tracer_autocorrelation(z,i)
        tracer_spectre(z,i)
        tracer_densite_spectrale(z,i)

    
tracer_courbes(100)


def serie_temporel(n):
    z1,z2,z3,t= creer_series(n)
    t = pd.DatetimeIndex(t)
    y=[k*0 for k in range(len(t))]
    for k in range(len(t)):
        y[k]=z1[k]+z2[k]+z3[k]
    data = pd.Series(y, index=t)
    data.plot()
    plt.title("Serie temporel y = z1 + z2 + z3 ")
    plt.show()
    plt.title("Fonction d'autocorrelation de la série temporelle y" )
    plt.xlabel("Lags")
    plt.acorr(data, maxlags = 20)
    plt.show()
    ps = np.abs(np.fft.fft(data))**2
    time_step = 1 / 30
    freqs = np.fft.fftfreq(len(data), time_step)
    idx = np.argsort(freqs)
    ps = np.abs(np.fft.fft(data))**2
    plt.title("Densité spectrale de la série temporelle y" )
    plt.plot(freqs[idx], ps[idx])
    plt.show()

#serie_temporel(100)

