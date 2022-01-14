# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:14:14 2020
@contact: atto / abatt@univ-smb.fr
"""
#%%
####################################################################
#################################   EXERCICE 3   ###################
####################################################################
#                    Modèles autorégressifs
####################################################################
####################################################################
## Etude d-une serie temporelle (non-stationnaire globalement)
##    mais composee par juxtaposition de 3 blocs stationnaires
####################################################################
####################################################################
# Utiliser systematique l-aide de python (fonction help), Exemple :
#    help(int) pour obtenir l-aide sur la fonction "int"
#%% 
import random as aleas  # pour generer des nombres aleatoires et +
import matplotlib.pyplot as plt # pour les graphiques
from scipy.signal import freqz   # pour avoir le TF de l'autocorrelation
import numpy as np #pour gerer les moyennes + covariances +++
####################################################################
#############   EXERCICE 3 / Partie 1            ###################
####################################################################
n=1536 #  nombre de donnees a genener
#% Sequences des parametres des 3 modeles AR du second ordre
a=[
    [1,-0.1344,0.9025], # coefficients du premier processus AR
    [1,-1.6674,0.9025], # Ccoefficients du second processus AR
    [1, 1.7820,0.8100]  # coefficients du troisieme processus AR
]
####################################################################
## Generer une serie temporelle (non-stationnaire globalement)
##    composee de 3 blocs stationnaires
####################################################################
####################################################################
## Generation et juxtaposition des 3 blocs de largeur n/3 chacun
t=range(-2,n-1)
y=[k*0 for k in t]
for k in range(1,int(n/3)):
    y[k+1]=-a[0][1]*y[k]-a[0][2]*y[k-1]+aleas.gauss(0,1)
for k in range(int(n/3)+1,2*int(n/3)):
    y[k+1]=-a[1][1]*y[k]-a[1][2]*y[k-1]+aleas.gauss(0,1)
for k in range(2*int(n/3)+1,n):
    y[k+1]=-a[2][1]*y[k]-a[2][2]*y[k-1]+aleas.gauss(0,1)
y=y[3:]  # suppression des donnees transitoires
t=t[3:]
# Trace- de la serie
plt.title("La serie principal: juxtapososition de 3 sous-series")
plt.plot(t,y,label='Data = juxtapososition de 3 sous-series stationnaires')
plt.show()
#
###########################################################################
#%%
###########################################################################
# Calcul et trace-s des spectres des trois sous-series a partir de freqz
#  Puisque l on connait les coefficients, le calcul est fait directement
#  a- partir des coefficients (ne depend donc pas du nombre d-echantillons) 
# 
def spectre(*args): 
	Np = 256 # nombre de points du spectre
	f=freqz(1,args[0],Np)[0] # recuperer les echantillons de frequences (abscisses)
	mag=[]   # hauteurs des frequences observables correspondantes (ordonnees)
	for arg in args:
		mag.append(abs(freqz(1,arg,Np)[1])) # calcul du spectre de chaque sous-serie
	return (f,mag)

f,mag=spectre(a[0],a[1],a[2])
## Calcul des spectres des trois sous-series 
plt.semilogy(
	f,mag[0],'-g',
	f,mag[1],':b',
	f,mag[2],'--r'
)
## Traces des spectres des trois sous-series 
plt.title("Les spectres des 3 series")
plt.show()
#
###########################################################################
#%%
# On choisit de decrire y par un modele AR d-ordre 3, puis d-ordre 4.
#    Estimation des coefficients des modeles AR d-ordres 3 et 4
# re-utiliser la partie deja ecrite pour superposer les estimations locales de spectres avec le resultat escompte-

def tracer_sous_série(debut,fin):
    D = np.cov([
        y[debut:fin] + [0, 0, 0, 0], 
        [0] + y[debut:fin] + [0, 0, 0], 
        [0, 0] +y[debut:fin] + [0, 0],
        [0, 0, 0] + y[debut:fin] + [0],
        [0, 0, 0, 0] +y[debut:fin]])

    a = "["+str(debut)+":"+str(fin)+"]"
    #on trace la série entre 0 et le début de l'intervalle
    plt.plot(t[debut:fin], y[debut:fin])
    plt.title("serie  %s"  %a)
    plt.show()

def tracer_spectre(debut,fin):  
   
	D = np.cov([
        y[debut:fin] + [0, 0, 0, 0], 
        [0] + y[debut:fin] + [0, 0, 0], 
        [0, 0] +y[debut:fin] + [0, 0],
        [0, 0, 0] + y[debut:fin] + [0],
        [0, 0, 0, 0] +y[debut:fin]])
	
	E = - np.linalg.inv(D[0:4, 0:4])# car on veut l'avoir à l'ordre 4
	H = - np.linalg.inv(D[0:3, 0:3])   # car on veut l'avoir à l'ordre 3
	E1 = np.append([1], E)  # vecteur de coefficients incluant a0(ordre 3)
	H1 = np.append([1], H)
	#on trace les spectres (estimation)
	f, mag = spectre(E1, H1)
	plt.semilogy(f, mag[0],f, mag[1],':r',linewidth = 2)
	a = "["+str(debut)+":"+str(fin)+"]"
	plt.title("spectre/calcul sur l'intervalle %s" %a)
	plt.show()           
	
tracer_sous_série(0,256)
tracer_spectre(0,256)
tracer_sous_série(350,605)
tracer_spectre(350,605)
tracer_sous_série(700,955)
tracer_spectre(700,955)
tracer_sous_série(956,1211)
tracer_spectre(956,1211)