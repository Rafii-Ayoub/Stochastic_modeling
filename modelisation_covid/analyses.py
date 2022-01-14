import numpy as np
from functions import *

class analyses:

    def __init__(self,dataset):
        self.dataset = dataset

    def covariance_confirmed(self,countries):
        n=len(self.dataset.select_country(countries[0]))
        p=len(countries)
        list = np.zeros([p,n])
        for i in range(p):
            column = self.dataset.select_country(countries[i])["Confirmed"].to_numpy()
            y = np.zeros(len(column))
            for k in range(len(column) - 1):
                y[k + 1] = column[k + 1] - column[k]
            column = y
            for j in range(n):
                list[i][j]= column[j]

        return empirique_covariance(list,0,500)



    def covariance_recovered(self,countries):
        n=len(self.dataset.select_country(countries[0]))
        p=len(countries)
        list = np.zeros([p,n])
        for i in range(p):
            column = self.dataset.select_country(countries[i])["Recovered"].to_numpy()
            y = np.zeros(len(column))
            for k in range(len(column) - 1):
                y[k + 1] = column[k + 1] - column[k]
            column = y
            for j in range(n):
                list[i][j] = column[j]
        return empirique_covariance(list, 0, 500)

    def covariance_deaths(self,countries):
        n=len(self.dataset.select_country(countries[0]))
        p=len(countries)
        list = np.zeros([p,n])
        for i in range(p):
            column = self.dataset.select_country(countries[i])["Deaths"].to_numpy()
            y = np.zeros(len(column))
            for k in range(len(column) - 1):
                y[k + 1] = column[k + 1] - column[k]
            column = y
            for j in range(n):
                list[i][j] = column[j]
        return empirique_covariance(list, 0, 500)





