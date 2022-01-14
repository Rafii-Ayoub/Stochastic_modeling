import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from math import *
from random import *
from typing import List

from scipy.integrate import quad
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi, exp

def analyse1():
    mu = 0
    sigma = 1.0

    data = np.random.normal(0, 1, 10000) * sigma + mu
    count, bins, ignored = plt.hist(data, 32, density=True, color="lightblue")

    plt.title('Generate a random variable of a normal distribution ')
    plt.grid()
    plt.show()


def analyse2():
    mu = 2.0
    sigma = 9.0

    data = np.random.normal(0, 1, 10000) * sigma + mu
    count, bins, ignored = plt.hist(data, 32, density=True, color="lightblue")
    x = np.linspace(-35.0, 35.0, 1000)
    plt.plot(x, scipy.stats.norm.pdf(x, 2, 9))
    plt.plot([2, 2], [0.0, 0.05])
    plt.title("normal_distribution \n µ=2 & V=9")
    plt.grid()
    plt.show()

def analyse3(N):
    emperique_var(100, N, 50)
    emperique_mean(100, N, 50)



def emperique_mean(min, max, pas):
        X = []
        Y = []
        V = []
        for k in range(min, max, pas):
            m = (np.random.normal(0, 1, k)*9 + 2).mean()
            X.append(m)
            Y.append(k)
        V.append(X)
        V.append(Y)
        fig, ax = plt.subplots()
        ax.plot(V[1], V[0], linewidth=2.0)
        x = [0, max]
        y = [2,2]
        ax.plot(x, y, linewidth=2.0, color="r")
        plt.show()


def emperique_var(min, max, pas):
    X = []
    Y = []
    V: list[list[None] | list[int]] = []
    for k in range(min, max, pas):
        m = (np.random.normal(0, 1, k)*3+2).var()
        X.append(m)
        Y.append(k)
    V.append(X)
    V.append(Y)
    fig, ax = plt.subplots()
    ax.plot(V[1], V[0], linewidth=2.0)
    x = [0, max]
    y = [9, 9]
    ax.plot(x, y, linewidth=2.0, color="r")
    plt.show()

