# coding=utf-8
import math
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_data():
    X = np.genfromtxt(r'C:\Users\fuko\PycharmProjects\exercise\ex4x.txt')
    Y = np.genfromtxt(r'C:\Users\fuko\PycharmProjects\exercise\ex4y.txt')
    Y = Y.reshape((len(Y),1))

def sigmoid(x):
    return 1/(1+math.exp(-x))

def cost_function(theta,X,Y):
    m = X.shape[0]
    theta = np.reshape(len(theta),1)
    J = (1/m)*(-Y.dot(np.log(sigmoid(X.dot(theta))))
               -(np.ones_like(Y)-Y).dot(np.log(np.ones_like(sigmoid(X.dot(theta)))-sigmoid(X.dot(theta)))))
    grad = (1/m)*X.T.dot((sigmoid(X.dot(theta))-Y))
    return J,grad

