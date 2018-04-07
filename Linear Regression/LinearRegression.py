# coding=utf-8

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_data(path):
    data = np.genfromtxt(path, delimiter=",")
    x = data[:,0]
    y = data[:,1]
    return (x,y)

def cost_function(theta, x, y):
    m = len(y)
    J = (1/(2*m)) * np.sum((x.dot(theta)-y)**2)
    return J

def load():
    pathx = r'C:\Users\fuko\Documents\machine learning\one variable\ex2x.txt'
    pathy = r'C:\Users\fuko\Documents\machine learning\one variable\ex2y.txt'
    X = np.genfromtxt(pathx)
    Y = np.genfromtxt(pathy)
    Y = Y.reshape((len(Y)), 1)
    X = np.column_stack((X, np.ones_like(Y)))
    return X, Y

def batch_gradient_descent(alpha, theta, x, y,num_iters):
    x_trans = np.transpose(x)
    m = len(y)
    J_history = np.zeros(num_iters)
    theta0_value = np.zeros(num_iters)
    theta1_value = np.zeros(num_iters)
    for i in range(num_iters):
        hyposis = x.dot(theta)
        loss = hyposis - y
        gradient = np.dot(x_trans,loss)/m
        theta = theta - alpha*gradient
        theta0_value[i] = theta[0]
        theta1_value[i] = theta[1]
        J_history[i] = cost_function(theta,x,y)
    return theta,J_history,theta0_value,theta1_value

def drwa_J(theta0_value,theta1_value,J_history):
    X,Y = np.meshgrid(theta0_value,theta1_value)
    Z = J_history
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='rainbow',rstride=100,cstride=100)
    plt.xlim(-1,1)
    plt.ylim(-3,3)

    #ax.contourf(X, Y, Z, cmap='rainbow')
    plt.show()

def surf(x,y):
    m =len(y)
    theta0_vals = np.linspace(-5, 5, 100)
    theta1_vals = np.linspace(-5, 5, 100)

    J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = [theta0_vals[i],theta1_vals[j]]
            t = np.asarray(t).reshape((2,1))
            J_vals[i][j] = (0.5/m)*(x.dot(t) - y).T.dot(x.dot(t) - y)

    X, Y = np.meshgrid(theta0_vals, theta1_vals)
    Z = J_vals.T
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='rainbow')
    # plt.xlim(-1, 1)
    # plt.ylim(-3, 3)
    # ax.contourf(X, Y, Z, cmap='rainbow')
    plt.show()

x, y = load()
iteration = 1500
alpha = 0.07
theta = np.zeros((2,1))
theta,J_history,theta0_value,theta1_value = batch_gradient_descent(alpha,theta,x,y,iteration)
#drwa_J(theta0_value,theta1_value,J_history)
surf(x,y)
print theta
print J_history

