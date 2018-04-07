# coding=utf-8

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    data = np.genfromtxt(path, delimiter=",")
    x1 = data[:,0]
    x2 = data[:,1]
    y = data[:,2]
    return (x1,x2,y)

def cost_function(theta, x, y):
    m = len(y)
    J = (1/(2*m)) * np.sum((x.dot(theta)-y)**2)
    return J

def batch_gradient_descent(alpha, theta, x, y,num_iters):
    x_trans = np.transpose(x)
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        hyposis = x.dot(theta)
        loss = hyposis - y
        gradient = np.dot(x_trans,loss)/m
        theta = theta - alpha*gradient
        J_history[i] = cost_function(theta,x,y)
    return theta,J_history

def drwa_J(J_history):
    Y = J_history
    #X = range(iter_num)
    plt.plot(Y)
    plt.show()

def test():
    x1,x2,y = load_data("D:\machine learning\linear regression\ex1data2.txt")
    x = np.column_stack(((x1 - np.mean(x1))/np.std(x1),(x2 - np.mean(x2))/np.std(x2),np.ones_like(y)))
    y = (y - np.mean(y))/np.std(y)
    y = np.array(y).reshape((len(y),1))
    iteration = 1500
    alpha = 0.001
    theta = np.zeros((3,1))
    theta,J_history= batch_gradient_descent(alpha,theta,x,y,iteration)
    print J_history

if __name__=='__main__':
    test()


def cost_function(theta,X,Y):
    m = X.shape[0]
    poly1 = -Y.T.dot(np.log(sigmoid(X.dot(theta))))
    poly2 = (1-Y).T.dot(np.log(1 - sigmoid(X.dot(theta))))
    Q = poly1[0]-poly2[0]
    #J = (1/m)*((-Y.T.dot(np.log(sigmoid(X.dot(theta))))-(1-Y).T.dot(np.log(1 - sigmoid(X.dot(theta))))))
    #grad = (1/m)*X.T.dot((sigmoid(X.dot(theta))-Y))
    return Q