# coding=utf-8

# 1.读入数据
# 2.计算损失函数
#   2.1 定义损失函数
# 3.读入数据以后计算损失函数,求解对应的参数值
from __future__ import division
import numpy as np


def load_data(path):
    data = np.genfromtxt(path, delimiter=",")
    x = data[:,0]
    y = data[:,1]
    return (x,y)


def cost_function(theta, x, y):
    m = len(y)
    J = 0
    J = 1/2 * (1/m) * sum((x.dot(theta).flatten()-y)**2)
    return J


def batch_gradient_descent(alpha, theta, x, y,num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta[0][0] = theta[0][0] - alpha * (1.0/m) * sum((x.dot(theta).flatten()-y) * x[:, 0])
        theta[1][0] = theta[1][0] - alpha * (1.0/m) * sum((x.dot(theta).flatten()-y) * x[:, 1])
        print theta[0][0],theta[1][0]
        J_history[i] = cost_function(theta,x,y)
    return J_history

def test():
    x, y = load_data("D:\machine learning\linear regression\ex1data1.txt")
    x = np.column_stack((x,np.ones_like(y)))
    iteration = 1500
    alpha = 0.01
    theta = np.zeros((2,1))
    J_history = batch_gradient_descent(alpha,theta,x,y,iteration)
    print theta
    print J_history

    #
    # #theta = np.array([0] * x.shape[1], dtype=np.float)
    # J = []
    # for i in range(200):
    #     theta = batch_gradient_descent(0.0001, theta, x, y)
    #     J.append(cost_function(theta,x,y))
    #     #print 'Cost:'+str(J[-1])
    #
    # print theta
    # print J

if __name__=='__main__':
    test()