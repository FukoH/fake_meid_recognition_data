#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

'''  
Implement Linear-Regression, using Python  
'''


def loadDataSet():
    '''  
    Read data from file  
    Return:  
        x:list, [[x0(0), x1(0)], [x0(1), x1(1)] ... [x0(m), x1(m)]]  
        y:list, [y(0), y(1), ... y(m)]  
    '''
    x = []
    y = []
    dataFile = open('D:\machine learning\linear regression\data1.txt')
    for line in dataFile:
        lineData = line.strip().split(',')
        x.append([1.0, float(lineData[0])])
        y.append(float(lineData[1]))
    return (x, y)


def h(theta, x):
    '''  
    Hypothesis Function For one sample  
        theta: 个数为n的一维ndarray  
        x: 个数为n的一维ndarray  
    Return: digit  
    '''
    return theta.dot(x)


def batch_gradient_descent(alpha, theta, x, y):
    '''  
    Batch-Gradient-Descent"  
        alpha: Learning Rate  
        x: list, [[x0(0), x1(0)], [x0(1), x1(1)] ... [x0(m), x1(m)]]  
        y: list, [y(0), y(1), ... y(m)]  
        theta: 默认为np.array([0]*n, dtype=np.float)  
    Return:  
        newTheta: 训练后的模型参数，个数为n的一维ndarray  
    '''
    m, n = x.shape
    newTheta = np.array([0]*n, dtype=np.float)
    for j in range(n):
        count = 0.0
        for i in range(m):
            # x[i,:]取x第i行，形成n个元素的一维矩阵
            count += (h(theta, x[i,:]) - y[i]) * x[i, j]
        newTheta[j] = theta[j] - alpha * count / m
    return newTheta


def normal_equation(x, y):
    '''  
    Normal Equation  
    '''
    return np.linalg.inv(np.transpose(x).dot(x)).dot(np.transpose(x)).dot(y)


def cost_function(theta, x, y):
    """  
    Cost Function  
        theta: 模型参数，个数为n的一维ndarray  
        x: m*n的二维ndarray  
        y: 个数为m的一维ndarray  
        x.dot(theta): 个数为m的一维矩阵  
    """
    m = x.shape[0]
    return (x.dot(theta) - y).dot(x.dot(theta) - y) / (2*m)


def test():
    '''  
    Test Function  
    '''
    x, y = loadDataSet()
    x = np.array(x)
    y = np.array(y)
    m, n = x.shape
    theta = np.array([0]*n, dtype=np.float)
    costs = []
    for iters in range(100):
        costs.append(cost_function(theta, x, y))
        theta = batch_gradient_descent(0.01, theta, x, y)
    print 'Batch-Gradient-Descent:', '\ncost:\n', costs
    print 'theta: ', theta
    print 'Hypothesis: ', h(theta, np.array([1.0, 5.4994])), '\n'

    print 'Normal-Equation:'
    theta = normal_equation(x, y)
    print 'theta: ', theta
    print 'Hypothesis: ', h(theta, np.array([1.0, 5.4994]))


if __name__=='__main__':
    test()