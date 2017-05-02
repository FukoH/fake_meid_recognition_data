#encoding = 'utf-8'
import matplotlib.pyplot as plt
import numpy as np
def load_data(path):
    data = np.genfromtxt(path, delimiter=",")
    x = data[:,0]
    y = data[:,1]
    return (x,y)

def show_pic((x,y)):
    plt.scatter(x,y)
    plt.plot(x,1.16636235*x-3.63029144)
    plt.show()

if __name__=='__main__':
    show_pic(load_data("D:\machine learning\linear regression\ex1data1.txt"))
