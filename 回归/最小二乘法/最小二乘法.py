import numpy as np 
from numpy.linalg import inv
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

def least_squares(x,y):

    x=np.c_[x,[1]*len(x)]
    y=y[:,None]

    w=inv(x.T.dot(x)).dot(x.T).dot(y)

    return w



if __name__ == '__main__':
    
    x,y=make_regression(50,1,noise=10)

    w=least_squares(x,y)

    xs=np.linspace(x.min(),x.max())

    xss=np.c_[xs[:,None],[1]*len(xs)]

    ys=xss.dot(w)

    plt.scatter(x.flatten(),y.ravel())
    plt.plot(xs,ys.ravel(),color='r')
    plt.show()

