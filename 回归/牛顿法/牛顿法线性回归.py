import numpy as np
from numpy.linalg import inv
from collections import deque
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def loss(x,y,w):

    return 0.5*(x.dot(w)-y).T.dot(x.dot(w)-y)


def first_derivative(x,y,w):

    return x.T.dot(x.dot(w)-y)


def second_derivative(x):

    return x.T.dot(x)


if __name__ == '__main__':
    
    x,y=make_regression(500,3)

    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7)

    m,n=x.shape
    w=np.zeros((n,1))
    dq=deque(maxlen=5)

    init_loss=loss(x_train,y_train[:,None],w)
    dq.append(init_loss)

    while True:

        fir=first_derivative(x_train,y_train[:,None],w)
        sed=second_derivative(x_train)

        w=w-inv(sed).dot(fir)

        new_loss=loss(x_train,y_train[:,None],w)
        dq.append(new_loss)

        ds=np.array(dq)

        if np.allclose(ds.mean(),ds):
            break

    yp=(x_test.dot(w)).ravel()
    print(np.allclose(yp,y_test))

        

