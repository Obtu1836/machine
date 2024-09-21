import numpy as np
from collections import deque
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def loss(x,w,y):

    y=y[:,None]
    t=x.dot(w)-y

    return 0.5*(t.T.dot(t))

def grad(x,w,y):
    
    y=y[:,None]
    return x.T.dot(x.dot(w)-y)


if __name__ == '__main__':

    x,y=make_regression(500,3)
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=0)
    m,n=x_train.shape
    dq=deque(maxlen=5)

    w=np.zeros((n,1))
    lr=0.001
    los=loss(x_train,w,y_train)
    dq.append(los)

    while True:
        w=w-lr*grad(x_train,w,y_train)
        new_los=loss(x_train,w,y_train)

        dq.append(new_los)
        rs=np.array(dq)

        if np.allclose(rs,rs.mean(keepdims=1)):
            break

    y_p=x_test.dot(w)

    print(np.allclose(y_p.ravel(),y_test,atol=1e-2))



    
