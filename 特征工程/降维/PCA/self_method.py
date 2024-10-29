import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def method(n,x):

    tr,mid=eig(x.T.dot(x))

    ind=np.argsort(tr)[::-1][:n]
    fea=mid[:,ind]

    return (x.dot(fea))
if __name__ == '__main__':
    
    com=load_iris()
    x,y=com.data,com.target

    res=method(2,x)

    plt.scatter(res[:,0],res[:,1],c=y)
    plt.show()




