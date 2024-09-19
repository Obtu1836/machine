import numpy as np 
from numpy.linalg import norm 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs

def plus(data:np.array,k=2):

    cents=[]
    cents.append(data[np.random.randint(len(data))])

    for i in range(1,k):
        dis=np.min(norm(data[:,None]-cents,axis=2),axis=1)
        dis=np.power(dis,2)

        prob=dis/dis.sum()
        cum=np.cumsum(prob)

        sign=np.random.rand()

        for j,p in enumerate(cum):
            if p>sign:
                cents.append(data[j])
                break
    return np.array(cents)

