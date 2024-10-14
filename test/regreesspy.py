import numpy as np
from numpy.linalg import norm
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

np.random.seed(1)

def plus(data,k=2):
    cents=[data[np.random.randint(len(data))]]

    while len(cents)<k:
        dis=np.min(norm(data[:,None]-cents,axis=2),axis=1)
        dis=np.power(dis,2)

        prob=dis/dis.sum()
        cum=np.cumsum(prob)

        sign=np.random.random()
        for j,p in enumerate(cum):
            if p>sign:
                cents.append(data[j])
                break
    return np.array(cents)


def km(data,k=2):

    m,n=data.shape
    tags=np.zeros((m,2))
    cents=plus(data,k)

    flag=True
    while flag:
        flag=False

        distance=norm(data[:,None]-cents,axis=2)
        dis=np.min(distance,axis=1)
        tag=np.argmin(distance,axis=1)

        if not((tags[:,0]==tag).all()):
            flag=True
        
        tags[:,0]=tag
        tags[:,1]=dis

        df=pd.DataFrame(data)
        cents=df.groupby(tag).apply(lambda x:x.mean(axis=0)).values 
    
    return cents,tags


def split(data,k):

    m,n=data.shape
    tags=np.zeros((m,2))

    cents=[data[np.random.randint(len(data))]]

    while len(cents)<k:
        sse=np.inf
        for i in range(len(cents)):
            cur_data=data[tags[:,0]==i]
            sp_cent,sp_tag=km(cur_data,2)
            sp_sse=sp_tag[:,1].sum()
            non_sp_sse=tags[tags[:,0]!=i,1].sum()
            new_sse=sp_sse+non_sp_sse
            
            if new_sse<sse:
                sse=new_sse
                sign=i
                mid_cent=sp_cent
                mid_tag=sp_tag

            

        mid_tag[mid_tag[:,0]==1,0]=len(cents)
        mid_tag[mid_tag[:,0]==0,0]=sign
        tags[tags[:,0]==sign,:]=mid_tag

        cents[sign]=mid_cent[0]
        cents.append(mid_cent[1])

    return np.array(cents),tags


if __name__ == '__main__':
    
    k=5
    x,y=make_blobs(300,2,centers=k)

    plt.scatter(x[:,0],x[:,1],c=y)
    cents,tags=split(x,k)
    plt.scatter(cents[:,0],cents[:,1],marker='*',color='r',s=100)
    plt.show()

