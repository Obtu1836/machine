import numpy as np 
from numpy.linalg import norm 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
import pandas as pd

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


def km(data,k=2):

    cents=plus(data,k)
    m,n=data.shape
    tags=np.zeros((m,2))

    flag=True
    while flag:
        flag=False

        distance=norm(data[:,None]-cents,axis=2)
        dis=np.min(distance,axis=1)
        tag=np.argmin(distance,axis=1)

        if not (tags[:,0]==tag).all():
            flag=True
        
        tags[:,0]=tag
        tags[:,1]=dis

        df=pd.DataFrame(data)
        cents=df.groupby(tag).apply(lambda x:x.mean(axis=0)).values  

    return cents,tags


def two(data,k):

    cents=[]
    cents.append(data[np.random.randint(len(data))])

    m,n=data.shape
    tags=np.zeros((m,2))

    while len(cents)<k:
        sse=np.inf

        for i in range(len(cents)):
            cur_data=data[tags[:,0]==i]
            split_cent,split_tag=km(cur_data)

            split_sse=split_tag[:,1].sum()
            non_split_sse=tags[tags[:,0]!=i,1].sum()

            _sse=split_sse+non_split_sse

            if _sse<sse:

                sign=i
                sse=_sse
                mid_cent=split_cent
                mid_tag=split_tag

        mid_tag[mid_tag[:,0]==1,0]=len(cents)
        mid_tag[mid_tag[:,0]==0,0]=sign
        tags[tags[:,0]==sign,:]=mid_tag

        cents[sign]=mid_cent[0]
        cents.append(mid_cent[1])

    return np.array(cents),tags

if __name__ == '__main__':
    
    k=6
    data,label=make_blobs(1200,2,centers=k)

    cents,tags=two(data,k)

    plt.scatter(data[:,0],data[:,1],c=label)

    plt.scatter(cents[:,0],cents[:,1],marker='*',s=75,
                color='r')

    plt.show()


