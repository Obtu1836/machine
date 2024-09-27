import numpy as np
from numpy.linalg import norm 
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter,defaultdict
from sklearn.metrics import confusion_matrix

np.random.seed(0)
def plus(data,k):
    '''
    这段代码 可以不用看 模拟kmean++的操作  用来尽可能分散初始位置
    '''

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


def kmean(data,k):

    cents=plus(data,k)
    m,n=data.shape

    tags=np.zeros((m,2))

    flag=True
    while flag:
        flag=False

        distane=norm(data[:,None]-cents,axis=2)
        dis=np.min(distane,axis=1)
        tag=np.argmin(distane,axis=1)

        if not (tags[:,0]==tag).all():
            flag=True
        tags[:,0]=tag
        tags[:,1]=dis

        df=pd.DataFrame(data)
        cents=df.groupby(tag).apply(lambda x:x.mean(axis=0)).values 

    return cents,tags

def f(arr):

    a,b=np.unique(arr,return_counts=1)
    s=dict(zip(a,b))

    return s

def px(arr1,arr2):

    s1=f(arr1)
    s2=f(arr2)

    k1=sorted(s1.items(),key=lambda x:x[1] )
    k2=sorted(s2.items(),key=lambda x:x[1])

    print(k1,k2)


def score(label,tag):

    counts=Counter(zip(label,tag))
    print(counts)
    dic=defaultdict(list)
    
    for var in np.unique(label):
        for k,v in counts.items():
            if k[0]==var:
                dic[var].append([*k,v])
    inps=[]
    for vs in dic.values():
        s=np.array(vs)
        inps.append(s[np.argmax(s[:,2])])
    
    inps=np.array(inps)
    maps=dict(zip(inps[:,0],inps[:,1]))
    print(maps)

    arr3=np.array(list(map(lambda x:maps[x],label)))

    acc=(arr3==tag).sum()/len(tag)
    print(acc)



if __name__ == '__main__':

    k=5 
    data,label=make_blobs(500,2,centers=k,random_state=1)

    cents,tags=kmean(data,k)
    

    # plt.scatter(data[:,0],data[:,1],c=label)
    # plt.scatter(cents[:,0],cents[:,1],marker='*',color='r',s=75)
    # plt.show()

    
    # sk=pd.DataFrame({'label':label,'tag':tags[:,0].astype(np.int32)})

    # print(sk.head(10))

    # # px(label,tags[:,0])
    score(label,tags[:,0])

    con=confusion_matrix(label,tags[:,0])
    print()

    print(con)




