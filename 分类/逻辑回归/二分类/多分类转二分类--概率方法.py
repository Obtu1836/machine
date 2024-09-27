import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris,load_digits,load_wine
import warnings
from collections import deque

np.set_printoptions(precision=2,suppress=True)

warnings.filterwarnings('ignore',category=RuntimeWarning)

class Logic:

    '''执行逻辑回归操作(二分类)'''
    
    @staticmethod
    def sigmoid(x,w):

        return 1/(1+np.exp(-x.dot(w)))
    
    @staticmethod
    def loss(x,w,y):

        return -(y*x.dot(w)-np.log(1+np.exp(x.dot(w)))).sum()
    
    @staticmethod
    def grad(x,w,y):

        return x.T.dot(Logic.sigmoid(x,w)-y)
    
def predict(x,y):
    
    m,n=x.shape
    w=np.zeros((n,1))

    logic=Logic()

    q=deque(maxlen=5)
    q.append(logic.loss(x,w,y))
    lr=0.001
    while True:
        w-=lr*logic.grad(x,w,y)
        new_loss=logic.loss(x,w,y)
        q.append(new_loss)
        s=np.array(q)
        if np.allclose(s.mean(),new_loss):
            break

    return w

def select(i,dk):

    dk[:,-1]=0

    con=xy[:,-1]==i
    x=xy[con]
    x[:,-1]=1

    rex=np.r_[x,dk]
    rex=np.random.permutation(rex)

    xx,yy=np.split(rex,[-1],axis=1)
    
    w=predict(xx,yy)

    return w

def test(x,w):

    x=x[None,:]

    p=np.exp(x.dot(w))

    return p.ravel()[0]


def main():

    global xy

    scale=MinMaxScaler()
    com=load_iris()
    # com=load_digits()
    # com=load_wine()
    x,y=com.data,com.target
    x=scale.fit_transform(x)

    xy=np.random.permutation(np.c_[x,y])

    num_class=len(np.unique(y))

    dk=xy[xy[:,-1]==num_class-1]

    ws=[]
    for i in range(num_class-1):
        w=select(i,dk)
        ws.append(w)

    ind=[]
    for var in x:
        ps=[]
        for w in ws:
            p=test(var,w)
            ps.append(p)
        ps=np.array(ps)
        pk=1/(ps.sum()+1)
        l=pk*ps
        l=np.append(l,[pk])
        ind.append(l.argmax())
    ind=np.array(ind)
    print((ind==y).sum()/len(ind))


if __name__ == '__main__':
    
    main()
