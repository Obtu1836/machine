import numpy as np 

np.set_printoptions(precision=5,suppress=True)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

class Logistic:

    def __init__(self,bathsize,lr,iters):

        self.iters=iters
        self.lr=lr
        self.batch=bathsize

    def softmax(self,x,w):

        f=np.exp(x.dot(w))
        return f/f.sum(axis=1,keepdims=1)
    

    
    def loss(self,x,w,mask):

        return -(mask*np.log(self.softmax(x,w))).sum()/len(x)
    
    def grad(self,x,w,mask):

        return -x.T.dot(mask-self.softmax(x,w))
    
    def fit(self,x,y):
        k=len(np.unique(y))
        mask=np.eye(k)[y]
        n=x.shape[1]
        w=np.zeros((n,k))

        ins=len(x)//self.batch
        inds=[i*self.batch for i in range(1,ins+1)]
        if inds[-1]==len(x):
            inds.pop()
        data=np.split(x,inds,axis=0)
        
        lab=np.split(mask,inds,axis=0)
        da_lb=list(zip(data,lab))

        for j in range(self.iters):
            for dat,las in da_lb:
                w-=self.lr*self.grad(dat,w,las)
        
        self.w=w

    def predict(self,x):

        y=x.dot(self.w).argmax(axis=1)

        return y
        


if __name__ == '__main__':
    
    com=load_digits()
    x,y=com.data,com.target
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,
                                                   stratify=y)
    logist=Logistic(64,0.001,500)
    logist.fit(x_train,y_train)

    y_tr_p=logist.predict(x_train)
    print(accuracy_score(y_train,y_tr_p))

    y_te_p=logist.predict(x_test)
    print(accuracy_score(y_test,y_te_p))