import numpy as np 
np.random.seed(230)
from collections import deque

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)

class LogisticRegression:

    def __init__(self,lr,alpha):

        self.lr=lr
        self.lamb=alpha

    def sigmoid(self,x,w):

        return 1/(1+np.exp(-x.dot(w)))
    
    def loss(self,x,w,y):
        
        reg=self.lamb*np.power(w,2).sum()/2 # 正则项

        '''
        loss1是常规的交叉熵
        '''
        loss1=-(y*x.dot(w)+np.log(1-self.sigmoid(x,w))).sum()
        
        return loss1+reg
    
    def grad(self,x,w,y):

        return -x.T.dot(y-self.sigmoid(x,w))+self.lamb*w
    
    def fit(self,x,y):
        m,n=x.shape
        w=np.zeros((n,1))
        q=deque(maxlen=5)
        q.append(self.loss(x,w,y))

        while True:
            neww=w-self.lr*self.grad(x,w,y)
            new_loss=self.loss(x,neww,y)
            # print(new_loss)
            q.append(new_loss)
            s=np.array(q)

            if np.allclose(s.mean(),new_loss):
                break
            w=neww
        self.w=w
    
    def predict(self,test):

        yp=test.dot(self.w)
        yp=np.where(yp>0.5,1,0)
        return yp 
    
    
if __name__ == '__main__':
    com=load_breast_cancer()
    x,y=com.data,com.target

    scale=MinMaxScaler()
    x=scale.fit_transform(x)

    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,
                                                   stratify=y)
    
    logic=LogisticRegression(lr=0.01,alpha=0.9)

    logic.fit(x_train,y_train[:,None])

    yp=logic.predict(x_test)

    print(accuracy_score(y_test,yp))

    
    
