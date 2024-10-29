import numpy as np 
np.random.seed(1100)
np.set_printoptions(precision=4,suppress=True)
from collections import deque
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)

class LogisticRegression:

    def __init__(self,lr,alpha):

        self.lr=lr
        self.alpha=alpha

    def sigmoid(self,x,w):

        return 1/(1+np.exp(-x.dot(w)))
    
    def loss(self,x,w,y):

        # l2=self.alpha*(np.power(w,2).sum())/2 # l2正则
        l1=self.alpha*np.abs(w).sum()    # l1 正则
        ls=-(y*x.dot(w)+np.log(1-self.sigmoid(x,w))).sum()
        return l1+ls
    
    def grad(self,x,w,y):

        # return -x.T.dot(y-self.sigmoid(x,w))+self.alpha*w # l2正则
        return -x.T.dot(y-self.sigmoid(x,w))+self.alpha*np.sign(w)
    
    def fit(self,x,y):

        m,n=x.shape
        w=np.zeros((n,1))
        q=deque(maxlen=5)
        init_loss=self.loss(x,w,y)
        q.append(init_loss)

        while True:
            w=w-self.lr*self.grad(x,w,y)
            new_loss=self.loss(x,w,y)
            q.append(new_loss)
            s=np.array(q)

            if np.allclose(s.mean(),new_loss):
                break
        self.w=w
        print(w)
    
    def predict(self,test):

        y=self.sigmoid(test,self.w).ravel()
        yp=np.where(y>0.5,1,0)
        return yp


if __name__ == '__main__':
    com=load_breast_cancer()
    x,y=com.data,com.target
    scale=MinMaxScaler()
    x=scale.fit_transform(x)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,
                                                   train_size=0.7,
                                                   stratify=y)
    logis=LogisticRegression(0.001,1e-1)
    logis.fit(x_train,y_train[:,None])
    yp=logis.predict(x_test)

    print(accuracy_score(y_test,yp))