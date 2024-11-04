import numpy as np 

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Sony:

    def __init__(self,lr,k1,x,y):

        self.y=y[:,None]
        m,n=x.shape
        self.w1=np.random.rand(n,k1)
        self.w2=np.random.rand(k1,1)
        self.x=x
        self.lr=lr
    
    @staticmethod
    def sigmoid(x):

        return 1/(1+np.exp(-x))
    
    def dsigmoid(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def forward(self,x):

        self.y1=x.dot(self.w1)
        self.f1=self.sigmoid(self.y1)

        self.y2=self.f1.dot(self.w2)
        self.f2=self.sigmoid(self.y2)

    def backward(self):

        err2=(self.f2-self.y)
        dw2=self.f1.T.dot(err2)

        err1=err2.dot(self.w2.T)*self.dsigmoid(self.y1)
        dw1=self.x.T.dot(err1)

        self.w2-=self.lr*dw2
        self.w1-=self.lr*dw1
    
    def fit(self,iters):
        for i in range(iters):
            self.forward(self.x)
            self.backward()
        
    def predict(self,x):
        self.forward(x)
        yp=self.f2
        return yp.ravel()

if __name__ == '__main__':
    
    com=load_breast_cancer()
    x,y=com.data,com.target

    scale=MinMaxScaler()
    x=scale.fit_transform(x)

    x_in,x_st,y_in,y_st=train_test_split(x,y,train_size=0.7,stratify=y)

    sony=Sony(0.001,60,x_in,y_in)
    mse=[]

    ts=5000
    sony.fit(ts)

    ys=sony.predict(x_in)
    yp=np.where(ys>0.5,1,0)
    print(accuracy_score(y_in,yp))


    yss=sony.predict(x_st)
    ypp=np.where(yss>0.5,1,0)
    print(accuracy_score(y_st,ypp))



    

