import numpy as np 
from tqdm import tqdm
from sklearn.datasets import load_wine,load_breast_cancer,load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptrons:

    def __init__(self,lr,k1,x,y,iters):

        m,n=x.shape
        k=len(np.unique(y))
        self.lr=lr
        self.w1=np.random.rand(n,k1)
        self.w2=np.random.rand(k1,k)

        self.iters=iters
        self.x=x
        self.mask=np.eye(k)[y]

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    @staticmethod
    def dsigmoid(x):
        return x*(1-x)
    
    @staticmethod
    def softmax(x):
        f=np.exp(x)
        return f/(f.sum(axis=1,keepdims=1))
    
    @staticmethod
    def dsoftmax(mask,f):

        return f-mask
    
    def forward(self,x):

        self.y1=self.sigmoid(x.dot(self.w1))
        self.y2=self.softmax(self.y1.dot(self.w2))

    def backward(self):

        err2=self.dsoftmax(self.mask,self.y2)
        dw2=self.y1.T.dot(err2)

        err1=err2.dot(self.w2.T)*self.dsigmoid(self.y1)
        dw1=self.x.T.dot(err1)

        self.w2=self.w2-self.lr*dw2
        self.w1=self.w1-self.lr*dw1
    
    def fit(self,x):
        for i in tqdm(range(self.iters),mininterval=1e-10):
            self.forward(x)
            self.backward()
        
    def predict(self,x):

        self.forward(x)

        return self.y2

if __name__ == '__main__':
    
    com=load_wine()
    # com=load_breast_cancer()
    # com=load_digits()
    x,y=com.data,com.target

    scale=MinMaxScaler()
    x=scale.fit_transform(x)

    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,
                                                   stratify=y)
    ps=Perceptrons(0.001,30,x_train,y_train,10000)

    ps.fit(x_train)

    train_yp=ps.predict(x_train).argmax(axis=1)

    print(accuracy_score(y_train,train_yp))

    test_yp=ps.predict(x_test).argmax(axis=1)
    print(accuracy_score(y_test,test_yp))








