import numpy as np 
from collections import deque

np.random.seed(20)

from sklearn.datasets import load_wine,load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

class Logist:

    def __init__(self,lr,):

        self.lr=lr

    @staticmethod
    def softmax(x,w):

        f=np.exp(x.dot(w))
        res=f/f.sum(axis=1,keepdims=1)
        return res
    
    def loss(self,x,w,mask):
        m=len(x)
        t=np.log(self.softmax(x,w))
        return -(mask*t).sum()/m
    
    def grad(self,x,w,mask):
        m=len(x)

        return -x.T.dot(mask-self.softmax(x,w))/m 
    
    def fit(self,x,y):
        
        k=len(np.unique(y))
        mask=np.eye(k)[y]
        m,n=x.shape
        w=np.random.rand(n,k)
        q=deque(maxlen=5)
        init_loss=self.loss(x,w,mask)
        q.append(init_loss)
        theta=0
        beta=0.99
        i=1
        while True:
            
            '''
            下面2行 动量优化步骤 当beta==0 动量优化不起作用 
            当beta=0.9~0.99时 能明显加快迭代
            '''
            theta=beta*theta-self.lr*self.grad(x,w,mask)
            w=w+theta

            # 下式与上式等价
            # theta=beta*theta+self.lr*self.grad(x,w,mask)
            # w=w-theta

            new_loss=self.loss(x,w,mask)
            q.append(new_loss)
            s=np.array(q)
            if np.allclose(s.mean(),new_loss):
                break
            i+=1
        print(f'迭代次数：{i}')
        self.w=w

    def predict(self,x):

        y=self.softmax(x,self.w)
        yp=y.argmax(axis=1)

        return yp

if __name__ == '__main__':
    
    com=load_iris()
    x,y=com.data,com.target

    # scale=MinMaxScaler() 
    # x=scale.fit_transform(x)

    x_in,x_st,y_in,y_st=train_test_split(x,y,train_size=0.7,
                                         stratify=y)
    
    logic=Logist(0.01)
    logic.fit(x_in,y_in)

    y_s=logic.predict(x_in)
    print(accuracy_score(y_in,y_s))

    yp=logic.predict(x_st)

    print(accuracy_score(y_st,yp))
    


        
