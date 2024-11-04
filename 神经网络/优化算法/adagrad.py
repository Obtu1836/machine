import numpy as np 
from collections import deque

np.random.seed(102)

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
        r=0
        beta=0.99
        i=1
        while True:
            
            '''
            原始的adagrad 并不一定能够减少迭代次数 因为梯度平方和累加的
            使r一直在变大 从而导致学习率 也一直减少 会导致在期初的下降速度
            可能就会很慢  加入指数加权后 梯度平方和的累加速度变慢 从而解决起初下降速度慢
            的情况  所以 这种方法相比原始的adagrad好
            '''
            dw=self.grad(x,w,mask)
            r=beta*r+(1-beta)*np.power(dw,2) # 加入指数移动加权后的adagrad
            # r=r+np.power(dw,2) # 原始的adagrad
            w-=(self.lr/(np.sqrt(r)+1e-8))*dw
            # w-=self.lr*dw # 不采用 步长控制
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

    x_in,x_st,y_in,y_st=train_test_split(x,y,train_size=0.7,
                                         stratify=y)
    
    logic=Logist(0.5)
    logic.fit(x_in,y_in)

    y_s=logic.predict(x_in)
    print(accuracy_score(y_in,y_s))

    yp=logic.predict(x_st)

    print(accuracy_score(y_st,yp))
    


        
