import numpy as np
import warnings
from collections import deque
from sklearn.datasets import load_iris,load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore',category=RuntimeWarning)

class LogisticRegression:

    def softmax(self,x,w):

        data=np.exp(x.dot(w))
        return np.divide(data,data.sum(axis=1)[:,None])
    
    def loss(self,mask,x,w):

        return -(mask*np.log(self.softmax(x,w))).sum()
    
    def grad(self,x,mask,w):

        return -x.T.dot(mask-self.softmax(x,w))


if __name__ == '__main__':
    
    com=load_iris()
    data,label=com.data,com.target

    scale=MinMaxScaler()

    x_train,x_test,y_train,y_test=train_test_split(data,label,
                                                   train_size=0.7,
                                                   shuffle=True,
                                                   stratify=label
                                                   )
    
    m,n=x_train.shape
    k=len(np.unique(y_train))
    w=np.zeros((n,k))
    mask=np.eye(k)[y_train]

    q=deque(maxlen=5)

    logic=LogisticRegression()

    q.append(logic.loss(mask,x_train,w))
    lr=0.001
    while True:
        w=w-lr*logic.grad(x_train,mask,w)
        new=logic.loss(mask,x_train,w)
        q.append(new)
        s=np.array(q)

        if np.allclose(s.mean(),new):
            break
    
    y_pred=np.argmax(logic.softmax(x_test,w),axis=1)
    acc=(y_test==y_pred).sum()/len(y_pred)

    print(acc)

    
