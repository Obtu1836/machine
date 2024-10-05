import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def softmax(x,w):

    f=x.dot(w)

    return  np.divide(f,f.sum(axis=1,keepdims=1))

if __name__ == '__main__':
    
    com=load_iris()
    x,y=com.data,com.target

    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,
                                                    st)
