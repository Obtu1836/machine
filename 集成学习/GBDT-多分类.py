from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def softmax(x):

    f=np.exp(x)

    return np.divide(f,f.sum(axis=1,keepdims=True))

def fit(x,y,max_depth,lr,nums):

    flag=True

    k=len(np.unique(y))
    mask=np.eye(k)[y]
    trees=[]
    f=np.zeros_like(mask,float)
    for i in range(nums):
        neg_grad=mask-softmax(f)
        tree=DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(x,neg_grad)
        pred=tree.predict(x)

        trees.append(tree)

        if flag:
            flag=False
            f=pred
        else:
            f+=lr*pred
    return trees

def predict(tree,x,lr=0.1):

    f=np.zeros(len(x),np.float32)
    flag=True

    for base in tree:
        pred=base.predict(x)

        if flag:
            flag=False
            f=pred
        else:
            f+=lr*pred

    f=softmax(f)
    f=np.argmax(f,axis=1)
    return f

if __name__ == '__main__':

    com=load_iris()

    x,y=com.data,com.target

    train_x,test_x,train_y,test_y=train_test_split(x,y,train_size=0.8)

    tree=fit(train_x,train_y,10,0.1,300)

    pred=predict(tree,test_x)
    print(pred)

    print(accuracy_score(test_y,pred))

