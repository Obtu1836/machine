from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def sigmoid(x):

    return 1/(1+np.exp(-x))

def fit(x,y,max_depth,lr,nums):

    f=np.zeros_like(y,np.float32)
    flag=True

    trees=[]
    for i in range(nums):
        neg_grad=y-sigmoid(f)
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

    f=sigmoid(f)
    f=np.where(f>0.5,1,0)
    return f

if __name__ == '__main__':

    com=load_breast_cancer()

    x,y=com.data,com.target

    train_x,test_x,train_y,test_y=train_test_split(x,y,train_size=0.8)

    tree=fit(train_x,train_y,5,0.1,300)

    pred=predict(tree,test_x)
    print(pred)

    print(accuracy_score(test_y,pred))

