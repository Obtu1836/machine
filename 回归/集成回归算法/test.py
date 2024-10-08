import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor  
from sklearn.metrics import r2_score 

def fit(x,y,n,lr):
    f=np.zeros_like(y)
    flag=True
    trees=[]
    for i in range(n):
        neg_grad=y-f
        tree=DecisionTreeRegressor(max_depth=3)
        tree.fit(x,neg_grad)
        trees.append(tree)
        pred=tree.predict(x)
        if flag:
            f=pred
            flag=False
        else:
            f+=lr*pred

    return trees

def predict(x,trees,lr):

    f=0
    flag=True
    for tree in trees:
        pred=tree.predict(x)
        if flag:
            f=pred
            flag=False
        else:
            f+=lr*pred
    return f


if __name__ == '__main__':
    data,label=make_regression(1000,3,random_state=0)
    x_train,x_test,y_train,y_test=train_test_split(data,label,
                                                   train_size=0.8,
                                                   random_state=0)
    lr=0.1
    trees=fit(x_train,y_train,300,lr)

    preds=[predict([var],trees,lr) for var in x_test]

    print(r2_score(y_test,preds))







