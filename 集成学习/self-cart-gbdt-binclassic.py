from cart import Cart
import numpy as np
from class_tree import Deci
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class BinaryClassification:

    def __init__(self,lr,max_depth,num_bases,min_samples):

        self.lr=lr
        self.max_depth=max_depth
        self.num_bases=num_bases
        self.min_samples=min_samples
    
    @staticmethod
    def sigmoid(x):

        return 1/(1+np.exp(-x))

    def fit(self,x,y):
        f=np.zeros_like(y,dtype=np.float32)
        flag=True

        self.bases=[]
        for i in range(self.num_bases):
            net_grad=y-self.sigmoid(f)
            base=Cart(self.max_depth,self.min_samples)
            base.fit(x,net_grad)
            self.bases.append(base)
            pred=base.predict(x)

            if flag:
                flag=False
                f=pred
            else:
                f+=self.lr*pred

    def predict(self,test):

        f=np.zeros(len(test))
        flag=True
        for base in self.bases:
            pred=base.predict(test)

            if flag:
                flag=False
                f=pred
            else:
                f+=self.lr*pred
        
        yp=np.where(self.sigmoid(f)>0.5,1,0)
        return yp

if __name__ == '__main__':
    com=load_breast_cancer()
    x,y=com.data,com.target

    x_train,x_test,y_train,y_test=train_test_split(x,y,
                                                   train_size=0.8,
                                                   stratify=y,
                                                   shuffle=True)
    
    max_depth=3
    min_samples=2
    lr=0.1

    model=BinaryClassification(lr,max_depth,100,min_samples)
    model.fit(x_train,y_train)
    yp=model.predict(x_test)
    print(f'self-model: {accuracy_score(y_test,yp)}')

    model2=GradientBoostingClassifier(max_depth=max_depth,
                                      learning_rate=lr)
    model2.fit(x_train,y_train)
    ypp=model2.predict(x_test)
    print(f"sklearn-model: {accuracy_score(y_test,ypp)}")

    deci=Deci(max_depth)
    deci.fit(x_train,y_train)
    yps=deci.predict(x_test)
    print(f"Decision_tree-model: {accuracy_score(y_test,yps)}")
