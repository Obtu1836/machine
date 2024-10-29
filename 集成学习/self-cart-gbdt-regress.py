from cart import Cart
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class Gbdt_regress:

    def __init__(self,num_bases,lr,max_depth=None,min_samples=3):

        self.num_bases=num_bases
        self.lr=lr
        self.max_depth=max_depth
        self.min_samples=min_samples

    def fit(self,x,y):

        f=np.zeros_like(y,np.float32)
        flag=True
        self.bases=[]

        for i in range(self.num_bases):
            neg_grad=y-f
            base=Cart(self.max_depth,self.min_samples)
            base.fit(x,neg_grad)
            pred=base.predict(x)
            self.bases.append(base)

            if flag:
                flag=False
                f=pred
            else:
                f+=self.lr*pred

    def predict(self,test):
        f=0
        flag=True
        for base in self.bases:
            pred=base.predict(test)
            # print(pred)

            if flag:
                flag=False
                f=pred
            else:
                f+=pred*self.lr
        return f

if __name__ == '__main__':
    
    data,label=make_regression(500,3)
    x_train,x_test,y_train,y_test=train_test_split(data,label,
                                                   train_size=0.7)
    
    model=Gbdt_regress(300,0.1,3,3)

    model.fit(x_train,y_train)

    yp=model.predict(x_test)

    print(f"self-model: {r2_score(y_test,yp)}")


    model2=GradientBoostingRegressor(n_estimators=300,learning_rate=0.1)

    model2.fit(x_train,y_train)

    yps=model2.predict(x_test)

    print(f"sklearn-model:{r2_score(y_test,yps)}")

    cart=Cart(3)
    cart.fit(x_train=x_train,y_train=y_train)

    bp=cart.predict(x_test)

    print(f"cart-model: {r2_score(y_test,bp)}")