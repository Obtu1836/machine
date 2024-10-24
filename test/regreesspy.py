import numpy as np
import pandas as pd
import scipy.stats as ss

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Gauss_bayes:

    def fit(self,x,y):
        
        n=x.shape[1]
        self.col_name=[('x'+str(i+1)) for i in range(n)]
        self.yps=np.log(pd.Series(y).value_counts()/len(y))

        df=pd.DataFrame(x,columns=self.col_name)
        mean_var=df.groupby(y).agg(['mean','var'])
        mean_var=mean_var.swaplevel(axis=1)
        mean_var.sort_index(level=0,inplace=True,axis=1)
        self.mean_var_group=mean_var.groupby(level=0)

    def two(self,df,ser):
        
        id=df.name
        mean=df['mean'].reindex(self.col_name,axis=1)
        var=df['var'].reindex(self.col_name,axis=1)
        prob=ss.norm(mean,np.sqrt(var)).pdf(ser)

        return np.log(prob).sum()+self.yps[id]

    def one(self,ser):

        ps=self.mean_var_group.apply(self.two,ser)
        
        return ps.idxmax()

    def predict(self,testx):

        res=np.apply_along_axis(self.one,axis=1,arr=testx)
        return res


if __name__ == '__main__':
    
    com=load_wine()
    x,y,=com.data,com.target

    x_train,x_test,y_train,y_test=train_test_split(x,y,
                                                   train_size=0.7,
                                                   shuffle=True,
                                                   stratify=y)
    
    gb=Gauss_bayes()
    gb.fit(x_train,y_train)

    yp=gb.predict(x_test)

    acc=accuracy_score(y_test,yp)
    print(acc)
