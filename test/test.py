import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.datasets import load_breast_cancer,load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning),

class GaussBayes:


    def fit(self,x,y):

        m,n=x.shape
        y=pd.Series(y)
        self.yps=np.log(y.value_counts()/y.sum())

        col_name=[('x'+str(i+1))for i in range(n)]
        df=pd.DataFrame(x,columns=col_name)

        mean_var=df.groupby(y).agg(['mean','var'])
        mean_var=mean_var.swaplevel(axis=1)
        mean_var.sort_index(level=0,axis=1,inplace=True)
        self.mean_var=mean_var.reindex(level=1,columns=col_name)

    def two(self,df,ser):

        ind=df.name
        mean=df['mean'].values
        var=df['var'].values

        px=np.log(ss.norm(mean,np.sqrt(var)).pdf(ser))

        return px.sum()+self.yps[ind]

    
    def one(self,ser):

        px=self.mean_var.groupby(level=0).apply(self.two,ser)

        return px.idxmax()
        
    def predict(self,testx):

        return np.apply_along_axis(self.one,axis=1,arr=testx)
    
if __name__ == '__main__':
    com=load_iris()
    x,y=com.data,com.target

    x_train,x_test,y_train,y_test=train_test_split(x,y,
                                                   train_size=0.7,
                                                   stratify=y)
    

    gb=GaussBayes()
    gb.fit(x_train,y_train)

    yp=gb.predict(x_test)

    print(accuracy_score(y_test,yp))