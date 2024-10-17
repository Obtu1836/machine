import numpy as np
import pandas as pd

class NaiveBayes:

    def __init__(self,alpha=0):

        self.alpha=alpha

    @staticmethod
    def cal_feature(df):

        p=df.apply(lambda x:x.value_counts())
        return p.T
    
    def fit(self,x,y):
        n=x.shape[1]
        col_name=[('x'+str(i+1)) for i in range(n)]
        k=len(np.unique(y))
        y=pd.Series(y)
        self.yps=(y.value_counts()+alpha)/(len(y)+k)

        df=pd.DataFrame(x,columns=col_name)
        model=df.groupby(y).apply(self.cal_feature)
        model.fillna(0,inplace=True)

        ser=df.apply(lambda x:len(x.unique()))
        ser=pd.Series(np.tile(ser,k),index=model.index)
        prob=(model+self.alpha).div(model.sum(axis=1)+ser*self.alpha,axis=0)
        self.prob=np.log(prob).groupby(level=0)

    def two(self,df,ser):

        ind=df.name
        df=df.values
        ps=df[range(len(df)),ser].sum()+self.yps[ind]

        return ps


    def one(self,ser):

        p=self.prob.apply(self.two,ser)
        
        return p.idxmax()

    def predict(self,testx):

        yp=np.apply_along_axis(self.one,axis=1,arr=testx)
        
        return yp

if __name__ == '__main__':
    
    train_x = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 2, 1],
                        [0, 0, 0]])
    train_y = np.array([0, 1, 0, 1, 1])
    test_x=np.array([[1,2,1]])
    alpha=1
    nb=NaiveBayes(alpha)
    nb.fit(train_x,train_y)

    print(nb.predict(test_x))
