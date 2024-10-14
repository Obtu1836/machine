import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class Tree:

    def __init__(self,col=None,val=None,leaf=None,l=None,
                 r=None):
        
        self.col=col
        self.val=val
        self.leaf=leaf
        self.l=l
        self.r=r


class Cart:

    def __init__(self,max_depth=None,min_samples=5):

        self.max_depth=max_depth
        self.min_samples=min_samples

    @staticmethod
    def mse(x,y):
        if len(x)==0:
            return 0
        return np.power(y-y.mean(),2).sum()
    
    @staticmethod
    def split(x,y,col,val):

        con=x[:,col]<val
        l_x,l_y=x[con],y[con]
        r_x,r_y=x[~con],y[~con]

        return l_x,l_y,r_x,r_y
    
    def fit(self,x_train,y_train):
        self.tree=self.build(x_train,y_train,self.max_depth,self.min_samples)

    def build(self,x,y,max_depth=None,min_samples=None):
        
        if len(np.unique(y))==1:

            return Tree(leaf=y[0])
        
        if max_depth==0 or len(x)<=min_samples:
            leaf=y.mean()
            return Tree(leaf=leaf)
        
        init=self.mse(x,y)
        diff=0

        mid_col=None
        mid_val=None
        mid_lx=None
        mid_ly=None
        mid_rx=None
        mid_ry=None
        n=x.shape[1]

        for col in range(n):
            for val in x[:,col]:
                l_x,l_y,r_x,r_y=self.split(x,y,col,val)
                l_mse=self.mse(l_x,l_y)
                r_mse=self.mse(r_x,r_y)

                new=l_mse+r_mse
                ds=init-new

                if ds>diff and len(l_x)>0 and len(r_x)>0:

                    mid_col=col
                    mid_val=val
                    mid_lx=l_x
                    mid_rx=r_x
                    mid_ly=l_y
                    mid_ry=r_y
                    diff=ds
        
        if diff>0:
            if max_depth:
                l=self.build(mid_lx,mid_ly,max_depth-1,min_samples)
                r=self.build(mid_rx,mid_ry,max_depth-1,min_samples)
            else:
                l=self.build(mid_lx,mid_ly,min_samples)
                r=self.build(mid_rx,mid_ry,min_samples)
            
            return Tree(col=mid_col,val=mid_val,l=l,r=r)
        
        else:
            return Tree(leaf=y.mean())
        
    def predict(self,tests):

        yp=np.apply_along_axis(self.pred,1,tests,self.tree)
        
        return yp

    def pred(self,var,tree):

        if tree.leaf!=None:
            return tree.leaf
        else:
            if var[tree.col]<tree.val:
                branch=tree.l
            else:
                branch=tree.r
            return self.pred(var,branch)

        
if __name__ == '__main__':

    data,label=make_regression(500,2)
    x_train,x_test,y_train,y_test=train_test_split(data,label,
                                                   train_size=0.8)
    max_depth=5
    min_samples=1
    model=Cart(max_depth,min_samples)
    model.fit(x_train=x_train,y_train=y_train)

    yp=model.predict(x_test)

    print(r2_score(y_test,yp))






        




    
    