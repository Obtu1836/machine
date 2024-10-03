import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class Tree:

    def __init__(self,col=None,val=None,leaf=None,l=None,r=None):

        self.col=col
        self.val=val
        self.leaf=leaf
        self.l=l
        self.r=r


class Regression:

    def fit(self,xy,max_depth=None):

        if len(xy)==0:
            return Tree()
        if max_depth==0:

            return Tree(leaf=xy[:,-1].mean())
        init=self.mse(xy)
        diff=0
        mid_col=None
        mid_val=None
        mid_l=None
        mid_r=None
        n=xy.shape[1]-1

        for col in range(n):
            for val in xy[:,col]:
                l_data,r_data=self.split(xy,col,val)
                l_mse=self.mse(l_data)
                r_mse=self.mse(r_data)

                ds=init-l_mse-r_mse

                if ds>diff and len(l_data)>0 and len(r_data)>0:

                    mid_col=col
                    mid_val=val
                    mid_l=l_data
                    mid_r=r_data
                    diff=ds
        if diff>0:
            if max_depth:
                l=self.fit(mid_l,max_depth-1)
                r=self.fit(mid_r,max_depth-1)
            else:
                l=self.fit(mid_l)
                r=self.fit(mid_r)
            return Tree(col=mid_col,val=mid_val,l=l,r=r)
        else:

            return Tree(leaf=xy[:,-1].mean())
        
    def predict(self,x,tree):

        if tree.leaf!=None:
            return tree.leaf
        
        else:
            if x[tree.col]<tree.val:
                branch=tree.l
            else:
                branch=tree.r
            return self.predict(x,branch)
    
    def prinf(self,tree,level='ROOT-'):

        if tree.leaf!=None:
            print(level+str(tree.leaf))
        else:
            print(level+'*'+str(tree.col)+'*'+str(tree.val))
            self.prinf(tree.l,level+'L-')
            self.prinf(tree.r,level+'R-')


    @staticmethod
    def mse(xy):
        if len(xy)==0:
            return 0
        y=xy[:,-1]
        return np.power(y-y.mean(),2).sum()
    
    @staticmethod
    def split(xy,col,val):
        con=xy[:,col]<val

        l_data=xy[con]
        r_data=xy[~con]

        return l_data,r_data
    
if __name__ == '__main__':
    
    data,label=make_regression(1000,5)
    x_train,x_test,y_train,y_test=train_test_split(data,label,
                                                   train_size=0.8)
    
    max_depth=5
    
    xy_train=np.c_[x_train,y_train]
    deci_regress=Regression()
    tree=deci_regress.fit(xy_train,max_depth)

    # deci_regress.prinf(tree)
    p=np.apply_along_axis(deci_regress.predict,1,x_test,tree)
    
    print(F"自己写的函数结果: {r2_score(y_test,p)}")

    model=DecisionTreeRegressor(max_depth=max_depth)
    model.fit(x_train,y_train)

    yp=model.predict(x_test)
    print(f"sklearn 结果: {r2_score(y_test,yp)}")


    
