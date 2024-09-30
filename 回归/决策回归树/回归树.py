from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

def mse(xy):
    if len(xy)==0:
        return 0
    y=xy[:,-1]
    return (np.power(y-y.mean(),2).sum())/len(y)

def split(x,col,val):

    con=x[:,col]<val
    l_data=x[con]
    r_data=x[~con]

    return l_data,r_data

class Tree:

    def __init__(self,col=-1,val=None,leaf=None,
                 l=None,r=None):
        
        self.col=col
        self.val=val
        self.leaf=leaf
        self.l=l
        self.r=r

def build(x,i=None): #参数i 限制递归深度 每递归一次 执行i-1  当i=0时 退出

    if len(x)==0 :
        return Tree() # leaf=None
    
    if i==0:
        return Tree(leaf=x[:,-1].mean())
    
    init=mse(x)
    mid_l=None
    mid_r=None
    mid_col=None #并不能保证下边的两次循环能执行 所以 先定义好
    mid_val=None
    n=x.shape[1]-1
    diff=0

    for col in range(n):
        for val in x[:,col]:

            l_data,r_data=split(x,col,val)
            l_mse=mse(l_data)
            r_mse=mse(r_data)
            new=l_mse+r_mse

            ds=init-new

            if ds>diff and len(l_data)>0 and len(r_data)>0:
                diff=ds
                mid_l=l_data
                mid_r=r_data
                mid_col=col
                mid_val=val
        
    if diff>0 :
        if i:
            l=build(mid_l,i-1)
            r=build(mid_r,i-1)
        else:
            l=build(mid_l)
            r=build(mid_r)

        return Tree(col=mid_col,val=mid_val,l=l,r=r) #leaf=None
    else:
        tag=x[:,-1].mean()

        return Tree(leaf=tag)
    
def predict(x,tree):

    if tree.leaf!=None:
        return tree.leaf

    else:

        if x[tree.col]<=tree.val:

            branch=tree.l
        else:
            branch=tree.r
        
        return predict(x,branch)
    

def prinf(tree,level='Root-'):

    if tree.leaf!=None:
        print(level,tree.leaf)

    else:
        print(level,tree.col,tree.val)
        prinf(tree.l,level+'L-')
        prinf(tree.r,level+'R-')

if __name__ == '__main__':

    data,label=make_regression(50,1)
    x_train,x_test,y_train,y_test=train_test_split(data,label,
                                                   train_size=0.8)
    
    train_xy=np.c_[x_train,y_train]

    tree=build(train_xy,5)

    yp=[]
    for var in x_test:
        yp.append(predict(var,tree))

    r2=r2_score(y_test,yp)
    print(r2)
    

