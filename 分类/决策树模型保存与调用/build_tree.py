import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Tree:

    def __init__(self,col=None,val=None,leaf=None,l=None,r=None,
                mode=None):
        
        self.col=col
        self.val=val
        self.leaf=leaf
        self.l=l
        self.r=r
        self.mode=mode
        

class Build:

    @staticmethod
    def split(data,col,val):
        con=data[:,col]<val
        l_data=data[con]
        r_data=data[~con]
        return l_data,r_data
    
    @staticmethod
    def cal_info(data,mode):
        counts=np.unique(data[:,-1],return_counts=1)[1]
        p=counts/counts.sum()
        try:
            assert mode in ['info','gini']
        except AssertionError:
            print('模式不正确')
        if mode=='info':
            res=(-p*np.log2(p)).sum()
        else:
            res=1-np.power(p,2).sum()

        return res
    
    def build(self,data,mode,max_depth=None):

        if len(data)==0:
            return Tree()
        
        if max_depth==0:
            only,counts=np.unique(data[:,-1],return_counts=1)
            leaf=only[np.argmax(counts)]
            return Tree(leaf=leaf)
        
        init=self.cal_info(data,mode)
        diff=0

        mid_col=None
        mid_val=None
        mid_l=None
        mid_r=None
        
        n=data.shape[1]-1
        for col in range(n):
            for val in data[:,col]:

                l_data,r_data=self.split(data,col,val)
                ls=self.cal_info(l_data,mode)*len(l_data)
                rs=self.cal_info(r_data,mode)*len(r_data)

                new=(ls+rs)/len(data)

                ds=init-new
                if ds>diff and len(l_data)>0 and len(r_data)>0:

                    mid_col=col
                    mid_val=val
                    mid_l=l_data
                    mid_r=r_data
                    diff=ds
        if diff>0:
            if max_depth:
                l=self.build(mid_l,mode,max_depth-1)
                r=self.build(mid_r,mode,max_depth-1)
            else:
                l=self.build(mid_l,mode)
                r=self.build(mid_r,mode)
            
            return Tree(col=mid_col,val=mid_val,l=l,r=r,mode=mode)
        
        else:
            only,counts=np.unique(data[:,-1],return_counts=1)
            leaf=only[np.argmax(counts)]
            return Tree(leaf=leaf)
        
    def printf(self,tree,level='ROOT-'):

        if tree.leaf!=None:
            print(level+"*"+str(tree.leaf))
        else:
            print(level+"-"+str(tree.col)+'-'+str(tree.val))
            self.printf(tree.l,level+'L-')
            self.printf(tree.r,level+'R-')

    
    def predict(self,test,tree):
        if tree.leaf!=None:
            return tree.leaf
        else:
            if test[tree.col]<tree.val:
                branch=tree.l
            else:
                branch=tree.r
            return self.predict(test,branch)
    @staticmethod
    def save(tree,name):

        try:
            with open(r'分类/models/{}.pkl'.format(name),'wb') as w:
                pickle.dump(tree,w)
                print('写入完成')
        except FileNotFoundError:
            print('写入出错')
        

        
if __name__ == '__main__':
    com=load_iris()
    dat,label=com.data,com.target

    data=np.c_[dat,label]
    x_train,x_test=train_test_split(data,train_size=0.8,
                                    shuffle=True,
                                    stratify=label)

    model=Build()
    tree=model.build(x_train,'info',3)

    # model.printf(tree)
    x,y=np.split(x_test,[-1],axis=1)

    yp=np.apply_along_axis(model.predict,1,x,tree)

    acc=accuracy_score(y.ravel(),yp)
    print(acc)

    model.save(tree,'iris')







