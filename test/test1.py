import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def cal_info_gini(data,info):

    label=data[:,-1]
    _,counts=np.unique(label,return_counts=1)
    p=counts/counts.sum()

    if info=='info':

        return (-p*np.log2(p)).sum()
    else:
        return 1-np.power(p,2)
    
def split(data,col,val):
    
    con=data[:,col]<val

    l_data=data[con]
    r_data=data[~con]
    
    return l_data,r_data

class Tree:
    def __init__(self,col=-1,val=None,leaf=None,l=None,
                 r=None,mode='info'):
        
        self.col=col
        self.val=val
        self.leaf=leaf
        self.l=l
        self.r=r
        self.mode=mode

def build(data,mode):

    if len(data)==0:
        return Tree()
    
    diff=0
    init=cal_info_gini(data,mode)
    n=data.shape[1]

    for col in range(n-1):
        for val in data[:,col]:
            l_data,r_data=split(data,col,val)
            l_ins=cal_info_gini(l_data,mode)*len(l_data)
            r_ins=cal_info_gini(r_data,mode)*len(r_data)
            new_ins=(l_ins+r_ins)/len(data)

            ds=init-new_ins

            if ds>diff and len(l_data)>0 and len(r_data)>0:

                diff=ds
                mid_l=l_data
                mid_r=r_data
                mid_col=col
                mid_val=val
    if diff>0:

        l=build(mid_l,mode)
        r=build(mid_r,mode)

        return Tree(col=mid_col,val=mid_val,l=l,r=r,mode=mode)
    
    else:

        lab=data[:,-1]
        ns,nm=np.unique(lab,return_counts=1)

        leaf=ns[np.argmax((nm))]

        return Tree(leaf=leaf)
    
def predict(data,tree):

    if tree.leaf!=None:
        return tree.leaf
    else:
        if data[tree.col]<tree.val:
            branch=tree.l
        else:
            branch=tree.r

    return  predict(data,branch)

def printf(tree,level='root-'):

    if tree.leaf!=None:
        print(level+str(tree.leaf)+"$")
    
    else:
        print(level+"*"+str(tree.col)+"*"+str(tree.val))
        printf(tree.l,level+'L-')
        printf(tree.r,level+'R-')


    
if __name__ == '__main__':
    
    com=load_iris()
    x,y=com.data,com.target

    xy=np.c_[x,y]

    xy=np.random.permutation(xy)

    train,test=train_test_split(xy,train_size=0.7)

    tree=build(train,'info')
    res=[]
    for var in test:
        res.append(predict(var,tree))

    print((res==test[:,-1]).sum()/len(res))

    printf(tree)