import numpy as np
from sklearn.datasets import load_iris,load_digits
from sklearn.model_selection import train_test_split

def cal_info(p):
    '''
    信息熵的计算方式
    熵越大 代表样本越均衡 所以划分数据集后 样本变得不均衡 熵也就变小
    划分数据集的标准也是 选用 信息增益最大的特征为标准 

    信息增益=初始熵-划分左右数据集以后 两部分的熵的均值  68 71行可见
    '''

    return (-p*np.log2(p)).sum()

def cal_gini(p):

    '''
    基尼系数 
    计算划分数据前和后的基尼系数 选取基尼系数降低最多的特征为划分依据
    见68 71行
    '''

    return 1-np.power(p,2).sum()

def cal_info_gini(data,mode):

    label=data[:,-1]
    lab,counts=np.unique(label,return_counts=1)
    p=counts/counts.sum()

    fun={'info':cal_info,'gini':cal_gini}

    return fun[mode](p)

def spilit(data,col,val):

    con=data[:,col]<val

    l_data=data[con]
    r_data=data[~con]

    return l_data,r_data

class Tree:
    def __init__(self,col=-1,val=None,leaf=None,
                 l=None,r=None,mode=None):
        
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
    init_entropy=cal_info_gini(data,mode)
    m,n=data.shape

    for col in range(n-1):
        for val in data[:,col]:

            l_data,r_data=spilit(data,col,val)
            ls=cal_info_gini(l_data,mode)*len(l_data)
            rs=cal_info_gini(r_data,mode)*len(r_data)

            new_entropy=(ls+rs)/len(data)

           
            ds=init_entropy-new_entropy

            if ds>diff and len(l_data)>0 and len(r_data)>0:
            
            # 无论采取信息熵或基尼系数的方法 都是选取 减少量最多的特征作为
            # 划分依据
                diff=ds
                mid_l=l_data
                mid_r=r_data
                mid_col=col
                mid_val=val

    if diff >0:
        l=build(mid_l,mode)
        r=build(mid_r,mode)

        return Tree(col=mid_col,val=mid_val,l=l,r=r,mode=mode)

    else:
        label=data[:,-1]
        lab,cuns=np.unique(label,return_counts=1)
        leaf=lab[np.argmax(cuns)]

        return Tree(leaf=leaf)
    
def predict(data,tree):

    if tree.leaf!=None:
        return tree.leaf

    else:
        if data[tree.col]<tree.val:
            branch=tree.l
        else:
            branch=tree.r

        return predict(data,branch)
    
def printf(tree,level='root-'):

    if tree.leaf!=None:
        print(level+"*"+str(tree.leaf))

    else:
        print(level+'^'+str(tree.col)+'^'+str(tree.val))
        printf(tree.l,level+'L-')
        printf(tree.r,level+'R-')
    
if __name__ == '__main__':
    
    # com=load_iris()
    com=load_digits()
    data=com.data
    label=com.target

    datas=np.c_[data,label]
    
    train,test=train_test_split(datas,train_size=0.7,shuffle=True)

    tree=build(train,'info')

    pred=[]
    for var in test:
        pred.append(predict(var,tree))
    pred=np.array(pred).astype(np.int32)

    acc=(pred==test[:,-1]).sum()/len(test)

    print(f'准确率: {acc}')

    # printf(tree)



    



