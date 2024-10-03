import numpy as np

def cal_info(p):

    return (-p*np.log2(p)).sum()

def cal_gini(p):

    return 1-np.power(p,2).sum()

def cal_info_gini(data,mode):

    label=data[:,-1]
    counts=np.unique(label,return_counts=1)[1]

    p=counts/counts.sum()

    method={'info':cal_info,'gini':cal_gini}

    return method.get(mode)(p)

def split(data,col,val):

    con=data[:,col]<val
    l_data=data[con]
    r_data=data[~con]

    return l_data,r_data

class Tree:
    def __init__(self,col=-1,val=None,leaf=None,
                      l=None,r=None,mode='info'):
        
        self.col=col
        self.val=val
        self.leaf=leaf
        self.l=l
        self.r=r
        self.mode=mode


def build(data,mode):

    if len(data)==0:
        return Tree()

    init_entropy=cal_info_gini(data,mode)
    pare=0

    m,n=data.shape

    for col in range(n-1):
        for val in data[:,col]:

            l_data,r_data=split(data,col,val)
            new_l=cal_info_gini(l_data,mode)*len(l_data)
            new_r=cal_info_gini(r_data,mode)*len(r_data)
            new_entropy=(new_l+new_r)/len(data)
            diff=init_entropy-new_entropy

            if diff>pare and len(l_data)>0 and len(r_data)>0:

                pare=diff
                mid_col=col
                mid_val=val
                mid_l=l_data
                mid_r=r_data
    
    if pare>0:

        l=build(mid_l,mode)
        r=build(mid_r,mode)

        return Tree(col=mid_col,val=mid_val,l=l,r=r,mode=mode)
    
    else:

        label=data[:,-1]

        lab,counts=np.unique(label,return_counts=1)

        sign=lab[counts.argmax()]

        return Tree(leaf=sign)
    
def predict(test,tree):

    if tree.leaf!=None:
        return tree.leaf
    
    else:
        
        if test[tree.col]<tree.val:
            branch=tree.l
        else:
            branch=tree.r

        return predict(test,branch)
    

def printf(tree,level='root-'):

    if tree.leaf!=None:
        print(level+str(tree.leaf))

    else:
        print(level+str(tree.col)+"-"+str(tree.val))
        printf(tree.l,level+'L-')
        printf(tree.r,level+'R-')
    

    

if __name__ == '__main__':
    
    data = np.array([[35, 176, 0, 20000, 0],
                    [28, 178, 1, 10000, 1],
                    [26, 172, 0, 25000, 0],
                    [29, 173, 2, 20000, 1],
                    [28, 174, 0, 15000, 1]])
    
    tree=build(data,'info')


    test_sample = np.array([[24, 178, 2, 17000],
                            [27, 176, 0, 25000],
                            [27, 176, 0, 10000]])
    
    pred=[]
    for var in test_sample:
        pred.append(predict(var,tree))

    print(pred)

    printf(tree)




    

    


    
