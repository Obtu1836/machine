import numpy as np
import sys


def calculate_information_entropy(data:np.array,mode:str):

    try:
        assert mode in ['info','gini'],'must'
    except AssertionError as w:
        print(w)
        sys.exit()

    label=data[:,-1]
    p=np.unique(label,return_counts=True)[1]
    p=p/p.sum()
    dic={'info':info,'gini':gini}
    fun=dic.get(mode)
    return fun(p)
    
        

def info(p):

    return (-p*np.log2(p)).sum()
    
def gini(p):
    
    return 1-np.power(p,2).sum()


def split(data,col,val):

    con=data[:,col]<val

    l_data=data[con]
    r_data=data[~con]

    return l_data,r_data


def information_gain(data,col,val,mode):

    l_data,r_data=split(data,col,val)

    ori_info=calculate_information_entropy(data,mode)

    sp_l=calculate_information_entropy(l_data,mode)*len(l_data)
    sp_r=calculate_information_entropy(r_data,mode)*len(r_data)

    return ori_info-(sp_l+sp_r)/len(data)



if __name__ == '__main__':
    
    data = np.array([[35, 176, 0, 20000,0],
                     [28, 178, 1, 10000,1],
                     [26, 172, 0, 25000,0],
                     [29, 173, 2, 20000,1],
                     [28, 174, 0, 15000,1]])

    res=information_gain(data,1,175,'info')    
    print(res)
    
