import numpy as np
import sys

def info(p):
    # 信息熵计算
    return (-p*np.log2(p)).sum()

def gini(p):
    # 基尼系数计算
    return 1-np.power(p,2).sum()


def cal_info_gini(data,mode):

    d={'info':info,'gini':gini}
    try:
        assert mode in ['info','gini'],'mode ->"info"|"gini"'
    except AssertionError as e:
        print(e)
        sys.exit()
    
    label=data[:,-1]
    counts=np.unique(label,return_counts=True)[1]
    p=counts/counts.sum()

    fun=d.get(mode)

    return fun(p)

def split(data,col,val):

    con=data[:,col]<val
    l_data=data[con]
    r_data=data[~con]

    return l_data,r_data

def gain_info_gini(data,col,val,mode):

    # 计算 数据划分以后 信息增益或基尼系数减小量

    l_data,r_data=split(data,col,val)
    init_info_gini=cal_info_gini(data,mode)

    sp_info_gigi=(cal_info_gini(l_data,mode)*len(l_data)+\
                  cal_info_gini(r_data,mode)*len(r_data))/len(data)
    
    return init_info_gini-sp_info_gigi



if __name__ == '__main__':

    data = np.array([[35, 176, 0, 20000, 0],
                    [28, 178, 1, 10000, 1],
                    [26, 172, 0, 25000, 0],
                    [29, 173, 2, 20000, 1],
                    [28, 174, 0, 15000, 1]])
    
    
    
    
