import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def dimensionalityReduction(n:int,x:np.array):

    u,sigma,vt=svd(x)
    '''
    绿色注释的部分 可以省略掉   不省略掉 则可以近似还原 原矩阵 
    ori 是还原的矩阵
    '''
    # sigma_c=np.diag(sigma[:n])
    # u_c=u[:,:n]
    vt_c=vt[:n,:]
    # ori=u_c.dot(sigma_c).dot(vt_c)
    res=x.dot(vt_c.T)

    return res


if __name__ == '__main__':
    com=load_iris()
    x,y=com.data,com.target
    n=2
    res=dimensionalityReduction(n,x)
    if res.shape[1]==1:
        res=np.c_[res[:,0],[1]*len(res)]

    plt.scatter(res[:,0],res[:,1],c=y)
    plt.show()

    

