import numpy as np 
from numpy.linalg import norm
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)

def init_mean(data,k):
    ''' 
    kmean++ 初始化均值
    '''
    cents=[data[np.random.randint(len(data))]]

    while len(cents)<k:
        dis=np.min(norm(data[:,None]-cents,axis=2),axis=1)
        dis=np.power(dis,2)

        prob=dis/dis.sum()
        cum=np.cumsum(prob)
        sign=np.random.rand()
        for j,p in enumerate(cum):
            if p>sign:
                cents.append(data[j])
                break
    return np.array(cents)

class GMM:

    def __init__(self,k:int,d:int,data:np.array):

        self.k=k
        self.d=d
        self.data=data

        self.alpha=np.eye(self.d)*1e-8
    
    def init_param(self):
        '''
        初始化参数  用来拟合数据的高斯分布的均值,方差,占比系数
        均值形状: (k,d) 表示 k个高斯分布的均值
        协方差形状（k,d,d) 表示k个(d,d)形状的矩阵
        占比系数 (1,k) 表示 每个高斯分布占的比例系数
        '''
        lenth=len(self.data)
        ind=np.random.randint(0,lenth,self.k)
        mean=self.data[ind,:]
        '''初始均值如果选用init_mean这个方法 效果好很多'''
        # mean=init_mean(self.data,self.k)

        cov=[np.eye(self.d)*np.random.rand(1) for i in range(self.k)]
        cov=np.array(cov)

        w=np.random.rand(self.k)
        ws=w/w.sum() # 归一化操作 用来限制生成ws.sum()=1

        return mean,cov,ws
    
    def e_step(self,m,c,w):
        '''
        E步 根据传入的均值m 协方差c 计算样本(data)在每个高斯分布的概率(后验概率)
                    w可以看成样本在每个高斯分布的先验概率 
        '''
        r=np.zeros((len(data),self.k))
        for i in range(self.k):
            norm=multivariate_normal(m[i],c[i]+self.alpha)
            r[:,i]=norm.pdf(self.data)
        z=r*w 
        
        return z
        
    def m_step(self,z):
        '''
        M步 根据传入的隐变量 估计出 每个高斯分布的 均值 协方差 和占比
        '''
        z=z/np.sum(z,axis=1,keepdims=1)

        r_t=z.sum(axis=0) # (1,k)

        ms=np.zeros((self.k,self.d)) # 更新均值
        for i in range(self.k):
            p=(self.data*z[:,i][:,None]).sum(axis=0)/r_t[i]
            ms[i]=p
        '''mk和ms这两种写法等价 上式容易理解'''
        # mk=np.tensordot(z,self.data,axes=[0,0])
        # mk=mk/r_t[:,None]
             
        ss=np.zeros((self.k,self.d,self.d)) #更新协方差
        for i in range(self.k):
            dx=self.data-ms[i] # (m,d)
            ss[i]=np.dot((dx*(z[:,i][:,None])).T,dx)/r_t[i]
        
        ws=r_t/len(self.data)  #更新占比

        return ms,ss,ws
    
    def fit(self,iters=1):
        it_mean,it_cov,it_ws=self.init_param()

        for i in range(iters): #迭代过程
            z=gmm.e_step(it_mean,it_cov,it_ws) 
            n_mean,n_cov,n_ws=gmm.m_step(z)
            it_mean,it_cov,it_ws=n_mean,n_cov,n_ws
        
        res=self.e_step(it_mean,it_cov,it_ws)

        yp=np.log(res).argmax(axis=1) # 直接选出拟合程度最高的高斯分布 

        return yp
        

if __name__ == '__main__':
    
    k,d=4,2
    data,lab=make_blobs(1000,n_features=d,centers=k,shuffle=True,cluster_std=0.5)

    gmm=GMM(k,d,data)
    yp=gmm.fit(1)

    model=GaussianMixture(n_components=k,covariance_type='diag',
                          init_params='random')
    model.fit(data)

    ys=model.predict(data)

    fig,(ax1,ax2,ax3)=plt.subplots(1,3)

    ax1.scatter(data[:,0],data[:,1],c=lab)
    ax1.set_title('ori')
    ax2.scatter(data[:,0],data[:,1],c=yp)
    ax2.set_title('self')

    ax3.scatter(data[:,0],data[:,1],c=ys)
    ax3.set_title('sklearn')
    plt.show()



