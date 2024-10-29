import numpy as np
np.set_printoptions(precision=4,suppress=True)
from numpy.linalg import norm
from scipy.stats import multivariate_normal
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)

class GMM_class:

    def __init__(self,k,d,data,alpha=1e-8):

        self.k=k
        self.d=d
        self.data=data
        self.alpha=np.eye(self.d)*alpha

    @staticmethod
    def k_plus(data,k):
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

    def init(self):

        mean=self.k_plus(self.data,self.k)

        cov=[np.eye(self.d)*np.random.rand() for i in range(self.k)]
        cov=np.array(cov)


        ws=np.random.rand(self.k)
        ws=ws/ws.sum()

        return mean,cov,ws
    
    def e_step(self,mean,cov,ws):

        rj=np.zeros((len(self.data),self.k))
        for i in range(self.k):
            ns=multivariate_normal(mean[i],cov[i]+self.alpha)
            rj[:,i]=ns.pdf(self.data)
        
        rj=rj*ws
        
        return rj
    
    def m_step(self,z): #(m,k)
        
        z=z/z.sum(axis=1)[:,None]

        z_p=z.sum(axis=0) #(1,k)

        m=np.tensordot(z,self.data,axes=[0,0])
        newm=m/z_p[:,None]

        sigs=np.zeros((self.k,self.d,self.d))
        for i in range(self.k):
            dx=self.data-newm[i] #(m,d)
            sigs[i]=np.dot(dx.T*z[:,i],dx)/z_p[i]
        
        neww=z_p/len(self.data)
    
        return newm,sigs,neww
    
    def fit(self,iters):

        mean,cov,ws=self.init()
        for i in range(iters):
            z=self.e_step(mean,cov,ws)
            newm,newc,neww=self.m_step(z)
            mean,cov,ws=newm,newc,neww
        
        return mean,cov,ws
    
    def predict(self,means,cov,ws):
        rj=np.zeros((len(self.data),self.k))
        for i in range(self.k):
            ns=multivariate_normal(means[i],cov[i]+self.alpha)
            rj[:,i]=ns.pdf(self.data)

        p=np.log(rj+1e-10)+np.log(ws)
        return p

if __name__ == '__main__':
    
    com=load_iris()
    
    x,y=com.data,com.target
    
    k=len(np.unique(y))
    d=x.shape[1]
    
    train_x,test_x,train_y,test_y=train_test_split(x,y,train_size=0.7,
                                                   stratify=y,
                                                   shuffle=True)
    
    dits={}
    iter=500
    ks=1
    for i in np.unique(y):
        dat=train_x[np.where(train_y==i)[0]]
        gmm=GMM_class(ks,d,dat)
        
        m,c,w=gmm.fit(iter)
        dits[i]={'mean':m,'cov':c,'ws':w}

    res=np.zeros((len(test_x),k))
    for var in dits:
        model=dits[var]
        gmt=GMM_class(ks,d,test_x)
        ms,cs,ws=model['mean'],model['cov'],model['ws']
        p=gmt.predict(ms,cs,ws)
        res[:,var]=p.sum(axis=1)

    yp=res.argmax(axis=1)

    print(accuracy_score(test_y,yp))


    
    
    

    
    