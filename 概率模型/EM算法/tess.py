import numpy as np
import pandas as pd

class em_algorithm:

    def __init__(self,x):

        self.x=x

        self.df=pd.DataFrame(self.x)
        self.count=self.df.apply(lambda x:x.value_counts(),axis=1)
        self.count.columns=['q','x']

    @staticmethod
    def cal_z(ser,pab):
        r=pab.map(lambda x:np.power(x,ser['x'])*np.power(1-x,ser['q']))
        return r

    def e_step(self,pab):

        res=self.count.apply(self.cal_z,args=(pab,),axis=1)
        res=res.divide(res.sum(axis=1),axis=0)
        return res
    
    @staticmethod
    def cal_pab(ser,count):

        r=ser.values.T*count
        return r.sum()

    def m_step(self,z):

        pab=z.T.groupby(level=0).apply(self.cal_pab,self.count)
        ab=pab
        v=ab.divide(ab.sum(axis=1),axis=0)
        return v.iloc[:,1]

if __name__ == '__main__':
    data = np.array([[0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                     [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                     [1, 0, 1, 1, 0, 0, 0, 0, 0, 1]])
    
    em=em_algorithm(data)

    pab=pd.Series([0.6,0.5],index=['a','b'])

    flag=True

    while flag:

        z=em.e_step(pab)
        pas=em.m_step(z)
        if np.allclose(pas,pab):
            break
        pab=pas

    print(pab)


