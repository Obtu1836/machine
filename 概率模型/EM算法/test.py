import numpy as np
import pandas as pd


class em_algorithm:

    def __init__(self, data):

        self.data = data
        self.df = pd.DataFrame(data)
        self.count = self.df.apply(lambda x: x.value_counts(), axis=1)
        self.count.columns =['q','x']

    @staticmethod
    def cal_z(ser,pab:pd.Series):

        r=pab.map(lambda x:np.power(x,ser['x'])*np.power(1-x,ser['q']))
        return (r.div(r.sum()))
        
    def e_step(self,pab:pd.Series):

        z=self.count.apply(self.cal_z,axis=1,args=(pab,))

        return z
    
    @staticmethod
    def f(ser,count):

        res=(ser.values.T)*count.values
        k=pd.DataFrame(res,columns=count.columns)

        return k.sum()['x']/k.sum().sum()
    
    def m_step(self,z):

        pab=z.T.groupby(level=0).apply(self.f,self.count)

        return pab

if __name__ == '__main__':

    data = np.array([[0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
                     [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                     [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                     [1, 0, 1, 1, 0, 0, 0, 0, 0, 1]])
    
    em=em_algorithm(data)

    # print(em.count)

    pab=pd.Series([0.6,0.5],index=['a','b'])

    flag=True
    i=0
    while flag:

        z=em.e_step(pab)
        new_pab=em.m_step(z)
        if np.allclose(pab,new_pab):
            flag=False
            
        i+=1
        pab=new_pab

    print(new_pab,i)


