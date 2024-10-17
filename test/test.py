import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.animation as mana

def lwlr(one_test,sample,k):

    m=len(sample)
    dis=norm(sample[:,None]-one_test,axis=2)
    dis=np.power(dis,2)
    dia=np.exp(-dis/(2*k**2))
    q=np.diag(dia.ravel())
    

if __name__ == '__main__':
    
    x=np.linspace(-10,10).reshape(-1,1)
    y=x**2-4*x+4+np.random.randn(len(x))*10


    tesx=np.linspace(-20,20).reshape(-1,1)


    lwlr(tesx[0],x,0.5)
