import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from warnings import filterwarnings

# 忽略np.exp(x)x非常大的 数值溢出报错
filterwarnings('ignore',category=RuntimeWarning) 


def sigmoid(x, w):

    f = -x.dot(w)
    
    return (1/(1+np.exp(f)))


def loss(x, w, y):# x(m,n) w(n,1) y(m,1）

    # 必须有sum 
    return -(y*x.dot(w)-np.log(1+np.exp(x.dot(w)))).sum()


def grad(x, w, y):

    return -x.T.dot(y-sigmoid(x, w))


if __name__ == '__main__':

    # 随机生成一定数量的点 f(x)>0为正类 标签设为1
    x=np.random.uniform(0,100,600).reshape(-1,2)
    f=lambda x:4*x[:,0]-5*x[:,1]+10
    y=np.where(f(x)>0,1,0)[:,None]

    ''' 构造x 使得形状为[[x0,y0,1]
                       [x1,y1,1]]'''
    xs=np.c_[x,[1]*len(x)]

    m,n=xs.shape
    '''
    构造w 使得形状为[[w1],
                   [w2],
                   [w3]]

    由上述 x w  可得  f(w)=x.dot(w)=0 为直线方程表达式 
    '''
    
    w=np.zeros((n,1))

    q=deque(maxlen=5)

    #梯度下降 迭代求w
    init_loss=loss(xs,w,y)
    q.append(init_loss)
    lr=0.0001
    while True:
       
        w-=lr*grad(xs,w,y)
        new=loss(xs,w,y)
        q.append(new)
        s=np.array(q)

        if np.allclose(s.mean(),new):
            break

    ws=w.ravel()
    px=np.linspace(0,100,100)
    #根据解出w 依据 px  求出yu (也就是y的值·)
    yu=-ws[0]/ws[1]*px-ws[2]/w[1]
    
    plt.scatter(x[:,0],x[:,1],c=y.ravel())
    plt.plot(px,yu.ravel())
    plt.show()
