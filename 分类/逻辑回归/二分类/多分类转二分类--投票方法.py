from sklearn.datasets import load_iris,load_wine,load_digits
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from collections import deque
import warnings
import pandas as pd
from itertools import combinations
np.set_printoptions(precision=3,suppress=True)

warnings.filterwarnings('ignore',category=RuntimeWarning)

class Logic:

    '''执行逻辑回归操作(二分类)'''
    
    @staticmethod
    def sigmoid(x,w):

        return 1/(1+np.exp(-x.dot(w)))
    
    @staticmethod
    def loss(x,w,y):

        return -(y*x.dot(w)-np.log(1+np.exp(x.dot(w)))).sum()
    
    @staticmethod
    def grad(x,w,y):

        return x.T.dot(Logic.sigmoid(x,w)-y)
    

def predict(x,y):
    
    '''
    预测函数
    传入 x 形状(m,n) y的形状(m,1)
    返回权重
    '''

    m,n=x.shape
    w=np.zeros((n,1))

    logic=Logic()

    q=deque(maxlen=5)
    q.append(logic.loss(x,w,y))
    lr=0.001
    while True:
        w-=lr*logic.grad(x,w,y)
        new_loss=logic.loss(x,w,y)
        q.append(new_loss)
        s=np.array(q)
        if np.allclose(s.mean(),new_loss):
            break

    return w


def select(tup):
    
    '''
    多分类比如[0,1,2] 转成两两结合的二分类 因此需要把对应两个类别的训练数据
    提取出来
    (0,1) (0,2) (1,2)
    传入的参数 为元祖 的格式 
    '''

    a,b=tup # a<b
    '''
    提取对应标签的数据  xy是 x和y在列方向合并的数据 
    目的是 保证 下面打乱顺序时 特征和标签能对应
    '''
    con=(xy[:,-1]==a)|(xy[:,-1]==b) 
    
    data=np.random.permutation(xy[con]) # 打乱顺序
    '''
    无论传入的标签是什么 都将标签 重新定为(0,1) 
    '''
    data[:,-1]=np.where(data[:,-1]==a,0,data[:,-1])
    data[:,-1]=np.where(data[:,-1]==b,1,data[:,-1])

    l_x,l_y=np.split(data,[-1],axis=1) # 分割 x ,y

    w=predict(l_x,l_y)

    return [tup,w] #返回 原本的标签和 计算的权重
    

def normal():
    num=np.unique(y)
    iterc=list(combinations(num,2))# 两两组合类别
    ws=[]
    for var in iterc:
        ws.append(select(var))
    return ws

def f(x):
    a,b=np.unique(x,return_counts=1)
    return a[np.argmax(b)]

def test(x): 
    ws=normal() # 存储了真实标签和每个分类器的权重的组合[[（0，1），w1],
                                               #   [(0,2), w2],
                                               #   [(1,2), w2]]
    res=[]
    for i,var in enumerate(x): # 遍历每条测试的样本
        ls=[]
        '''
        将原本标签和样本 依次送入到每个分类器 返回的k则为样本在本分类器的概率
        如果概率>0.5 则是本分类器的正样本(1) 否则为负样本(0)
        同时 0,1 也为sign的索引 也就是返回了 原本的标签
        '''
        for sign,w in ws: 
            k=Logic.sigmoid(var[None,:],w).ravel()[0]
            k=1 if k>0.5 else 0
            label=sign[k]
            ls.append(label)
        res.append(ls) # 最终每条样本返回3个分类器给出的实际标签

    df=pd.DataFrame(res)  # 投票  选出每条 标签数量最多的标签作为预测标签
    ret=df.apply(f,axis=1).values
    print((ret==y).sum()/len(y))


if __name__ == '__main__':

    # com=load_breast_cancer()
    com=load_wine()
    # com=load_digits()
    scale=MinMaxScaler()
    x,y=com.data,com.target

    x=scale.fit_transform(x)

    xy=np.c_[x,y]

    np.array(test(x))

    



    
    
        

    


    

    


    
    


    

