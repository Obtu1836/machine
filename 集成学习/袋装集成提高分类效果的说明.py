import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt


'''
假设 （以二分类为例）
    1  袋装的分类器 为同质的  总数为 n (奇数) 设为奇数主要是好看(对称)
    2  每个弱分类器的预测正样本的概率 统一为r (r>0.5)
    3  袋装分类器里  n//2+1(n的一半并取整) 的弱分类器的预测结果正 则判断为正
'''

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

def sony(p,n):
    r=0
    for k in range(n//2+1,n+1):
        r+=comb(n,k)*p**k*(1-p)**(n-k)
    return r


if __name__ == '__main__':
    
    p=np.linspace(0,1,50)

    for n in [5,13,37,81]:
        r=np.apply_along_axis(sony,0,p,n)
        plt.plot(p,r,label=f'{n}个弱分类器')
    plt.xlabel('单个分类器预测成功的概率')
    plt.ylabel('袋装分类器预测成功的概率')
    plt.legend()
    plt.show()
