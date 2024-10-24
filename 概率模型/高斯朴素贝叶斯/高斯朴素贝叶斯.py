import numpy as np
import pandas as pd
import scipy.stats as ss

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)

from sklearn.datasets import load_iris,load_wine,load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class GaussBayes:

    def fit(self,x,y):

        y=pd.Series(y)
        self.yps=np.log(y.value_counts(normalize=1))# 先验概率

        n=x.shape[1]
        self.col_name=[('x'+str(i+1)) for i in range(n)] # 定义特征名
        df=pd.DataFrame(x,columns=self.col_name)
        '''
        按类别分组并计算各组的均值方差 
        上步完成以后 列上有两重索引 分别是 特征名 和 mean+var 为了方便 交换一下
        合并mean var  最终的列索引  mean          var
                                  x1 x2 x3..... x1 x2 x3...
        '''
        mean_var=df.groupby(y).agg(['mean','var']) 
        mean_var=mean_var.swaplevel(axis=1)
        mean_var.sort_index(level=0,axis=1,inplace=True)

        self.group=mean_var.groupby(level=0)

    def two(self,df,ser):
        '''
        计算 每个测试样本在每个分组里各个特征的概率
        '''

        id=df.name # 保存类别的编号 方便提取类别的先验概率

        '''
        分别提取各个类别的特征的均值方差 需要reindex 一下 
        (因为df里的特征名顺序和测试样本的特征顺序可能存在差别
        测试样本的特征名x1 x2 .....
        而df里的特征名 因为经过第30行的sort_index 可能与测试样本特征名顺序不一致
        比如 df 里 x1 x2 x12 x14这种的顺序会出现问题）
        最后确定出样本的高斯分布概率 p(x1|y) p(x2|y)... 最后和先验概率累加(已转对数)
        '''
        
        mean=df['mean'].reindex(columns=self.col_name).values
        var=df['var'].reindex(columns=self.col_name).values
        ps=np.log(ss.norm(mean,np.sqrt(var)).pdf(ser))

        return ps.sum()+self.yps[id]


    def one(self,ser):
        '''
        返回每条样本在各个类别的概率 取类别最大的
        '''
        yp=self.group.apply(self.two,ser)
        return yp.idxmax()

    def predict(self,x):

        '''
        计算每条样本 预测三步走系列 predict  one two  三个函数分别承接
        的是testx,ser,df 代表了 预测 （每条测试样本）在（每个类别下的每个特征的概率）
        '''

        return np.apply_along_axis(self.one,axis=1,arr=x)
        

if __name__ == '__main__':
    
    # com=load_iris()
    com=load_breast_cancer()
    x,y=com.data,com.target

    x_trian,x_test,y_train,y_test=train_test_split(x,y,
                                                   train_size=0.7,
                                                   stratify=y)
    
    gb=GaussBayes()
    gb.fit(x_trian,y_train)

    yp=gb.predict(x_test)
    
    acc=accuracy_score(y_test,yp)
    print(acc)







