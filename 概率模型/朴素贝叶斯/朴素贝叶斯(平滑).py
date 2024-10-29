import numpy as np
import pandas as pd


class NaiveBayes:

    def __init__(self, alpha=0):
        self.alpha = alpha

    @staticmethod
    def stat_feature(df): #统计df里每个特征有几个可选值
        p=df.apply(lambda x:x.value_counts())
        return p.T

    def fit(self, x, y):
        n=x.shape[1] # 统计列数
        col_name=[('x'+str(i+1)) for i in range(n)]# 生成列名
        df=pd.DataFrame(x,columns=col_name) # 转df格式
        '''
        根据类别数分组 并统计每个类别中 每个特征的的可选值有几个
        '''
        groups=df.groupby(y).apply(self.stat_feature) 

        groups.fillna(0,inplace=True)

        y=pd.Series(y)
        class_num=len(y.unique()) # 统计有几个类别
        '''
        计算先验概率 
        '''
        y_class_num=y.value_counts() # 计算先验概率(alpha为平滑系数)
        y_prob=(y_class_num+self.alpha)/\
                    (y_class_num.sum()+class_num*self.alpha)
        self.y_prob=np.log(y_prob)# 转对数

        '''
        feat 是统计总样本里 每个特征分别有几个可选值
        '''
        feat=df.apply(lambda x:len(x.unique()),axis=0)
        feat=pd.Series(np.tile(feat,class_num),index=groups.index)
        
        model=(groups+alpha).div(feat*alpha+groups.sum(axis=1),axis=0)
        ## 重新排序下 防止非自然排序 (以防万一)
        model=model.reindex(level=1,index=col_name)
        self.model=np.log(model)

        
    @staticmethod
    def dete_prob(df,x,yp):
        id=df.name
        x=x.values
        array=df.values
        x_p=array[range(len(df)),x].sum()+yp[id]# 选取对应的概率
        return x_p

    def class_part(self,x):
        res=self.model.groupby(level=0).apply\
                            (self.dete_prob,x,self.y_prob)
        print(res)
        return res

    '''
    对测试样本进行预测
    predict 执行每条测试样本
    new 每条测试样本在每个类别
    old 根据每条样本的 选出对应的类别下,每个特征值的概率(条件概率P(x|y))
    并与先验概率py相加(因为已经转对数了)
    最终 计算出每条样本在不同类别下的概率 选择概率最大的那个类别输出
    '''
    def predict(self, x):
        '''
        本步 也可以用np.apply_along_axis直接解决
        '''
        n=x.shape[1]
        df=pd.DataFrame(x,columns=[('x'+str(i+1))for i in range(n)])
        res=df.apply(self.class_part,axis=1).values
        return res.argmax(axis=1)


if __name__ == '__main__':

    train_x = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 2, 1],
                        [0, 0, 0]])
    alpha = 1
    train_y = np.array([0, 1, 0, 1, 1])

    nb=NaiveBayes(alpha)
    nb.fit(train_x,train_y)

    test_x=np.array([[1,2,1]])
    print(nb.predict(test_x))
