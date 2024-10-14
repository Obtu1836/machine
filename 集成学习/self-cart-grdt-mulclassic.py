import numpy as np
from cart import Cart
from class_tree import Deci
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


class MultiCategorization:

    def __init__(self, max_depth=5, min_sample=1, tree_num=100, lr=0.1):

        self.max_depth = max_depth
        self.min_sample = min_sample
        self.tree_num = tree_num
        self.lr = lr

    def fit(self, x_train, y_train):

        self.tree = self.f(x_train, y_train, self.max_depth,
                           self.min_sample, self.tree_num)

    def f(self, x_train, y_train, max_depth=None, min_sample=None,
          iter_num=None):

        '''
        初始化先验概率 样本中的概率
        '''

        lab,num=np.unique(y_train,return_counts=1)
        f=np.tile([num],len(y_train)).reshape(len(y_train),-1)
        f=f/f.sum(axis=1,keepdims=1)

        # f=np.zeros_like(mask)  # 先验概率初始化为0

        self.k = len(lab) #类别数目
        mask = np.eye(self.k)[y_train] # one-hot格式
    
        n_trees = []
        flag = True
        for i in range(iter_num): #外循环 迭代次数
            k_models = []
            f=self.softmax(f)
            for cls in range(self.k): # 内循环 类别个数
                neg_grad = mask[:, cls]-f[:,cls] #每棵树里每个类别的负梯度
                model = Cart(max_depth, min_sample) #树模型实例
                model.fit(x_train, neg_grad) # 树训练
                pred = model.predict(x_train) # 树输出
                k_models.append(model)  
                if flag:  # 如果是
                    f[:,cls] = pred 
                else:
                    f[:,cls] += self.lr*pred
            flag = False
            n_trees.append(k_models)
        return n_trees

    def predict(self, x_test):

        lab = self.pred(x_test, self.tree)
        return lab

    def pred(self, x_test, trees):

        if np.array(x_test).ndim == 1: # 如果输入的维度是1 则扩展一下
            x_test = x_test[None, :]

        f = np.zeros(( len(x_test),self.k))#初始化0
        
        flag = True
        '''
        trees= [[cart(class1),cart(class2),cart(class3)],
               [cart(class1],cart(calss2),cart(class3)]]
        也就是说 共有m个小列表 每个小列表 包含k个类别的小树
        ''' 
        for tree in trees:
            for i, te in enumerate(tree):
                pred = te.predict(x_test)
                if flag:
                    f[:,i] = pred
                else:
                    f[:,i] += self.lr*pred
            flag = False

        res=self.softmax(f)
        lab = np.argmax(res, axis=1)
        return lab

    @staticmethod
    def softmax(x):
        p = np.exp(x)
        return np.divide(p, p.sum(axis=1,keepdims=1))
    

if __name__ == '__main__':

    com = load_iris()
    # com=load_breast_cancer()
    x, y = com.data, com.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,
                                                        shuffle=True,
                                                        stratify=y)

    max_depth, lr, simpale, tree_num = 3, 0.05, 2, 200
    GBDT = MultiCategorization(max_depth, simpale, tree_num, lr)
    GBDT.fit(x_train, y_train)
    res = GBDT.predict(x_test)
    acc1 = accuracy_score(y_test, res)
    print(f"self-acc: {acc1}")

    model = GradientBoostingClassifier(learning_rate=lr, n_estimators=tree_num,
                                       max_depth=max_depth)
    model.fit(x_train, y_train)
    m_res = model.predict(x_test)
    acc2 = accuracy_score(y_test, m_res)
    print(f"sklearn-acc: {acc2}")

    # cart = Cart(max_depth, simpale)
    deci=Deci(max_depth,min_samples=simpale)
    deci.fit(x_train, y_train)
    c_res = deci.predict(x_test)
    acc3 = accuracy_score(y_test, c_res)
    print(f"decitree-acc: {acc3}")
