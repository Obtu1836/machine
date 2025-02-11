from sklearn.datasets import load_digits
import warnings
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from numpy.linalg import norm
import numpy as np
np.set_printoptions(precision=4, suppress=True)

'''
用高斯混合模型进行分类任务 :
     使用k个高斯分布拟合 (k是自定义的) 每一个类别的数据分布  假设有m个类别
     最后得到m个分类器。在每个分类器 最终输出的是当前的分类器对当前的类别数据的
     k个高斯分布的加权和 (rij*ws).sum(axis=1) 

     （当一个新的测试样本分别送入到各个类别的分类器时  如果属于该类别 那么概率值
     会是最大的）

     汇总每个分类器的最终结果 选出最大的索引 
'''

warnings.filterwarnings('ignore', category=RuntimeWarning)

class GMM_class:

    def __init__(self, k, d, data, alpha=1e-8):

        self.k = k
        self.d = d
        self.data = data
        self.alpha = np.eye(self.d)*alpha

    @staticmethod
    def k_plus(data, k):  # 初始化均值 轮盘法选取距离相对分散的
        cents = [data[np.random.randint(len(data))]]
        while len(cents) < k:
            dis = np.min(norm(data[:, None]-cents, axis=2), axis=1)
            dis = np.power(dis, 2)

            prob = dis/dis.sum()
            cum = np.cumsum(prob)
            sign = np.random.rand()
            for j, p in enumerate(cum):
                if p > sign:
                    cents.append(data[j])
                    break
        return np.array(cents)

    def init(self):  # 初始化 均值方差 权重

        mean = self.k_plus(self.data, self.k)

        cov = [np.eye(self.d)*np.random.rand() for i in range(self.k)]
        cov = np.array(cov)

        ws = np.random.rand(self.k)
        ws = ws/ws.sum()

        return mean, cov, ws

    def e_step(self, mean, cov, ws):  # e步 根据mean,cov,ws 确定z

        rj = np.zeros((len(self.data), self.k))
        for i in range(self.k):
            ns = multivariate_normal(mean[i], cov[i]+self.alpha)
            rj[:, i] = ns.pdf(self.data)  # 计算概率

        rj = rj*ws  # 加权
        rj = rj/rj.sum(axis=1)[:, None]

        return rj

    def m_step(self, z):  # (m,k) m步 根据 因变量 更新 mean，cov,ws
        
        z_p = z.sum(axis=0)  # (1,k)
        # m=np.tensordot(z,self.data,axes=[0,0])
        m = np.einsum('ik,in->kn', z, self.data)
        newm = m/z_p[:, None]

        sigs = np.zeros((self.k, self.d, self.d))
        for i in range(self.k):
            dx = self.data-newm[i]  # (m,d)
            sigs[i] = np.dot(dx.T*z[:, i], dx)/z_p[i]

        neww = z_p/len(self.data)
        return newm, sigs, neww

    def fit(self, iters):

        mean, cov, ws = self.init()
        for i in range(iters):
            z = self.e_step(mean, cov, ws)
            newm, newc, neww = self.m_step(z)
            mean, cov, ws = newm, newc, neww
        self.means, self.cov, self.ws = mean, cov, ws

    def predict(self, x):
        rj = np.zeros((len(x), self.k))
        for i in range(self.k):
            ns = multivariate_normal(self.means[i], self.cov[i]+self.alpha)
            rj[:, i] = ns.pdf(x)

        rj = rj.sum(axis=1)
        return rj  # 分类器中k个高斯分布的加权和


if __name__ == '__main__':

    com = load_digits()
    x, y = com.data, com.target

    scale = MinMaxScaler()
    pc = PCA(n_components=20)
    x = scale.fit_transform(x)
    x = pc.fit_transform(x)
    k = len(np.unique(y))
    d = x.shape[1]

    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.7,
                                                        stratify=y,
                                                        shuffle=True)

    dits = {}
    iter = 2000
    ks = 2
    for i in np.unique(y):  # 根据类别 选出数据 进行训练
        dat = train_x[np.where(train_y == i)[0]]
        gmm = GMM_class(ks, d, dat)
        gmm.fit(iter)
        dits[i] = gmm

    res = np.zeros((len(test_x), k))  # 将测试样本分别送到各个分类器
    for var in dits:
        model = dits[var]
        res[:, var] = model.predict(test_x)

    yp = res.argmax(axis=1)
    print(accuracy_score(test_y, yp))
