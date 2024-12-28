import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv,norm
import matplotlib.pyplot as plt
import matplotlib.animation as mpan

def lrlw(tx, x, y, k):

    dis=norm(x[:,None]-tx,axis=2)
    dis=np.power(dis,2)
    dia=np.exp(-dis/(2*k**2))
    q = np.diag(dia.ravel())
    w = inv(x.T.dot(q).dot(x)).dot(x.T).dot(q).dot(y)

    return tx[:, None].dot(w)


def paint(n):

    k = ks[n]

    yp=np.apply_along_axis(lrlw,1,testx,x,y,k)

    p.set_data(testx.ravel(), yp)
    st.set_text(f'Gauss_ker={k}')

    return p


if __name__ == '__main__':
    '''
    局部加权回归 属于局部回归 这种方法 需要测试样本（一条）跟 训练样本(m条)
    确定权值矩阵 (为对角矩阵)  这也就意味着 这种方法不能生成模型
    因为不同的测试样本生成的权值矩阵是不同的 

    生成权值矩阵 利用了高斯核 通过调节K的大小 确定拟合程度
    '''

    # 生成训练用的数据
    trainx = np.linspace(-3, 3, 50)
    def f(x): return x**2+4*x+4

    trainy = f(trainx)+np.random.randn(len(trainx))*1

    x = trainx[:, None]
    y = trainy[:, None]

    # 生成测试数据 并画出图 用来查看拟合程度
    testx = np.linspace(-3, 3, 200).reshape(-1, 1)
    # 不同的k 生成不同的权值
    ks = [5, 2, 1, 0.5, 0.05]

    fig = plt.figure()
    plt.ylim(trainy.min()-5, trainy.max()+5)

    p = plt.plot([], [], color='g')[0]
    st = plt.text(-2, 2, '')

    plt.scatter(trainx.ravel(), trainy.ravel())
    ani = mpan.FuncAnimation(fig=fig, func=paint, frames=range(len(ks)),
                             interval=1000)
    plt.show()
