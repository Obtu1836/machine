import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
import matplotlib.animation as manp 

''''
多项式回归 一般适用于fx=ax+bx^2+cx^3......

也就是一元多次项方程  在多项式回归里 通常把 x^n作为一个新的特征 
通过poly这个函数 把x的多次项构造成新的特征 
则 新的函数 就为 fx=ax+bx1+cx2...  
 
'''
def paint(k):

    feat=PolynomialFeatures(k)
    
    x=feat.fit_transform(scax[:,None])

    model.fit(x,scay)

    yp=model.predict(feat.transform(basicx[:,None]))

    p.set_data(basicx,yp)
    st.set_text(f'k={k}')

    return [p]


if __name__ == '__main__':
    
    f=lambda x:10+5*x+4*x**2+6*x**3

    scax=np.linspace(-3,3,20)  
    scay=f(scax)+np.random.randn(len(scax))*30
    '''
    scax scay 生成训练用的样本和标签
    '''

    basicx=np.linspace(-3,3,100) 
    '''
    这个数据不用来训练 是用来测试的
    将 这个数据 带入训练好的模型生成 新的y值 然后画出图像 
    查看看这个图像是否过拟合
    
    '''
    fig=plt.figure()
    p=plt.plot([],[],'r')[0]
    st=plt.text(x=-2,y=150,s='')


    model=LinearRegression()


    plt.scatter(scax,scay,color='g')

    ani=manp.FuncAnimation(fig,paint,frames=range(1,20),
                           interval=1000,repeat=False)
    plt.show()

