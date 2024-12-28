import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def creat_gauss_kernel(kernel_size=3,k=1, sigmas=[1,1],means=[0,0]):
    X = np.linspace(-k, k, kernel_size)
    Y = np.linspace(-k, k, kernel_size)
    x, y = np.meshgrid(X, Y)
    xy=np.stack((x,y),axis=2).reshape(-1,2)
    gauss=multivariate_normal(means,sigmas).pdf(xy).reshape(kernel_size,kernel_size)
    return x,y,gauss

x,y,gauss=creat_gauss_kernel(kernel_size=30,k=5,sigmas=[2,3],means=[0,1])
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.plot_surface(x,y,gauss,cmap='viridis')
plt.show()
