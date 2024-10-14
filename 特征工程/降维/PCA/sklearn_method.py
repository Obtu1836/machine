from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def pca(n,x):

    model=PCA(n_components=n)
    res=model.fit_transform(x)

    return res


if __name__ == '__main__':
    
    com=load_iris()
    x,y=com.data,com.target

    res=pca(n=2,x=x)

    plt.scatter(res[:,0],res[:,1],c=y)
    plt.show()


