import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Tree:
    def __init__(self,col=-1,val=None,leaf=None,
                 l=None,r=None):
        
        self.col=col
        self.val=val
        self.leaf=leaf
        self.l=l
        self.r=r


class Deci:

    def __init__(self,max_dapth=None,mode='info',min_samples=5):

        self.max_depth=max_dapth
        self.mode=mode
        self.min_samples=min_samples
        
    def cal(self, y):

        _, counts = np.unique(y, return_counts=1)
        p = counts/counts.sum()

        if self.mode == 'info':
            return (-p*np.log2(p)).sum()
        else:
            return 1-np.power(p, 2).sum()


    @staticmethod
    def split(x, y, col, val):

        con = x[:, col] < val
        l_x, l_y = x[con], y[con]
        r_x, r_y = x[~con], y[~con]

        return l_x, l_y, r_x, r_y

    def fit(self, x, y):

        self._tree = self.build(x, y,self.max_depth,self.min_samples)

        return self._tree

    def build(self, x, y,max_depth=None,min_samples=0):
        
        if len(np.unique(y))==1:

            return Tree(leaf=y[0])

        if max_depth == 0 or len(x)<=min_samples:
            _, sign = np.unique(y, return_counts=1)
            leaf = _[np.argmax(sign)]
            return Tree(leaf=leaf)

        diff = 0
        init = self.cal(y)

        mid_col = None
        mid_val = None
        mid_lx = None
        mid_ly = None
        mid_rx = None
        mid_ry = None

        n = x.shape[1]

        for col in range(n):
            for val in np.unique(x[:, col]):

                l_x, l_y, r_x, r_y = self.split(x, y, col, val)
                l_ins = self.cal(l_y)*len(l_x)
                r_ins = self.cal(r_y)*len(r_x)

                new = (l_ins+r_ins)/len(x)

                ds = init-new

                if ds > diff and len(l_x) > 0 and len(r_x) > 0:

                    mid_col = col
                    mid_val = val
                    mid_lx = l_x
                    mid_rx = r_x
                    mid_ly = l_y
                    mid_ry = r_y
                    diff = ds

        if diff > 0:
            if max_depth:
                l=self.build(mid_lx,mid_ly,max_depth-1,min_samples)
                r=self.build(mid_rx,mid_ry,max_depth-1,min_samples)
            else:
                l = self.build(mid_lx, mid_ly,min_samples)
                r = self.build(mid_rx, mid_ry,min_samples)

            return Tree(col=mid_col, val=mid_val, l=l, r=r)

        else:
            _, clf = np.unique(y, return_counts=1)
            leaf = _[np.argmax(clf)]
            return Tree(leaf=leaf)

    def predict(self, test):

        yp = np.apply_along_axis(self.one_test, 1, test, self._tree)

        return yp

    def one_test(self, test, tree):

        if tree.leaf != None:
            return tree.leaf

        else:
            if test[tree.col] < tree.val:
                branch = tree.l
            else:
                branch = tree.r

            return  self.one_test(test, branch)
        
    def print_path(self,level='ROOL-'):
        self.pf(self._tree,level=level)


    def pf(self, tree, level=None):

        if tree.leaf != None:
            print(level+str(tree.leaf))

        else:
            print(level+str(tree.col)+"-"+str(tree.val))
            self.pf(tree.l, level+'L-')
            self.pf(tree.r, level+'R-')


if __name__ == '__main__':

    com = load_iris()
    x, y = com.data, com.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        shuffle=True,
                                                        stratify=y,
                                                        train_size=0.7)

    model = Deci(3,'gini')

    model.fit(x_train,y_train)

    yp=model.predict(x_test)

    model.print_path()

    print(accuracy_score(y_test,yp))
