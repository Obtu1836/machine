from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt

if __name__ == '__main__':

    com = load_iris()
    data = com.data
    label = com.target

    train_x, test_x, train_y, test_y = train_test_split(data, label,
                                                        train_size=0.7,
                                                        shuffle=True)
    tr=DecisionTreeClassifier(criterion='entropy')

    tr.fit(train_x,train_y)

    res=tr.predict(test_x)

    print((res==test_y).sum()/len(res))

    tree.plot_tree(tr,feature_names=com.feature_names)
    # plt.show()
    # plt.savefig('./分类/决策树/tree.png')