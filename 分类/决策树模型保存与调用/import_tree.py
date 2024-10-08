from build_tree import Tree # 必须导入这个类
import pickle


with open(r'/Users/yan/git-test/machine/分类/models/iris.pkl','rb') as r:
    tree=pickle.load(r)

print(tree)