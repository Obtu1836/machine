import numpy as np


def cal_info_gini(data, mode):

    label = data[:, -1]
    counts = np.unique(label, return_counts=1)[1]
    p = counts/counts.sum()

    if mode == 'info':
        return (-p*np.log2(p)).sum()
    else:
        return 1-np.power(p, 2)


def split(data, col, val):

    con = data[:, col] < val

    l_data = data[con]
    r_data = data[~con]

    return l_data, r_data


class Tree:
    def __init__(self, col=-1, val=None, sign=None,
                 l=None, r=None, mode=None):

        self.col = col
        self.val = val
        self.sign = sign
        self.l = l
        self.r = r
        self.mode = mode


def build(data, mode):

    if len(data) == 0:
        return Tree()

    basic = 0
    init_info_gini = cal_info_gini(data, mode)

    n = data.shape[1]-1

    for col in range(n):
        for val in data[:, col]:

            l_data, r_data = split(data, col, val)

            l_info_gini = cal_info_gini(l_data, mode)*len(l_data)
            r_info_gini = cal_info_gini(r_data, mode)*len(r_data)

            new_info_gini = (l_info_gini+r_info_gini)/len(data)

            ins = init_info_gini-new_info_gini

            if ins > basic and len(l_data) > 0 and len(r_data) > 0:

                basic = ins
                best_col = col
                best_val = val
                best_l = l_data
                best_r = r_data

    if basic > 0:
        l = build(best_l, mode)
        r = build(best_r, mode)

        return Tree(col=best_col, val=best_val, l=l, r=r, mode=mode)

    else:

        lab, cuns = np.unique(data[:, -1], return_counts=1)
        sign = lab[np.argmax(cuns)]
        return Tree(sign=sign, mode=mode)
    
def printf(tree,level='root-'):

    if tree.sign!=None:
        print(level+'-'+str(tree.sign))
    else:
        print(level+'-'+str(tree.col)+'-'+str(tree.val))
        printf(tree.l,level+'L')
        printf(tree.r,level+'R')


def predict(test, tree):

    if tree.sign != None:
        return tree.sign

    else:
        testval = test[tree.col]
        if testval >= tree.val:
            branch = tree.r
        else:
            branch = tree.l
        return predict(test, branch)


if __name__ == '__main__':

    data = np.array([[35, 176, 0, 20000, 0],
                    [28, 178, 1, 10000, 1],
                    [26, 172, 0, 25000, 0],
                    [29, 173, 2, 20000, 1],
                    [28, 174, 0, 15000, 1]])

    tree = build(data, 'info')

    test_sample = np.array([[24, 178, 2, 17000],
                            [27, 176, 0, 25000],
                            [27, 176, 0, 10000]])
    
    printf(tree,'root-')
    
    # res=[]
    # for var in test_sample:
    #     res.append(predict(var,tree))
    
    # print(res)
