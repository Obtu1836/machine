import numpy as np

'''
设f(x)=x**3+np.exp(x)/2+5*x-6
计算f(x)=0的解

思路: 另g(x)=f(x)**2 或者|f(x)| 因绝对值函数难操作 故采取平方的形式
这样 f(x)=0时 g(x)取极小值
所以 上述问题 就转化为 取极值的问题了
'''
f=lambda x:x**3+np.exp(x)/2+5*x-6
# f=lambda x: (x-1)**2-10``
g=lambda x:f(x)**2

def g_grad(x):
    delta=0.000001
    return (g(x+delta)-g(x))/(delta)

x=0
lr=0.00001
y=f(x)
while True:
    x=x-lr*g_grad(x)
    new_y=f(x)
    if np.allclose(y,new_y):
        break
    y=new_y

print(f'当x={round(x,5)}时,g(x)取得极小值 即f(x)的解' )


