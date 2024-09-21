import numpy as np 
from collections import deque

f=lambda x:(x)**3-(x+3)**2-5*x

gradf=lambda x:3*x**2-2*(x+3)-5

dq=deque(maxlen=5)

x=-3
dq.append(x)

while True:
    x=x-f(x)/gradf(x)
    dq.append(x)
    ds=np.array(dq)
    if np.allclose(ds.mean(),ds):
        break

print(x)


