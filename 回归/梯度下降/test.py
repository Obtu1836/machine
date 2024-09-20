import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    
    x,y=make_regression(400,3)

    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,
                                                   random_state=0)
    
    model=LinearRegression()
    model.fit(x_train,y_train)

    yp=model.predict(x_test).astype(np.int32)

    print(yp.ravel())
    print(y_test)

    # print(yp)
    # print(y_test)


