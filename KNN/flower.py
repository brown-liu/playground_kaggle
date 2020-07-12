import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pylab as pyb



X,y=datasets.load_iris(return_X_y=True)
X=X[:,:2]
print(X.shape)
pyb.scatter(X[:,0],X[:,1],c=y)
pyb.title("IRIS FLOWER")
pyb.xlabel("Size_X")
pyb.ylabel("Size_Y")
pyb.show()

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
x1=np.linspace(4,8,100)
y1=np.linspace(2,4.5,80)
X1,Y1=np.meshgrid(x1,y1)
print(X1.shape,Y1.shape)
pyb.scatter(X1,Y1,c='gray')
X1=X1.reshape(-1,1)
Y1=Y1.reshape(-1,1)
print(X1.shape,Y1.shape)

X_test=np.concatenate([X1,Y1],axis=1)
print(X_test.shape)
y_ = knn.predict(X_test)
from matplotlib.colors import ListedColormap
lc=ListedColormap(['red','blue','green'])
pyb.scatter(X_test[:,0],X_test[:,1],c=y_,cmap=lc)

pyb.show()