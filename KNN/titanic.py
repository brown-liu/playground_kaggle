import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

data=pd.read_csv(r'D:\projects\sklearn_kaggle\data\taitanic\train.csv')
#initial data shape
print(data.shape)
print(data.head())

cleanData= data.dropna(axis=0)
cleanedData=cleanData.iloc[:,5:6]

SalePrice=cleanData['Survived']
print('X====>',cleanedData.shape)
print('X====>',cleanedData.head())

print('Y====>',SalePrice.shape)
print('Y====>',SalePrice.head())

X_train,X_valid,y_train,y_valid=train_test_split(cleanedData,SalePrice,train_size=0.7,test_size=0.3,random_state=0)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

suvived=knn.predict(X_valid)

print(type(suvived))
print(type(y_valid))


