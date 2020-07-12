import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

path_train=r'data\housingprice\train.csv'
path_test= r'data\housingprice\test.csv'
Full_train=pd.read_csv(path_train)
Full_test = pd.read_csv(path_test)

# def checkData(data):
#     model=pd.DataFrame(data)
#     print(model.size)
#     print(model.shape)


# checkData(Full_train)
# checkData(Full_test)
Full_train.dropna(axis=0,)



print(Full_train.shape)
print(Full_train.columns)
print(Full_train['RoofMatl'])

