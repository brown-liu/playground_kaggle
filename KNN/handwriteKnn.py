import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

digit=cv2.imread(r'D:\projects\sklearn_kaggle\data\handwrite\13.bmp')
#reduce the image from 3D to 2D
digit=cv2.cvtColor(digit,code=cv2.COLOR_BGR2GRAY)

print(digit.shape)
print(digit)
#cmap=plt.cm.gray change color
plt.imshow(digit,cmap=plt.cm.gray)
plt.show()
