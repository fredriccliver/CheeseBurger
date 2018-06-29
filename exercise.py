import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection

import os


train = pd.read_csv("./train.csv")
#test = pd.read_csv("./test.csv")
#test_id = test.ID
print(train.shape , "Train shape")
#print(test.shape , "Test shape")

print(train.head())