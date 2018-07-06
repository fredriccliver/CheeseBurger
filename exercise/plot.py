import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) +'/..'))


import lib.CheeseBurger as cb
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('data/train.csv')


train['Fare'].hist(bins=50)


train['Age'].hist(bins=50)
plt.show()

train['Parch'].hist(hins=10)

