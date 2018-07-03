# make fiting entropy calculation algorithm

import numpy as np
import pandas as pd
import math
import numpy
import scipy.stats


train = pd.read_csv('../data/train.csv')

'''
train['Generation'] = math.floor(train['Age']) >> 안됨.
train['Generation'] = numpy.floor(train['Age']) >> 됨.
numpy 를 써야함.

Scipy, Sympy, Matplotlib, Pandas, Numpy 는 따로 훑어봐야할 듯.
'''

# Age 와 Generation 의 분포정도(편향도) 가 같다고 표현해낼 수 있는 entropy 계산식을 만들어야 함.



# entropy 계산.

train['Generation'] = numpy.floor(train['Age']/10)
train['Generation_by20'] = numpy.floor(train['Age']/20)
train['Old'] = numpy.floor(train['Age']/50)
train = train.dropna()

print(
    train.ix[:,['Age', 'Generation','Generation_by20' 'Old']]
)

print(scipy.stats.entropy(train['Age']))
print(scipy.stats.entropy(train['Generation']))
print(scipy.stats.entropy(train['Generation_by20']))
print(scipy.stats.entropy(train['Old']))
print(scipy.stats.entropy(train['Fare']))

'''
# print(scipy.stats.entropy(train['Sex']))

pk = 1.0*pk / np.sum(pk, axis=0)
TypeError: can't multiply sequence by non-int of type 'float'

> int 형이 아니면 entropy 계산시 오류가 남.
이 부분 오류가 안나면 좋을 듯.
'''



# nan 값이 있는 row 도 drop 하지 않고, 값이 없음 역시 값으로 다루면 좋을 듯.