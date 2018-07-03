# make fiting entropy calculation algorithm

import numpy as np
import pandas as pd
import math
import numpy
import scipy.stats


train = pd.read_csv('../data/train.csv')



# Age 와 Generation 의 분포정도(편향도) 가 같다고 표현해낼 수 있는 entropy 계산식을 만들어야 함.



train.loc[
    train['Sex'] == 'male',
    'Sex_digit'
] = 1

train.loc[
    train['Sex'] == 'female',
    'Sex_digit'
] = 0


# entropy 계산.

train['Generation'] = numpy.floor(train['Age']/10)
train['Generation_by20'] = numpy.floor(train['Age']/20)
train['Old'] = numpy.floor(train['Age']/50)
train = train.dropna()

print(
    train.ix[:,['Sex_digit', 'Age', 'Generation','Generation_by20' 'Old']]
)

print(scipy.stats.entropy(train['Sex_digit']))
print(scipy.stats.entropy(train['Pclass']))
print(scipy.stats.entropy(train['Parch']))
print(scipy.stats.entropy(train['Age']))
print(scipy.stats.entropy(train['Generation']))
print(scipy.stats.entropy(train['Generation_by20']))
print(scipy.stats.entropy(train['Old']))
print(scipy.stats.entropy(train['Fare']))
# 4.553876891600542     (Sex_digit)
# 5.137922658766879     (Pclass)
# 4.035678834169101     (Parch)
# 5.100398923741883     (Age)
# 5.052093558144925     (Generation, group by mod 10)
# 4.951890442507133     (Generation_by20, group by mod 20)
# 3.58351893845611      (Old, group by mod 50)
# 4.8541666107143575    (Fare)
#
# age 를 10단위, 20단위로 묶어도 entropy 는 동일함. > 제대로 동작.
# 하지만 feature 마다 값차이가 별로 없음. log 함수로 값을 뽑아서 그런지?
# 이대로는 못씀.



'''
# print(scipy.stats.entropy(train['Sex']))

pk = 1.0*pk / np.sum(pk, axis=0)
TypeError: can't multiply sequence by non-int of type 'float'

> int 형이 아니면 entropy 계산시 오류가 남.
이 부분 오류가 안나면 좋을 듯.
'''


# nan 값이 있는 row 도 drop 하지 않고, 값이 없음 역시 값으로 다루면 좋을 듯.



grouped = train.groupby('Pclass').aggregate(np.sum)
grouped = train.groupby('Pclass').aggregate(np.count_nonzero)

print(grouped.iloc[:10,:6].describe)

