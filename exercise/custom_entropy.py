# make fiting entropy calculation algorithm

import numpy as np
import pandas as pd
import math
import numpy
import scipy.stats
import lib.CheeseBurger as cb


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

# print(
#     train.ix[:,['Sex_digit', 'Age', 'Generation','Generation_by20' 'Old']]
# )
#
# print(scipy.stats.entropy(train['Sex_digit']))
# print(scipy.stats.entropy(train['Pclass']))
# print(scipy.stats.entropy(train['Parch']))
# print(scipy.stats.entropy(train['Age']))
# print(scipy.stats.entropy(train['Generation']))
# print(scipy.stats.entropy(train['Generation_by20']))
# print(scipy.stats.entropy(train['Old']))
# print(scipy.stats.entropy(train['Fare']))
#
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


# nan 값이 있는 row 도 drop 하지 않고, 값이 없음 역시 또 하나의 값으로 다루면 좋을 듯.

'''
각 feature 마다 값별 분포를 뽑아보는 중.
'''
# grouped_sum = train.groupby('Survived').aggregate(np.sum)
# grouped_count = train.groupby('Survived').aggregate(np.count_nonzero)
#
#
# print(grouped_sum.describe)
# print(grouped_count.iloc[:10,:6].describe)
#
#
# <bound method NDFrame.describe of           PassengerId  Pclass      Age  ...   Generation  Generation_by20   Old
# Survived                                ...
# 0               24179      73  2481.00  ...        219.0             95.0  19.0
# 1               59153     145  4047.42  ...        347.0            141.0  17.0
#
# [2 rows x 10 columns]>
# <bound method NDFrame.describe of           PassengerId  Pclass  Name  Sex    Age  SibSp
# Survived
# 0                  60      60    60   60   60.0     19
# 1                 123     123   123  123  123.0     54>

# 값별로 group화 해서 값별 count 뽑는 법은 /exercise/dataframe_groupby.py 참조.

# group by 해서, 값별 count.
original = [1, 2, 1, 1, 0, 1]
# entropy = 1.33

# leveled data of above.
leveled = [3, 2, 1]
# entropy = 1.16

leveled_more = [4,2]
# entropy = 1.11


# 적합한 entropy 값을 뽑아내는 함수를 만들어야 한다.
# 아래 조건들을 만족해야 함.
# condition : 데이터 갯수와 데이터 값 자체의 높고 낮음은 영향을 주지 않아야 한다.
only1 = [1,1,1,1]
only10 = [10,10,10,10,10,10,10,10]
# entropy = 1.0

# below should have same entropy.
original = [1, 2, 1, 1, 0, 1]
double_original = [1, 2, 1, 1, 0, 1, 1, 2, 1, 1, 0, 1]
# entropy = 1.33

# below should have same entropy.
left_side = [10,10,10,1,1,1,1,1,1,1,1,1,1]
center_side = [1,1,1,1,1,10,10,10,1,1,1,1,1]
# entropy = 2.5187


def entropy(arr):
    ent = 0
    for val in arr:
        ent += (val/sum(arr)) ** 2

    return ent * len(arr)


print(entropy(original))
print(entropy(leveled))
print(entropy(leveled_more))


entropy(only1)
entropy(only10)

entropy(double_original)

entropy(left_side)
entropy(center_side)


print('----------------')


print(entropy([0,0,0,0,0,0,0,0,100,0,0,0,]))
# 12.0

print(entropy([0,0,0,0,0,0,0,0,1,0,0,0,]))
# 12.0

print(entropy([1,1,1,1,1,1,1,1,1,1,1,1001,1,1,1,1,1,1,1,1,]))
# 19.26

print(entropy([1,1,1,1,1,1,1,1,1,1,1,1001,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
# 48.70

# conclusion.
# this entropy value is perfect to apply as weight.

print('----------------')