import pandas as pd


train = pd.read_csv('../data/train.csv')


print(
    train.groupby(['Pclass', 'Sex']).size()
)
# Pclass  Sex
# 1       female    74
#         male      84
# 2       female     9
#         male       6
# 3       female     5
#         male       5
# dtype: int64

print(
    train.groupby(['Pclass']).size()
)
# Pclass
# 1    216
# 2    184
# 3    491
# dtype: int64

print(
    train.groupby(['Sex']).size()
)

print(
    train.groupby(['Age']).size()
)


