import pandas as pd
import scipy.stats

train = pd.read_csv("./data/train.csv", index_col='PassengerId')
feature_arr = train.ix[:, "Pclass"]

print(train.head())
print(feature_arr.head())

prob_arr = feature_arr.assign(pro=feature_arr.data.map(feature_arr.data.value_counts(normalize=True)))
print(prob_arr)