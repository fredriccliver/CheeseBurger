import os, sys

# this row may be not necessary by your enviroments.
sys.path.append(os.path.abspath(os.path.dirname(__file__) +'/..'))

import lib.CheeseBurger as cb
import numpy as np
import pandas as pd

# train = pd.read_csv("./data/titanic/splited_train_for_fit.csv")
# test = pd.read_csv("./data/titanic/splited_train_for_test.csv")

# load data
train = pd.read_csv("./data/titanic/train.csv")
test = pd.read_csv("./data/titanic/test.csv")

# define instance for classificating prediction.
model = cb.Classifier()

# fill the null cell
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# make some new feature by combine existing features.
train['GenerationsBy10'] = np.floor(train['Age']/10)
test['GenerationsBy10'] = np.floor(test['Age']/10)
train['FamilySize'] = train['Parch'] + train['SibSp']
test['FamilySize'] = test['Parch'] + test['SibSp']
train['Fare_grade'] = np.floor(test['Fare']/100)
test['Fare_grade'] = np.floor(test['Fare']/100)

features = []

# define feature for make an model
## opt 1 : use all column
features = train.columns.values[2:].tolist()
## opt 2 : use some column
# features.append("Sex")
# features.append("Pclass")
# features.append("Embarked")
# #features.append("Fare_grade")
# #features.append("FamilySize")
# features.append("Parch")
# features.append("SibSp")
# features.append("Fare")
# features.append("Ticket")
# features.append("Cabin")

# Label column name.
label = "Survived"

# make a prediction model.
model.fit(train, features, label)
model.meta_save("./data/meta.cbmeta") ## you can save model

## or you can just load model, after saved.
# model.meta_load("./data/meta.cbmeta") ## load model.

predictions = model.getPredictions(test, mode='CLASS')
print(predictions)

# print the accuracy of prediction.
print(model.accuracy(train, label))
# the accuracy is 0.72952 (above 72%)

# it just making making for submiting to kaggle.
submission = pd.read_csv("./data/titanic/gender_submission.csv", index_col="PassengerId")
submission["Survived"] = predictions
submission.to_csv("./data/titanic/result_cheeseburger.csv")

## predict one row example.
# print(model.predict_row(['female', 1, 'C', 2, 0], debug=True))


