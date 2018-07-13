import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) +'/..'))

import lib.CheeseBurger as cb
import numpy as np
import pandas as pd

# train = pd.read_csv("./data/titanic/splited_train_for_fit.csv")
# test = pd.read_csv("./data/titanic/splited_train_for_test.csv")

train = pd.read_csv("./data/titanic/train.csv")
test = pd.read_csv("./data/titanic/test.csv")

model = cb.Classifier()

# #PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

# 채우기
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# 새 feature 만들기
train['GenerationsBy10'] = np.floor(train['Age']/10)
test['GenerationsBy10'] = np.floor(test['Age']/10)
train['FamilySize'] = train['Parch'] + train['SibSp']
test['FamilySize'] = test['Parch'] + test['SibSp']
train['Fare_grade'] = np.floor(test['Fare']/100)
test['Fare_grade'] = np.floor(test['Fare']/100)

# 학습, 예측할 feature
features = train.columns.values[2:]
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



# 학습에 사용할 label
label = "Survived"


model.fit(train, features, label)
model.meta_save("./data/meta.cbmeta")

#model.meta_load("./data/meta.cbmeta")



# predictions = model.getPredictions(test, mode=1)



predictions = model.getPredictions(test, mode=0)
print(predictions)


print(model.accuracy(train, label))



submission = pd.read_csv("./data/titanic/gender_submission.csv", index_col="PassengerId")
submission["Survived"] = predictions

submission.to_csv("./data/titanic/result_cheeseburger.csv")






# print(model.predict_row(['female', 1, 'C', 2, 0], debug=True))


