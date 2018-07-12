import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) +'/..'))

import lib.CheeseBurger as cb
import numpy as np
import pandas as pd

train = pd.read_csv("./data/train-sep-for-learn.csv")
test = pd.read_csv("./data/train-sep-for-test.csv")

model = cb.Classifier()


# #PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked


train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

train['GenerationsBy10'] = np.floor(train['Age']/10)
test['GenerationsBy10'] = np.floor(test['Age']/10)

train['FamilySize'] = train['Parch'] + train['SibSp']
test['FamilySize'] = test['Parch'] + test['SibSp']

features = ["Sex", "Pclass", "Embarked", "Parch", "SibSp"]
label = "Survived"


model.fit(train, features, label)
model.meta_save("./data/meta.cbmeta")

# model.meta_load("./data/meta.cbmeta")





predictions = []

for i in range(0, test.shape[0]):
    predictions.append(model.probability_to_class(model.predict_row(test.loc[i,features].values.tolist())))

print(predictions)

submission = pd.read_csv("./data/gender_submission.csv", index_col="PassengerId")
submission["Survived"] = predictions

submission.to_csv("./data/result_cheeseburger.csv")






#print(model.predict_row(['female', 1, 'C', 2, 0, 0], debug=True))


