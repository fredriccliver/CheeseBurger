import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) +'/..'))

import lib.CheeseBurger as cb
import numpy as np
import pandas as pd

train = pd.read_csv("./data/leaf-classification/train.csv")
test = pd.read_csv("./data/leaf-classification/test.csv")

model = cb.Classifier()

# #PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

# 채우기
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# 새 feature 만들기
##

# 학습, 예측할 feature

features = train.columns.values[2:]

# 학습에 사용할 label
label = "species"


model.fit(train, features, label)
model.meta_save("./data/meta.cbmeta")

#model.meta_load("./data/meta.cbmeta")



# predictions = model.getPredictions(test, mode=1)



predictions = model.getPredictions(test, mode=0)
print(predictions)


print(model.accuracy(train, label))



submission = pd.read_csv("./data/leaf-classification/gender_submission.csv", index_col="PassengerId")
submission["Survived"] = predictions

submission.to_csv("./data/leaf-classification/result_cheeseburger.csv")






# print(model.predict_row(['female', 1, 'C', 2, 0], debug=True))


