import pandas as pd

train = pd.read_csv("./data/train.csv", index_col="PassengerId")
test = pd.read_csv("./data/test.csv", index_col="PassengerId")

# preprocessing

# Encode Sex
train.loc[train["Sex"] == "male", "Sex_encode"] = int(0)
train.loc[train["Sex"] == "female", "Sex_encode"] = 1
test.loc[test["Sex"] == "male", "Sex_encode"] = 0
test.loc[test["Sex"] == "female", "Sex_encode"] = int(1)

# Fill Fare
train["Fare_fillin"] = train["Fare"]
train.loc[ train["Fare"].isnull() , "Fare_fillin" ] = 0
test["Fare_fillin"] = test["Fare"]
test.loc[ test["Fare"].isnull() , "Fare_fillin" ] = 0

# Encode Embarked
# [C, S, Q]
train["Embarked_C"] = train["Embarked"] == "C"
train["Embarked_S"] = train["Embarked"] == "S"
train["Embarked_Q"] = train["Embarked"] == "Q"
test["Embarked_C"] = test["Embarked"] == "C"
test["Embarked_S"] = test["Embarked"] == "S"
test["Embarked_Q"] = test["Embarked"] == "Q"

#train
# feature_names = ["Pclass","Sex_encode","Fare_fillin","Embarked_C","Embarked_S","Embarked_Q",]
feature_names = ["Pclass","Sex_encode","Embarked_C","Embarked_S","Embarked_Q",]

X_train = train[feature_names]
y_train = train['Survived']
X_test = test[feature_names]

from sklearn.tree import ExtraTreeClassifier as dtc

model = dtc(max_depth=5)
predictions = model.fit(X_train, y_train).predict(X_test)

print (predictions)

submission = pd.read_csv("./data/gender_submission.csv", index_col="PassengerId")
submission["Survived"] = predictions

submission.to_csv("./data/result_decisiontree.csv")