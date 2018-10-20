## titanic prediction
## Cheese Burger vs Decision Tree

#%%
import os, sys
workpath = os.path.abspath(os.path.dirname('__file__'))+"/"
sys.path.append(workpath)
workpath

#%%
import lib.CheeseBurger as cb
import numpy as np
import pandas as pd

data = pd.read_csv("./data/titanic/train.csv")
data
# 891 rows × 12 columns

#%%
data = data.drop(columns=['Name', 'Ticket', 'Cabin'])
data
# 891 rows × 9 columns

#%%
data.dropna(inplace=True)
data
# 712 rows × 9 columns

#%%
data.loc[data['Sex']=='male', 'Sex_encoded'] = 0
data.loc[data['Sex']=='female', 'Sex_encoded'] = 1

data["Embarked_C"] = data["Embarked"] == "C"
data["Embarked_S"] = data["Embarked"] == "S"
data["Embarked_Q"] = data["Embarked"] == "Q"

data['Fare_grade'] = np.floor(data['Fare']/50)
data['Age_grade'] = np.floor(data['Age']/10)

data['FamilySize'] = data['Parch'] + data['SibSp']

data

#%%
features = ['Pclass', 'SibSp', 'Parch', 'Sex_encoded', 'Embarked_C', 'Embarked_S', 'Embarked_Q', 'Fare_grade', 'Age_grade']
label = 'Survived'

# loc 으로 가져오면 index 기준으로 가져와서 500개보다 적음.
train = data.iloc[:500,:] # 500개
test = data.iloc[501:,:] # 211개
print(train.shape, test.shape)
# train[features] # training X

#%%
import math
def validation(l1, l2):
    match_count = 0
    if(len(l1)!=len(l2)):
        print("two list is not same length")
        return -1
    for i in range(len(l1)):
        if(l1[i]==l2[i]):
            match_count = match_count+1
    return str(math.floor(match_count/len(l1) * 100)) +"%"


#%%
from sklearn.tree import DecisionTreeClassifier as dtc
model = dtc()
predictions = model.fit(X=train[features], y=train[label]).predict(X=test[features])
print(validation(predictions, list(test[label])))

#%%
model = cb.Classifier()
model.class_names
model.fit(train, features, label)
test[features]
predictions = model.getPredictions(test[features], mode='CLASS')
predictions
print(validation(predictions, list(test[label])))
