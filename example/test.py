import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) +'/..'))

import lib.CheeseBurger as cb
import numpy as np
import pandas as pd

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

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

#model.meta_load("./data/meta.cbmeta")





predictions = []

for i in range(0, test.shape[0]):
    predictions.append(model.probability_to_class(model.predict_row(test.loc[i,features].values.tolist())))

print(predictions)

submission = pd.read_csv("./data/gender_submission.csv", index_col="PassengerId")
submission["Survived"] = predictions

submission.to_csv("./data/result_cheeseburger.csv")






#print(model.predict_row(['female', 1, 'C', 2, 0, 0], debug=True))





# {
#     'features': ['Sex', 'Pclass', 'Embarked', 'GenerationsBy10', 'Parch', 'SibSp'],
#     'class_names': [0, 1],
#     'feature_level_dict': {
#         'Sex': ['female', 'male'],
#         'Pclass': [1, 2, 3],
#         'Embarked': [0, 'C', 'Q', 'S'],
#         'GenerationsBy10': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
#         'Parch': [0, 1, 2, 3, 4, 5, 6],
#         'SibSp': [0, 1, 2, 3, 4, 5, 8]
#     },
#     'recipe': [
#         [
#             [81, 233],
#             [468, 109]
#         ],
#         [
#             [80, 136],
#             [97, 87],
#             [372, 119]
#         ],
#         [
#             [0, 2],
#             [75, 93],
#             [47, 30],
#             [427, 217]
#         ],
#         [
#             [149, 90],
#             [61, 41],
#             [143, 77],
#             [94, 73],
#             [55, 34],
#             [28, 20],
#             [13, 6],
#             [6, 0],
#             [0, 1]
#         ],
#         [
#             [445, 233],
#             [53, 65],
#             [40, 40],
#             [2, 3],
#             [4, 0],
#             [4, 1],
#             [1, 0]
#         ],
#         [
#             [398, 210],
#             [97, 112],
#             [15, 13],
#             [12, 4],
#             [15, 3],
#             [5, 0],
#             [7, 0]
#         ]
#     ],
#     'weight_matrix': [
#         [0.748454716474066, 0.565729626209774],
#         [0.5115875527951135, 0.34391778666940254],
#         [0.6309302225274634, 0.4842686638623851],
#         [0.1964857449046287, 0.19349543449266438],
#         [0.6717661852482242, 0.5140385075749805],
#         [0.5589928367855448, 0.48594439314660925]
#     ]
# }
