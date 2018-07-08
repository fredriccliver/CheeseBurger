import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) +'/..'))

import lib.CheeseBurger as cb
import numpy as np
import pandas as pd

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

model = cb.Classifier()
features = ["Sex", "Pclass", "Embarked"]
label = "Survived"


# model.fit(train, features, label)
# model.meta_save("./data/meta.cbmeta")

model.meta_load("./data/meta.cbmeta")
print(model.meta)

predictions = []

for i in range(0, test.shape[0]):
    predictions.append(model.probability_to_class(model.predict_row(test.loc[i,features].values.tolist())))

print(predictions)

submission = pd.read_csv("./data/gender_submission.csv", index_col="PassengerId")
submission["Survived"] = predictions

submission.to_csv("./data/result_cheeseburger.csv")




