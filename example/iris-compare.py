import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) +'/..'))

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 
from sklearn import datasets 
from sklearn import svm
import pandas as pd
import lib.CheeseBurger as cb
import numpy as np


iris = datasets.load_iris() 

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)  
data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['target'])
print(data1)
#Using SVM classifier 
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train) 
print(clf.score(X_test, y_test))


## SVM : 97.77%


''' cheese burger '''


features = []
features = data1.columns.values[0:4].tolist()
print(features)
model = cb.Classifier()
label = "target"
model.fit(data1, features, label)
model.meta_save("./data/iris_model.cbmeta") ## you can save model
predictions = model.getPredictions(data1, mode='CLASS')
print(predictions)
print(model.accuracy(data1, label))

## CB : 96.00%