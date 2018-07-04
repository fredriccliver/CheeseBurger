# Custom Machine Learing Class.
# 2018.06.27

import pandas

class Classifier:


    fwVector = [ 1,2,3 ]

    burgerMatrix = [
        [],[],[]
    ]

    recipe = [
        [],[],[]
    ]

    # x : row x feature matrix
    # t : row x label matrix
    def fit(self, x, y):
        return


    def predictClass(self, arr):
        predictedClass = np.argmax(arr)

        if predictedClass == 0:
            return "Class 1"
        elif predictedClass == 1:
            return "Class 2"
        else:
            return "Class 3"



    # data frame 을 받아서 모델을 생성.
    # 모델의 구체적인 형태는? data frame? matrix?
    def learn(self):
        return


    def calculateVariance(self):
        # 각 feature 별 Variance 를 array 로 return.
        # varVec[0.34, 0.33, 0.56]
        # element 객수는
        return



    # 행렬을 받아서 그 행렬에 모델을 적용하고, 정답 배열을 return.
    # return 되는 것도 행렬. [index, class] 로 return.
    def predict(self, disp={"class", "class-prob", "probabilities"}):

        pointArray = np.matmul(self.fwVector,self.burgerMatrix)
        # result = [ ,,, ] elements count : counts of classes.

        probabiliityArray = [
            pointArray[0] / np.sum(pointArray),
            pointArray[1] / np.sum(pointArray),
            pointArray[2] / np.sum(pointArray),
            ...
        ]

        # bestProb = [ className, Probability ]
        bestProb = [ 1,2,3]

        if disp=="class":               return bestProb[0]
        elif disp=="class-prob":        return bestProb
        elif disp=="probabilities":     return probabiliityArray
        else:                           return

import numpy as np



class Appetizer:

    featureDictionary = [
        [0, "First feature name"],
        [1, "Secont feature name"],
        [2, "Third feature name"]

    ]

    # Store min, max value for every features before scailing.
    # scalingMetaData = [
    #     [min, max]
    #     [min, max]
    # ]
    # scailing need no more.
    # scailing can adjust to countinuous feature only.

    # if want to know original value diversity of 'feature_A', code is below
    # scalingMetaDate[featureDictionary.getIdx("feature_A")]

    # 1.0 <= entropy
    # entropy = 1.0 : in flatten perfectly.
    # entropy is represent to how many
    def entropy(self, selected_col: pandas.DataFrame):
        # arr = arr.groupby(['Pclass']).size()
        # list(train[['Sex']].values.flatten())

        counts_list = selected_col.groupby(list(selected_col)[0]).size()
        arr = list(counts_list)

        ent = 0
        for val in arr:
            ent += (val / sum(arr)) ** 2
        return ent

    levelingDictionary = []

    def scaling(self):
        return

    def fill(self):
        return

    # def simpleReady(self):
    #     fill(self)
    #     scaling(self)
    #     leveling(self)
    #     return True

    # Leveling should apply to continuous values.
    # need developing other leveling algorithm for Discontinuous(Categorical) value.
    def leveling(self, data=[], level = 10):

        # check if need leveling.
        current_level = len(set(data))
        if(current_level <= level):
            return data


'''
Countinuous 한 featue는 
쉽게 leveling 이 가능하지만 Categorycal 한 data 는 불가능.

> Clustering 으로 leveling을 할 수 있을 듯.
'''
class Cluster:
    def initialize(self):
        return
