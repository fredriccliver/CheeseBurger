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
    #
    # 나이분포가
    # 10대 : 10명, 20대 : 20명, 30대 : 30명 일때.
    # entropy = ( (10/60)^2 + (20/60)^2 + (30/60)^2 ) * 3
    #
    # entropy([0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, ])
    # # 12.0
    #
    # entropy([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, ])
    # # 12.0
    #
    # entropy([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1001, 1, 1, 1, 1, 1, 1, 1, 1, ])
    # # 19.26
    #
    # entropy(
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1001, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # # 48.70
    #
    # # 아래 조건을 만족해야 함.
    # # condition : 데이터 갯수와 데이터 값 자체의 높고 낮음은 영향을 주지 않아야 한다.
    # only1 = [1, 1, 1, 1]
    # only10 = [10, 10, 10, 10, 10, 10, 10, 10]
    # # entropy = 1.0
    #
    # # below should have same entropy.
    # original = [1, 2, 1, 1, 0, 1]
    # double_original = [1, 2, 1, 1, 0, 1, 1, 2, 1, 1, 0, 1]
    # # entropy = 1.33
    #
    # # below should have same entropy.
    # left_side = [10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # center_side = [1, 1, 1, 1, 1, 10, 10, 10, 1, 1, 1, 1, 1]
    # # entropy = 2.5187
    #
    '''
    argument
    selected_col : pandas.DataFrame
        input the 1 col dataframe.
        entropy(df[["column_A]])
    '''
    def entropy(self, selected_col: pandas.DataFrame):
        counts_list = selected_col.groupby(list(selected_col)[0]).size()
        arr = list(counts_list)

        ent = 0
        for val in arr:
            ent += (val / sum(arr)) ** 2
        return ent * len(arr)

    '''
    summary entropy for class, devided by each class
    whole feature 'a' entropy = entropy(feature 'a' rows, in label=Class_A) + entropy(feature 'a' rows, in label=Class_B) + ...
    '''
    def featureEntropy(self, selected_col: pandas.DataFrame):
        
        return


    # class_size = len(data_with_class[0])

    # def sumEntropyByClass(data:list):
    #     entropy_class_summary = 0
        
    #     for col in range(class_size):
            
    #         entropy_class_summary += cb.Appetizer.entropy_list(
    #             cb, 
    #             list(map(lambda x: x[col], data_with_class))
    #         )
    #     return entropy_class_summary    
        

    '''
    calculate entropy using the list type augument.
    eg) data = [2, 10, 10, 5, 2, 1]
        value is data point count
        element count is same with feature values level.
    '''
    # def entropy_from_list(self, data:list):
    #     ent = 0
    #     for val in data:
    #         ent += (val / sum(data)) ** 2
    #         print(ent)
    #     return ent * len(data)

    def entropy_from_list(self, data:list):
        
        ent = 0
        for val in data:
            ent += (val / sum(data)) ** 2
        
        return ent * len(data)

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
entropy 가 높아지는 방향으로 clustering 이 되도록 하면 될듯?
'''
class Cluster:
    def initialize(self):
        return
