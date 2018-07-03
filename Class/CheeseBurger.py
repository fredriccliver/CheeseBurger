# Custom Machine Learing Class.
# 2018.06.27



import numpy as np



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
    def fit(x, y):
        return
    
        
    def predictClass(self, arr):
        predictedClass = np.argmax(arr)

        if predictedClass == 0:
            return "Class 1"
        elif predictedClass == 1:
            return "Class 2" 
        else:
            return "Class 3"
    return
    
    
    # data frame 을 받아서 모델을 생성.
    # 모델의 구체적인 형태는? data frame? matrix?
    def learn(self):
        return


    def calculateVariance(self):
        # 각 feature 별 Variance 를 array 로 return.
        # varVec[0.34, 0.33, 0.56]
        # element 객수는 
        return

    # 생성해낸 모델.
    model = np.array(
            [[],[]]
        )

    # 행렬을 받아서 그 행렬에 모델을 적용하고, 정답 배열을 return.
    # return 되는 것도 행렬. [index, class] 로 return.
    def predict(disp={"class", "class-prob", "probabilities"}):

        pointArray = numpy.matmul(fwVector,burgerMatrix)
        # result = [ ,,, ] elements count : counts of classes. 
        
        probabiliityArray = [
            pointArray[0] / numpy.sum(pointArray),
            pointArray[1] / numpy.sum(pointArray),
            pointArray[2] / numpy.sum(pointArray),
            ...
        ]

        # bestProb = [ className, Probability ]
        bestProb = [ ,, ]

        if disp=="class":           return bestProb[0]
        elif disp=="class-prob":    return bestProb
        elif disp=="probabilities"  return probabiliityArray
        else                        return

        

class Appetizer:

    featureDictionary = [
        0: "First feature name",
        1: "Secont feature name",
        2: "Third feature name",
        ...
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


    levelingDictionary = []

    def scaling:
    
    def fill:

    def simpleReady:
        fill()
        scaling()
        leveling()
    
    # Leveling should apply to continuous values.
    # need developing other leveling algorithm for Discontinuous(Categorical) value.
    def leveling(data=[], level = 10):
        
        # check if need leveling.
        current_level = len(set(data))
        if(current_level <= level):
            return data

        

class Cluster:
    def init:
        return
