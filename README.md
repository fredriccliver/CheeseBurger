# CheeseBurger Machine Learning Library
28 Jun, 2018


- Regression 은 추후 구현.
- 우선은 Classifier 만 제작할 예정.
- 모든 feature를 continuous 하지 않고, cetegorical 한 데이터로 다룸.
- Class 3개 이상 가능.
- feature 마다 Entropy 를 계산하여 변별력 정도를 확인 후 prediction 에 적용.


## 연산순서
1. feature **scaling** (entropy를 구할때는 scaling 필요 없음.)
2. calculate **entropy** (entropy 값을 그대로 weight 로 사용가능.)
3. calculate **feature weight** (how importants features than others)
4. summary data point, stack to **the recipe** (the model)
5. call predict function with test data
6. get the suited point of every feature as test data's value from recipe (the making of **Burger Matrix**)
7. product **weight vector**(in 3.) **Burger Matrix**(in 6.)
8. find the best probable class.



## Descriptions for Directory and Files 

> **./exercise**
> 
> 개발과정 중에 만든 파이썬 문법 및 라이브러리 연습용 파일을 모아둡니다.


> **./Class/CheeseBurger.py**
> 
> 제작 중인 머신러닝 클래스.
> 우선은 Classifier 기능을 구현해서 Kaggle Titanic 에 써볼 수 있도록.


> **data/train-sep-for-...**
>
> kaggle 에 올리지 않고 테스트를 하기 위해 train data 를 학습용, 테스트용으로 분리함.

> **data/train-sep-for-learn.csv**
> 
> 학습용 749개 row



Categorical 한 data 는 scailing 불가, leveling 도 불가.
numerical 한 featrue 는 1~10, 10~20 등으로 묶을 수 있지만, 
categorical 한 data 는 불가능. 
하지만, clustering 으로는 categorical 한 data를 묶을 수 있지 않겠나?

[]clustering 부분 개발.


> **data/train-sep-for-test.csv**
> 
> 테스트용 142개 row
    

- 모델 예시

```python
# recipe(call model, officially) example
# structure of recipe is matrix of arrays.
recipe = [
    [ # feature1
        # [ 'class1 count', 'class2 count', 'class3 count']
        [ 4 , 0 , 0 ], # feature1 = val1
        [ 2 , 4 , 0 ], # feature1 = val2
        [ 0 , 2 , 3 ], # feature1 = val3
        [ 0 , 0 , 1 ]  # feature1 = val4
    ],
    [ # feature2
        [ 0 , 1 , 2 ], # feature2 = val1
        [ 5 , 1 , 1 ], # feature2 = val2
        [ 5 , 1 , 2 ], # feature2 = val3
        [ 3 , 2 , 2 ], # feature2 = val4
        [ 1 , 2 , 1 ]  # feature2 = val5
    ],
    [ #feature3
        [ 0 , 1 , 5 ], 
        [ 7 , 3 , 2 ]
    ]
]


# fwVector will choosed after calculating Deviations.
# fwVector = [ weight for feature 1, wight for feature 2 , ... ]
# have to scailing before calculate fwVector
fwVector = [ 5 , 3 , 2 ]

# test row has given like below
test = [ 1 , 3 , 0 ]

# predict(1 of 2)
# make a 'burger matrix'
# pull class point vectors by feature value of test.
# burget matrix is always m by m Square Matrix. 
# (m = counts of class)
burgerMatrix = [
    [2,4,0],    # recipe[0][1]
    [3,2,2],    # recipe[1][3]
    [0,1,5]     # recipe[2][0]
]

# predict(2 of 2)
# pointArray : fwVector X burgerMatrix (행렬곱)
pointArray = numpy.matmul(fwVector,burgerMatrix)
# result = [19, 28, 16] = [ point of Class1 , point of Class2 , point of Class3 ]

probabiliityArray = [
    pointArray[0] / numpy.sum(pointArray),      # 19/63 = 0.30
    pointArray[1] / numpy.sum(pointArray),      # 28/63 = 0.44
    pointArray[2] / numpy.sum(pointArray),      # 16/63 = 0.25
]

# Probability of Class2 is 44%.
```

