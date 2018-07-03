import numpy as numpy


# recipe(call model, officially) example
# structure of recipe is array of matrix.
recipe = [
    [
        [ 4 , 0 , 0 ],
        [ 2 , 4 , 0 ],
        [ 0 , 2 , 3 ],
        [ 0 , 0 , 1 ]
    ],
    [
        [ 0 , 1 , 2 ],
        [ 5 , 1 , 1 ],
        [ 5 , 1 , 2 ],
        [ 3 , 2 , 2 ],
        [ 1 , 2 , 1 ]
    ],
    [
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
