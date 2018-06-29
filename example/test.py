import numpy as np
from CheeseBurger import Classifier 
#import CheeseBurger as cb

# dataRow 는 총 데이터 row 수.
# 각 row 는 행렬. (class count) By (feature count) matrix
# 아래 예제는 3 x 2 행렬. 분류되는 총 class 의 갯수가 3개 이며, 분석에 사용하는 feature 의 수는 2개.

# arr_0 = dataRow[0]
arr_0 = np.array(
    [
        [0.73, 0.55],
        [0.55, 0.32],
        [0.32, 0.77]
    ]
)

wVec = np.array(
    [1.33, 2]
)

# wApplied : 가중치가 적용된 class 별 feature 점수 배열
wApplied = arr_0 * wVec

classPointArr = []
for row in wApplied:
    # print(row) 
    # [0.9709 1.1   ]
    # [0.7315 0.64  ]
    # [0.4256 1.54  ]
    
    #print(sum(row))
    # 2.0709
    # 1.3715000000000002
    # 1.9656

    classPointArr.append(sum(row))

print(cb.Classifier.predictClass(classPointArr))

