import pandas as pd

train = pd.read_csv('../data/train.csv')

# structure of recipe is matrix of arrays.
# recipe = [
#     [ # feature1
#         # [ 'class1 count', 'class2 count', 'class3 count']
#         [ 4 , 0 , 0 ], # feature1 = val1
#         [ 2 , 4 , 0 ], # feature1 = val2
#         [ 0 , 2 , 3 ], # feature1 = val3
#         [ 0 , 0 , 1 ]  # feature1 = val4
#     ],
#     [ # feature2
#         [ 0 , 1 , 2 ], # feature2 = val1
#         [ 5 , 1 , 1 ], # feature2 = val2
#         [ 5 , 1 , 2 ], # feature2 = val3
#         [ 3 , 2 , 2 ], # feature2 = val4
#         [ 1 , 2 , 1 ]  # feature2 = val5
#     ],
#     [ #feature3
#         [ 0 , 1 , 5 ],
#         [ 7 , 3 , 2 ]
#     ]
# ]


# ex) 승객정보만으로 승객의 목적지 확인
#
# 분류 가짓수 확인
#     classes = [europe, middle_east, us, japan]
#
# feature1 에서 value 값 별로 갯수 확인
#     feature1_groupby = [ economy:3 , business:5 , first:8 ]
#
# feature1 = feature1_groupby[0] & label = classes[0] 인 row 갯수 찾기. (0 by 0)
# feature1 = feature1_groupby[0] ...
#
#
# df.groupby([feature1, label]) 로 피벗테이블 형태의 list를 제작
#
# [
#     feature1 = val1, class=europe           : 30개   : 0x0 element
#     feature1 = val1, class=middle_east      : 50개   : 0x1 element
#     feature1 = val1, class=us               : 14개   : 0x2 element
#     feature1 = val1, class=japan            : 10개   : 0x3 element
#     feature1 = val2, class=europe           : 14개   : 1x0 element
#     feature1 = val2, class=middle_east      : 14개   : 1x1 element
#     feature1 = val2, class=us               : 14개   : ...
#     feature1 = val2, class=japan            : 14개
#     feature1 = val3, class=europe           : 14개
#     feature1 = val3, class=middle_east      : 14개
#     feature1 = val3, class=us               : 14개
#     ...
# ]
#
#
# featrue2 에서 value 값 종류 확인
#     feature2_groupby = [ 100dollar:120, 300dollar:50, 1000dollar:20]
#
# ... 마지막 feature 까지 반복


# ------------------------- 사전설명 끝. 아래는 구현. ----------------

# 분류 가짓수 확인
#     classes = [europe, middle_east, us, japan]
label_col_name = "Survived"
label = train[[label_col_name]]
#label.groupby([train.values])

groupby_label = label.groupby(list(label)[0]).size()
#print(dict(groupby_label)) # dataframe 을 object 형태로 바꿔줌.

classes = groupby_label.keys()
print("Classes : ")
print(classes)



# feature1 에서 value 값 별로 갯수 확인
#     feature1_groupby = [ economy:3 , business:5 , first:8 ]


def calFeaturePointMat(train_data: pd.DataFrame, feature_name : str):


    print(train_data[[feature_name]].groupby(
        list(train_data[[feature_name]])[0]
    ).size().head())
    return

features = ['Sex', 'Age']
for feature in features:
    calFeaturePointMat(train_data = train, feature_name = feature)
    pass



# feature1 = feature1_groupby[0] & label = classes[0] 인 row 갯수 찾기. (0 by 0)
# feature1 = feature1_groupby[0] ...
#
#
# df.groupby([feature1, label]) 로 피벗테이블 형태의 list를 제작
#
# [
#     feature1 = val1, class=europe           : 30개   : 0x0 element
#     feature1 = val1, class=middle_east      : 50개   : 0x1 element
#     feature1 = val1, class=us               : 14개   : 0x2 element
#     feature1 = val1, class=japan            : 10개   : 0x3 element
#     feature1 = val2, class=europe           : 14개   : 1x0 element
#     feature1 = val2, class=middle_east      : 14개   : 1x1 element
#     feature1 = val2, class=us               : 14개   : ...
#     feature1 = val2, class=japan            : 14개
#     feature1 = val3, class=europe           : 14개
#     feature1 = val3, class=middle_east      : 14개
#     feature1 = val3, class=us               : 14개
#     ...
# ]
#
#
# featrue2 에서 value 값 종류 확인
#     feature2_groupby = [ 100dollar:120, 300dollar:50, 1000dollar:20]
#
# ... 마지막 feature 까지 반복