import pandas as pd

train = pd.read_csv('data/train.csv')

# structure of recipe is matrix of arrays.
# recipe = {
#     feature1:{ 
#         # [ 'class1 count', 'class2 count', 'class3 count']
#         [ 4 , 0 , 0 ], # feature1 = val1
#         [ 2 , 4 , 0 ], # feature1 = val2
#         [ 0 , 2 , 3 ], # feature1 = val3
#         [ 0 , 0 , 1 ]  # feature1 = val4
#     },
#     feature2:{ # feature2
#         [ 0 , 1 , 2 ], # feature2 = val1
#         [ 5 , 1 , 1 ], # feature2 = val2
#         [ 5 , 1 , 2 ], # feature2 = val3
#         [ 3 , 2 , 2 ], # feature2 = val4
#         [ 1 , 2 , 1 ]  # feature2 = val5
#     },
#     feature3:{ #feature3
#         [ 0 , 1 , 5 ],
#         [ 7 , 3 , 2 ]
#     }
# }


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
features = ['Sex', 'Age']
label_col = train[[label_col_name]]
#label.groupby([train.values])

groupby_label = label_col.groupby(list(label_col)[0]).size()
# Survived
# 0    549
# 1    342
# dtype: int64

# cf.
# print(dict(groupby_label)) 
# dataframe 을 dictionary 형태로 바꿔줌.
# {0: 549, 1: 342}

classes = list(groupby_label.keys())
# print(classes)
# [0, 1]

# feature1 에서 value 값 별로 갯수 확인
# feature1_groupby = [ economy:3 , business:5 , first:8 ]

def calFeaturePointMat(train_data: pd.DataFrame, feature_name : str):
    
    # groupby 할 때, feature 값이 NaN 인 경우, 그냥 사라지지 않도록 해야하나?
    print(dict(train_data.groupby([feature_name, label_col_name]).size()))
    # {('female', 0): 81, ('female', 1): 233, ('male', 0): 468, ('male', 1): 109}
    # {(0.42, 1): 1, (0.67, 1): 1, (0.75, 1): 2, (0.83, 1): 2, (0.92, 1): 1, (1.0, 0): 2, (1.0, 1): 5, (2.0, 0): 7, (2.0, 1): 3, (3.0, 0): 1, (3.0, 1): 5, (4.0, 0): 3, (4.0, 1): 7, (5.0, 1): 4, (6.0, 0): 1, (6.0, 1): 2, (7.0, 0): 2, (7.0, 1): 1, (8.0, 0): 2, (8.0, 1): 2, (9.0, 0): 6, (9.0, 1): 2, (10.0, 0): 2, (11.0, 0): 3, (11.0, 1): 1, (12.0, 1): 1, (13.0, 1): 2, (14.0, 0): 3, (14.0, 1): 3, (14.5, 0): 1, (15.0, 0): 1, (15.0, 1): 4, (16.0, 0): 11, (16.0, 1): 6, (17.0, 0): 7, (17.0, 1): 6, (18.0, 0): 17, (18.0, 1): 9, (19.0, 0): 16, (19.0, 1): 9, (20.0, 0): 12, (20.0, 1): 3, (20.5, 0): 1, (21.0, 0): 19, (21.0, 1): 5, (22.0, 0): 16, (22.0, 1): 11, (23.0, 0): 10, (23.0, 1): 5, (23.5, 0): 1, (24.0, 0): 15, (24.0, 1): 15, (24.5, 0): 1, (25.0, 0): 17, (25.0, 1): 6, (26.0, 0): 12, (26.0, 1): 6, (27.0, 0): 7, (27.0, 1): 11, (28.0, 0): 18, (28.0, 1): 7, (28.5, 0): 2, (29.0, 0): 12, (29.0, 1): 8, (30.0, 0): 15, (30.0, 1): 10, (30.5, 0): 2, (31.0, 0): 9, (31.0, 1): 8, (32.0, 0): 9, (32.0, 1): 9, (32.5, 0): 1, (32.5, 1): 1, (33.0, 0): 9, (33.0, 1): 6, (34.0, 0): 9, (34.0, 1): 6, (34.5, 0): 1, (35.0, 0): 7, (35.0, 1): 11, (36.0, 0): 11, (36.0, 1): 11, (36.5, 0): 1, (37.0, 0): 5, (37.0, 1): 1, (38.0, 0): 6, (38.0, 1): 5, (39.0, 0): 9, (39.0, 1): 5, (40.0, 0): 7, (40.0, 1): 6, (40.5, 0): 2, (41.0, 0): 4, (41.0, 1): 2, (42.0, 0): 7, (42.0, 1): 6, (43.0, 0): 4, (43.0, 1): 1, (44.0, 0): 6, (44.0, 1): 3, (45.0, 0): 7, (45.0, 1): 5, (45.5, 0): 2, (46.0, 0): 3, (47.0, 0): 8, (47.0, 1): 1, (48.0, 0): 3, (48.0, 1): 6, (49.0, 0): 2, (49.0, 1): 4, (50.0, 0): 5, (50.0, 1): 5, (51.0, 0): 5, (51.0, 1): 2, (52.0, 0): 3, (52.0, 1): 3, (53.0, 1): 1, (54.0, 0): 5, (54.0, 1): 3, (55.0, 0): 1, (55.0, 1): 1, (55.5, 0): 1, (56.0, 0): 2, (56.0, 1): 2, (57.0, 0): 2, (58.0, 0): 2, (58.0, 1): 3, (59.0, 0): 2, (60.0, 0): 2, (60.0, 1): 2, (61.0, 0): 3, (62.0, 0): 2, (62.0, 1): 2, (63.0, 1): 2, (64.0, 0): 2, (65.0, 0): 3, (66.0, 0): 1, (70.0, 0): 2, (70.5, 0): 1, (71.0, 0): 2, (74.0, 0): 1, (80.0, 1): 1}
    
    
    return

    print(
        train_data[[feature_name]].groupby(
            list(train_data[[feature_name]])[0]
        ).size().head()
    )
    return

calFeaturePointMat(train_data = train, feature_name = "Sex")

# for feature in features:
#     calFeaturePointMat(train_data = train, feature_name = feature)
#     pass



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