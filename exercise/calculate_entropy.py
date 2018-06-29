import pandas as pd
import numpy
import scipy.stats

train = pd.read_csv("./data/train.csv", index_col='PassengerId')
feature_arr = train.ix[:, "Pclass"]

train['FamilySize'] = train['SibSp'] + train['Parch']

# string 으로 읽고 싶다면 str() 말고 .astype() 을 사용해야 함.
#train['SexAndPclass'] = train['Sex'] + str(train['Pclass'])
train['SexAndPclass'] = train['Sex'] + train['Pclass'].astype("str")
train['GenerationsBy10'] = numpy.ceil(train['Age']/10)
train['GenerationsBy20'] = numpy.ceil(train['Age']/20)
train['GenerationsBy50'] = numpy.ceil(train['Age']/50)
#train['GenerationsBy150'] = numpy.ceil(train['Age']/150)

print(train.head())
#print(train.ix[0:10,"AgeClass"])
# print(feature_arr.head())

# percent_arr = pd.DataFrame(
#     {'Percentage': train.groupby(('Pclass')).size() / len(train)}
# )
# print(percent_arr)

# print(scipy.stats.entropy(percent_arr))


def calEntropy(feature_name):
    return scipy.stats.entropy(
        pd.DataFrame(
            {'Percentage': train.groupby(
                (feature_name)).size() / len(train)
            }
        )
    )[0]
# feature 의 leveling 에 따라 entropy 가 달라짐.
# 그렇다고해서 weight 에 level 갯수를 곱해주니, Name 과 같은 의미없는 feature 의 가중치가 너무 높아짐.
# Entropy :
#        Survived : 0.6659119735267652
#            Name : 6.792344427470811
#           SibSp : 0.9278185183972898
#           Parch : 0.7821038727533371
#           Cabin : 4.897561648814724
#          Pclass : 0.9976616191577423
#             Sex : 0.6489276088957644
#        Embarked : 0.7602918973432804
#             Age : 4.045611490075091
#            Fare : 4.881534445778674
#      FamilySize : 1.2638367056021338
#    SexAndPclass : 1.6370862937306083
# GenerationsBy10 : 1.752125743672301
# GenerationsBy20 : 1.095247737266915

def printEntropy(feature_name):
    print( '%15s'%feature_name + " : " + str(
        calEntropy(feature_name)
    ))
    return

def calculateFeatureWeight(feature_name):
    return (1/calEntropy(feature_name)) # * len(train.groupby(feature_name))

def printFeatureWeight(feature_name):
    print( '%15s'%feature_name + " : " + str(
        calculateFeatureWeight(feature_name)
    ))
    return

print("Entropy :")
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
printEntropy("Survived")
printEntropy("Name")
printEntropy("SibSp")
printEntropy("Parch")
printEntropy("Cabin")
printEntropy("Pclass")
printEntropy("Sex")
printEntropy("Embarked")
printEntropy("Age")
printEntropy("Fare")
printEntropy("FamilySize")
printEntropy("SexAndPclass")
printEntropy("GenerationsBy10")
printEntropy("GenerationsBy20")


print("----------")


print("Feature Weight :")
printFeatureWeight("Survived")
printFeatureWeight("Name")
printFeatureWeight("SibSp")
printFeatureWeight("Parch")
printFeatureWeight("Cabin")
printFeatureWeight("Pclass")
printFeatureWeight("Sex")
printFeatureWeight("Embarked")
printFeatureWeight("Age")
printFeatureWeight("Fare")
printFeatureWeight("FamilySize")
printFeatureWeight("SexAndPclass")
printFeatureWeight("GenerationsBy10")
printFeatureWeight("GenerationsBy20")
printFeatureWeight("GenerationsBy50")


# print(train.groupby("GenerationsBy10").size())
print(len(train.groupby("GenerationsBy10")))
print(len(train.groupby("GenerationsBy20")))
print(len(train.groupby("GenerationsBy50")))