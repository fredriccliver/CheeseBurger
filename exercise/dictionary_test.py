# it calls to "Dictionary" like below object in python.
# it call "Associative array" or "Hash" also, in other language.
obj = {0: 549, 1: 342}

print(obj.get(0))
# 549

print(obj[0])
# 549

print(obj.fromkeys("01"))
# {'0': None, '1': None}

obj["attr"] = 10
obj['list'] = {0:1, 1:2}
print(obj)
# {0: 549, 1: 342, 'attr': 10, 'list': {0: 1, 1: 2}}

del obj['attr']
print(obj)
# {0: 549, 1: 342, 'list': {0: 1, 1: 2}}

print(obj.keys())
# dict_keys([0, 1, 'list'])

print(list(obj.keys()))
# [0, 1, 'list']

print(obj.get("name"))
# None

print(obj.get("name", "NO"))
# NO

print("list" in obj)
print("name" in obj)
# True
# False


recipe = {
    "feature1":{ 
        # [ 'class1 count', 'class2 count', 'class3 count']
        "val1":[ 4 , 0 , 0 ], # feature1 = val1
        "val2":[ 2 , 4 , 0 ], # feature1 = val2
        "val3":[ 0 , 2 , 3 ], # feature1 = val3
        "val4":[ 0 , 0 , 1 ]  # feature1 = val4
    },
    "feature2":{ # feature2
        "val1":[ 0 , 1 , 2 ], # feature2 = val1
        "val2":[ 5 , 1 , 1 ], # feature2 = val2
        "val3":[ 5 , 1 , 2 ], # feature2 = val3
        "val4":[ 3 , 2 , 2 ], # feature2 = val4
        "val5":[ 1 , 2 , 1 ]  # feature2 = val5
    },
    "feature3":{ #feature3
        "val1":[ 0 , 1 , 5 ],
        "val2":[ 7 , 3 , 2 ]
    }
}

print(recipe)

input = {"feature1":"val1", "feature2":"val3", "feature3":"val1"}

def getBurgerMat(test:dict):
    burgerMat = []
    feature_names = list(input.keys())
    for feature_name in feature_names:
        
        burgerMat.append(
            recipe.get(feature_name).get(input.get(feature_name))
        )
    return burgerMat

print("-----")
print("Burger Matrix : ")
print(getBurgerMat(input))
print("-----")
