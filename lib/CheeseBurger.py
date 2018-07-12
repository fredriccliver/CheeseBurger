# Custom Machine Learing Class.
# 2018.06.27
import numpy as np
import pandas

class Classifier:
    features = []
    class_names = []
    feature_level_dict = {}
    recipe = []     # recipe[feature_index][level_index][class_index]
    weight_matrix = []
    train_count = []


    def meta_load(self, path:str):
        f = open(path, 'r')
        meta = eval(f.read())
        f.close()

        self.train_count = meta['train_count']
        self.features = meta['features'] 
        self.class_names = meta['class_names']
        self.feature_level_dict = meta['feature_level_dict']
        self.recipe = meta['recipe']
        self.weight_matrix = meta['weight_matrix']

        print("CHEESEBURGER : loaded meta file.")
        print(str(meta))
        return

    def meta_save(self, path:str):
        meta = {}
        
        meta['train_count'] = self.train_count
        meta['features'] = self.features
        meta['class_names'] = self.class_names
        meta['feature_level_dict'] = self.feature_level_dict
        meta['recipe'] = self.recipe
        meta['weight_matrix'] = self.weight_matrix
        
        f = open(path, 'w')
        data = str(meta)
        f.write(data)
        f.close()
        
        print("CHEESEBURGER : saved meta file.")
        print(data)
        return 
    

    def fit(self, data:pandas.DataFrame, features:list, label_col_name:str):
        self.features = features
        self.class_names = list(data[[label_col_name]].groupby(label_col_name).size().keys())
        self.train_count = data.shape[0]

        # feature level dict 추출.
        for feature in features:
            levels = list(data[[feature]].groupby(feature).size().keys())
            self.feature_level_dict[feature] = levels

        # point matrix 계산.
        for feature in features:

            by_level_arr = []
            for level in self.feature_level_dict[feature]:
                
                by_class_arr = []
                for class_name in self.class_names:
                    cnt = len(data[(data[feature] == level) & (data[label_col_name] == class_name)])
                    by_class_arr.append(cnt)
                
                by_level_arr.append(by_class_arr)

            self.recipe.append(by_level_arr)

        # weight matrix 계산
        for i in range(0, len(self.recipe)):
            arr = np.transpose(self.recipe[i])
            
            weight_arr = []
            for i in range(0, len(self.class_names)):
                
                weight_arr.append(Appetizer.cal_weight(self,arr[i]).tolist())
            
            #print("---")
            mean = np.mean(weight_arr) # class x feature 별로 따로 weight 를 메기니 classification 이 너무 한 쪽으로 치우쳐서 평균 값 하나만 쓰도록 함. 
            #print(np.mean(weight_arr))
            arr = []
            for i in range(0, len(self.class_names)):
                arr.append(mean)
            weight_arr = arr


            self.weight_matrix.append(weight_arr)
        return


    def predict_row(self, row: list, debug=False)->list:
        
        burger_matrix = []
        for i in range(0,len(row)):
            try:
                index = self.feature_level_dict[self.features[i]].index(row[i])
                
                
                
                burger_matrix.append(
                    #self.recipe[i][index] / sum(self.recipe[i][index])
                    (np.array(self.recipe[i][index]) / sum(self.recipe[i][index])).tolist()
                )
                pass
            except :
                burger_matrix.append([0]*len(self.class_names))
                pass
        
        if(debug == True): 
            print("burger_matrix :")
            print(burger_matrix)
        
        if(debug == True):
            print("weight_matrix :")
            print(self.weight_matrix)
        
        probability_arr = []
        for i in range(0, len(self.class_names)):
            
            probability_arr.append(
                np.matmul(
                    np.transpose(self.weight_matrix).tolist()[i],
                    np.transpose(burger_matrix)[i]
                    
                )
            )

        
        return probability_arr
    
    def probability_to_class(self, prob:list):
        return self.class_names[prob.index(max(prob))]

    def predictClass(self, arr):
        predictedClass = np.argmax(arr)

        if predictedClass == 0:
            return "Class 1"
        elif predictedClass == 1:
            return "Class 2"
        else:
            return "Class 3"



    # 추후에는 weight 를 feature, class 마다 따로 구해야 하지 않은지.
    # multi class classification 에서는 return 을 [ class1 weight, class2 weight, ...]
    # 클래스마다 중요한 feture 가 다를 수 있음.
    def get_feature_weight(self, train_data: list, label_col_name: str, feature_col_name: str):
        '''
        arguments
            train_data: list, 
            label_col_name: str, 
            feature_col_name: str
        '''
        class_names = list(train_data[[label_col_name]].groupby(label_col_name).size().keys())
        # levels = list(train_data[[feature_col_name]].groupby(feature_col_name).size().keys())
        
        divided_col_by_classes = []
        
        # for level in levels:
        #     divided_col_by_classes.append(
        #         list(train_data.loc[train_data[feature_col_name] == level, [label_col_name,feature_col_name]].groupby(label_col_name).size())
        #     )

        for selected_class in class_names:
            divided_col_by_classes.append(
                list(train_data.loc[train_data[label_col_name] == selected_class, [label_col_name,feature_col_name]].groupby(feature_col_name).size())
            )
        
        summary_entropy = 0

        # print(class_names)
        # print(levels)
        # print(divided_col_by_classes)
        for val_list_by_class in divided_col_by_classes:
            # print(val_list_by_class)
            # print(
            #     Appetizer.entropy_from_list(
            #         self,
            #         val_list_by_class
            #     )
            # )
            summary_entropy += Appetizer.entropy_from_list(
                self,
                val_list_by_class
            )
        #print([summary_entropy, len(levels)])
        return summary_entropy / len(class_names)

    def get_feature_weight_vector(self, train_data: list, label_col_name: str, feature_col_name: str):
        '''
        arguments
            train_data: list, 
            label_col_name: str, 
            feature_col_name: str
        '''
        class_names = list(train_data[[label_col_name]].groupby(label_col_name).size().keys())
        # levels = list(train_data[[feature_col_name]].groupby(feature_col_name).size().keys())
        
        divided_col_by_classes = []
        
        # for level in levels:
        #     divided_col_by_classes.append(
        #         list(train_data.loc[train_data[feature_col_name] == level, [label_col_name,feature_col_name]].groupby(label_col_name).size())
        #     )

        for selected_class in class_names:
            divided_col_by_classes.append(
                list(train_data.loc[train_data[label_col_name] == selected_class, [label_col_name,feature_col_name]].groupby(feature_col_name).size())
            )
        
        weight_vector = []

        # print(class_names)
        # print(levels)
        # print(divided_col_by_classes)
        for val_list_by_class in divided_col_by_classes:
            # print(val_list_by_class)
            # print(
            #     Appetizer.entropy_from_list(
            #         self,
            #         val_list_by_class
            #     )
            # )
            weight_vector.append(
                Appetizer.entropy_from_list(
                    self,
                    val_list_by_class
                )
            )
        #print([summary_entropy, len(levels)])
        return weight_vector #/ len(class_names)
    
    def get_weight(self, train_data: list, label_col_name: str, feature_col_name: str, return_type = "scalar"):
        '''
        return_type : scalar(or 0, default) | vector (or 1)
        '''
        if(return_type == "vector" or return_type == 1):
            return Classifier.get_feature_weight_vector(self, train_data, label_col_name, feature_col_name)
        else:
            return Classifier.get_feature_weight(self, train_data, label_col_name, feature_col_name)

    

    # 행렬을 받아서 그 행렬에 모델을 적용하고, 정답 배열을 return.
    # return 되는 것도 행렬. [index, class] 로 return.
    def predict(self, disp={"class", "class-prob", "probabilities"}):
        return



class Appetizer:

    featureDictionary = [
        [0, "First feature name"],
        [1, "Secont feature name"],
        [2, "Third feature name"]

    ]

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



    def entropy_from_list(self, data:list):
        
        ent = 0
        for val in data:
            ent += (val / sum(data)) ** 2
        
        return ent #* len(data) #클래스별 합산 엔트로피 구할때 문제가 생겨서 len 곱하는 부분 제거함.

    def cal_weight(self, data:list):
        w = 0
        for val in data:
            w += (val / sum(data)) ** 2
        return w

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
