# Custom Machine Learing Class.
# 2018.06.27
# Fredric Cliver

import numpy as np
import pandas

class Classifier:
    
    features = []
    class_names = []
    feature_level_dict = {}
    recipe = []
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
        self.weight_matrix = []

        # making feature level(each value's step) dictionary
        for feature in features:
            levels = list(data[[feature]].groupby(feature).size().keys())
            self.feature_level_dict[feature] = levels

        # calculating point matrix
        for feature in features:

            by_level_arr = []
            for level in self.feature_level_dict[feature]:
                
                by_class_arr = []
                for class_name in self.class_names:
                    cnt = len(data[(data[feature] == level) & (data[label_col_name] == class_name)])
                    by_class_arr.append(cnt)
                
                by_level_arr.append(by_class_arr)

            self.recipe.append(by_level_arr)

        # calculating weight matrix 
        for i in range(0, len(self.recipe)):
            arr = np.transpose(self.recipe[i])
            
            weight_arr = []
            for i in range(0, len(self.class_names)):
                
                weight_arr.append(Appetizer.cal_weight(self,arr[i]).tolist())
            
            # When making weights to 2d matrix by each classes and features, classification is too much tilted to one side.
            # so decide to use juse mean value.
            mean = np.mean(weight_arr) 
            
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

    def get_feature_weight(self, train_data: list, label_col_name: str, feature_col_name: str):
        '''
        arguments
            train_data: list, 
            label_col_name: str, 
            feature_col_name: str
        '''
        class_names = list(train_data[[label_col_name]].groupby(label_col_name).size().keys())
        
        divided_col_by_classes = []
        for selected_class in class_names:
            divided_col_by_classes.append(
                list(train_data.loc[train_data[label_col_name] == selected_class, [label_col_name,feature_col_name]].groupby(feature_col_name).size())
            )
        
        summary_entropy = 0

        for val_list_by_class in divided_col_by_classes:
            summary_entropy += Appetizer.entropy_from_list(
                self,
                val_list_by_class
            )
        return summary_entropy / len(class_names)

    def get_feature_weight_vector(self, train_data: list, label_col_name: str, feature_col_name: str):
        '''
        arguments
            train_data: list, 
            label_col_name: str, 
            feature_col_name: str
        '''
        class_names = list(train_data[[label_col_name]].groupby(label_col_name).size().keys())
        
        divided_col_by_classes = []

        for selected_class in class_names:
            divided_col_by_classes.append(
                list(train_data.loc[train_data[label_col_name] == selected_class, [label_col_name,feature_col_name]].groupby(feature_col_name).size())
            )
        
        weight_vector = []

        for val_list_by_class in divided_col_by_classes:
            weight_vector.append(
                Appetizer.entropy_from_list(
                    self,
                    val_list_by_class
                )
            )
        return weight_vector
    
    def get_weight(self, train_data: list, label_col_name: str, feature_col_name: str, return_type = "scalar"):
        ''' return_type : scalar(or 0, default) | vector (or 1) '''
        
        if(return_type == "vector" or return_type == 1):
            return Classifier.get_feature_weight_vector(self, train_data, label_col_name, feature_col_name)
        else:
            return Classifier.get_feature_weight(self, train_data, label_col_name, feature_col_name)

    
    def getPredictions(self, data:pandas.DataFrame, mode = 0) -> list:
        ''' mode : 0|CLASS, 1|BOTH, 2|POINT '''
        
        if(mode == 'CLASS'):
            mode = 0
            pass
        elif(mode == 'BOTH'):
            mode = 1
            pass
        elif(mode == 'POINT'):
            mode = 2
            pass

        predictions = []
        # for i in range(0, data.shape[0]):
        for i in data.index.values:
            if(mode == 0):
                # return the array of classes
                predictions.append(Classifier.probability_to_class(self, Classifier.predict_row(self, data.loc[i, self.features].values.tolist())))
                pass
            elif(mode == 1):
                # return the both of Class and Probability.
                point_arr = Classifier.predict_row(self, data.loc[i, self.features].values.tolist())
                class_and_accuracy = {}
                class_and_accuracy["Class"] = Classifier.probability_to_class(self, point_arr) 
                class_and_accuracy["Probability"] = max(point_arr) / sum(point_arr) 
                predictions.append(class_and_accuracy)
                pass
            elif(mode == 2):
                # return the array of raw point by each classes
                # eg. [3.292727107245485, 1.5072688747234735]
                predictions.append( Classifier.predict_row(self, data.loc[i, self.features].values.tolist()) )
                pass
            else:
                return ["Invalid mode"]
        
        return predictions

    def accuracy(self, data:pandas.DataFrame, label_col_name:str) -> float:
        predictions = self.getPredictions(data)
        
        real_vals = data[label_col_name].tolist()

        if(len(predictions) != len(real_vals)):
            print(len(predictions))
            print(real_vals)
            print("the length is different with predictions and label list, while calculating accuracy")
            return 0
 
        correct_cnt = 0
        for i in range(0, len(predictions)):
            if(predictions[i] == real_vals[i]) : 
                correct_cnt += 1
        
        return "%.5f" % (correct_cnt / len(predictions))


class Appetizer:

    # def entropy(self, selected_col: pandas.DataFrame):
    #     counts_list = selected_col.groupby(list(selected_col)[0]).size()
    #     arr = list(counts_list)

    #     ent = 0
    #     for val in arr:
    #         ent += (val / sum(arr)) ** 2
    #     return ent * len(arr)

    def entropy_from_list(self, data:list):
        
        ent = 0
        for val in data:
            ent += (val / sum(data)) ** 2
        
        return ent

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

class Cluster:
    def initialize(self):
        return
