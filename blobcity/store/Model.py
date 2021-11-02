# Copyright 2021 BlobCity, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from blobcity.utils import get_dataframe_type,Progress,write_dataframe
from blobcity.code_gen import code_generator
import yaml
"""
Python file consists of Class Model to initialize/store and retrive data associated to trained machine learning model.
"""
class Model:
    params=dict()
    featureList=[]
    model=None
    metrics=dict()
    yamldata=None
    feature_importance_=dict()
    target_encode=dict()
    plot_data=None
    def __init__(self):
        self.params=dict()
        self.featureList=[]
        self.model=None
        self.metrics=dict()
        self.yamldata=None
        self.feature_importance_=dict()
        self.plot_data=None
        self.target_encode=dict()
    
    def __quick_clean(self,test_dataframe):
        """
        """
        cols=test_dataframe.columns.to_list()
        for i in cols:
            if(isinstance(test_dataframe[i], pd.Series) and (test_dataframe[i].dtype in ["float64","int64","float","int"])):
                if(len(np.unique(test_dataframe[i]))<=3):
                    test_dataframe[i].fillna(test_dataframe[i].mode()[0],inplace=True)
                else:
                    test_dataframe[i].fillna(test_dataframe[i].mean(),inplace=True)
            elif(isinstance(test_dataframe[i], pd.Series)):
                test_dataframe[i].fillna(test_dataframe[i].mode()[0],inplace=True)
        if "object" in test_dataframe.dtypes.to_list():
            test_dataframe=pd.get_dummies(test_dataframe)
        return test_dataframe
        
    def predict(self,test,return_type="list",path=""):
        """
        param1: self
        param2: pd.DataFrame
        param3: string : either list or pd.DataFrame to return 
        param4: string : file path to store output pd.DataFrame,supported file types {'csv','json','xlsx'} 
        return: List/pd.DataFrame

        Function returns List/Array for predicted value from the trained model.
        """
        test=get_dataframe_type(test)
        if isinstance(test,pd.DataFrame):test=Model().__quick_clean(test[self.yamldata['features']['X_values']])
        if self.model.__class__.__name__ not in ['XGBClassifier','XGBRegressor']:result=self.model.predict(test)
        else:
            if type(test)=="list":
                test_df=pd.DataFrame(test, columns=self.featureList)
                result=self.model.predict(test_df)  
            else:
                result=self.model.predict(test) 
            
        if self.yamldata['problem']["type"]=='Classification':
            main_result=[]
            if self.target_encode!={}:
                for i in range(len(result)):
                    for k in self.target_encode.keys():
                        if k == result[i]:
                            main_result.append(self.target_encode[k])
                result= main_result 
        
        result_dataframe=test.copy(deep=True)
        result_dataframe['prediction']=result
        
        if path!="":write_dataframe(dataframe=result_dataframe,path=path)
        if return_type=="list": return result
        elif return_type=="df" and isinstance(test,pd.DataFrame):return result_dataframe

    def parameters(self):
        """
        return: Dictionary

        Function returns dictionary consisting of tuned parameters value for the trained model.
        """
        return self.params

    def features(self):
        """
        return: List/Array

        Function returns List of features used by model to train. This is also used to recognise the features to be passed as input into the predict function.
        """
        return self.featureList

    def save(self, model_path=None):
        """
        param: Entire Path for model pickle file, Supported formats is .pkl.
        
        Function saves the model and its weights serially and returns the filepath where it is saved.
        """
        if model_path not in [None,""]:
            path_components = model_path.split('.')
            extension = path_components[1] if len(path_components)<=2 else path_components[-1]

            if extension == '/':
                final_path = os.path.join(model_path, 'autoaimodel.pkl')
                pickle.dump(self, open(final_path, 'wb'))
                print("The model is stored at {}".format(final_path))
            elif extension == 'pkl':
                final_path = model_path
                pickle.dump(self, open(final_path, 'wb'))
                print("The model is stored at {}".format(final_path))
             
                """ 
            elif extension == 'h5' or self.yamldata['model']['type'] in ['TF','tf']:
                model_path = model_path if model_path!="./" else os.path.join(model_path, 'autoaimodel.h5')
                class_path = model_path if model_path!="./" else os.path.join(model_path, 'autoaimodel.pkl')
                try:
                    tfmodel_temp=self.model
                    self.model.save(model_path)
                    self.model=None
                    pickle.dump(self, open(class_path, 'wb'))
                    self.model=tfmodel_temp
                    print("The model is stored at {}".format(model_path))
                    return model_path
                except:
                    raise TypeError("Your model is not a Keras model of type .h5. Try .pkl extension.") """  
            else:
                raise TypeError(f"{extension} file type must be .pkl")
        else:
            raise ValueError("model_path cant be None or empty string")


    def stats(self):
        """
        Function to print/log/display all the metrics associated with problem type for the selected trained model. Usally used to check the effectiveness of training, or to assess the model fit. 
        """
        print ("\nMetrics on trained data:\n{:<10} {:<10}".format('METRIC', 'VALUE'))
        for key, value in self.metrics.items():
            print ("{:<10} {:<10}".format(key, value))

    def spill(self,filepath=None,doc=None):
        """
        param1: string : Filepath and format of generated file to store. either .py or .ipynb
        param2: boolean :  Whether generate code along with documentation.

        Function calls generator functions to generate source code for the AutoAI Procedure
        """
        if filepath not in [None,""]:
            data=self.yamldata
            code_generator(data,filepath,doc)
        else:
            raise ValueError("filepath can't be None or empty string")

    def generate_yaml(self,path=None):
        """
        param1: string : File path to store .yaml file,if not specified store in current directory with `Process.yaml`
        
        Function generated and create YAML configuration file for the complete AutoAI procedures.
        """
        if path!=None:
            extension = os.path.splitext(path)[1]
            filepath=path
        else:
            filepath = './Process.yaml'
            extension=".yaml"
        if extension in [".yaml",".yml"]:
            with open(filepath, 'w') as file:
                yaml.dump(self.yamldata, file,sort_keys=False)
        else:
            raise TypeError(f"{extension} file type must be .yml or .yaml")

    def plot_feature_importance(self):
        """
        Function to plot feature importance calculated during auto feature selection.
        """
        val=self.feature_importance_
        if(val!=None):
            val=dict(sorted(val.items(), key=lambda item: item[1],reverse=False))
            plt.figure(figsize = (12, 6))
            plt.barh(range(len(val)), list(val.values()), align='center')
            plt.yticks(range(len(val)),list(val.keys()))
            plt.show()
        else:
            print("Feature importance not available for dataset with less then 2 columns") 

    def get_prediction_data(self):
        """
        return: array/2D array/Dictionary/pandas.Dataframe
        Function returns predicted and actual target data.
        """
        problem=self.yamldata['problem']["type"]
        if problem=='Classification':
            return self.plot_data
        elif problem=="Regression":
            return {'true':self.plot_data[0],'predicted':self.plot_data[1]}

            
    def plot_prediction(self,n_rows=1000):
        """
        param1: integer : signed and unsigned integer for plot number of records for regression problem.

        Function plots either confusion matrix or regression plot(line plot comparing true and predicted values) based on problem type.
        """
        
        problem=self.yamldata['problem']["type"]
        if problem=='Classification':
            targets=self.target_encode.values()
            cf_matrix=self.plot_data
            group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
            labels = [f'{v1}\n\n{v2}' for v1, v2 in zip(group_counts,group_percentages)]
            labels = np.asarray(labels).reshape(cf_matrix.shape[0],cf_matrix.shape[0])
            sns.heatmap(cf_matrix, annot=labels, fmt='',xticklabels=targets,yticklabels=targets,cmap='Blues')
            plt.show()
        elif problem=="Regression":
            #plot for regression problem
            if  abs(n_rows)!=0:
                if abs(n_rows)<=len(self.plot_data[0]) or abs(n_rows)==1000:
                    n=len(self.plot_data[0]) if len(self.plot_data[0])<abs(n_rows) else n_rows
                    if n < 0: true,predict=self.plot_data[0][n:],self.plot_data[1][n:]
                    else:true,predict=self.plot_data[0][0:n],self.plot_data[1][0:n]
                    plt.figure(figsize=(14,10))
                    plt.plot(range(abs(n)),true, color = "green")
                    plt.plot(range(abs(n)),predict,linestyle='--',color = "red")
                    plt.legend(["Actual","prediction"]) 
                    plt.xlabel("Record number")
                    plt.ylabel(self.yamldata['features']['Y_values'])
                    plt.show()
                elif abs(n_rows)>len(self.plot_data[0]):
                    raise ValueError("entered row counts {} more than actual row counts {}".format(abs(len(self.plot_data[0])-abs(n_rows)),len(self.plot_data[0])))
            else:
                raise ValueError("Number of rows can't be Zero")