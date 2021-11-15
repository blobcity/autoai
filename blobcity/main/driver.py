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
import dill
import numpy as np
import pandas as pd
import autokeras as ak
import tensorflow as tf
from blobcity.store import DictClass
from blobcity.utils import get_dataframe_type,dataCleaner
from blobcity.utils import AutoFeatureSelection as AFS
from blobcity.main.modelSelection import model_search
from blobcity.code_gen import yml_reader,code_generator
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,f_regression,f_classif
def train(file=None, df=None, target=None,features=None,use_neural=False,accuracy_criteria=0.99):
    """
    param1: string: dataset file path 

    param2: (optional) pandas.DataFrame object

    param3: string: target/dependent column name.

    param4: float: range[0.1,1.0] 

    return: Model Class Object
    Performs a model search on the data proivded. A yaml file is generated once the best fit model configuration
    is discovered. The yaml file is later used for generating source code. 

    Input to the function must be one of file or data frame (df). Passing both parameters of file and df in a single
    invocation is an incorrect use.
    """
    dict_class=DictClass()
    dict_class.resetVar()
    #data read
    if file!=None:
        dataframe= get_dataframe_type(file, dict_class)
    else: 
        dataframe = df
        dict_class.addKeyValue('data_read',{"type":"df","class":"df"})
        
    if(features==None):
        featureList=AFS.FeatureSelection(dataframe,target,dict_class)
        CleanedDF=dataCleaner(dataframe,featureList,target,dict_class)
    else:
        CleanedDF=dataCleaner(dataframe,features,target,dict_class)
    #model search space
    accuracy_criteria= accuracy_criteria if accuracy_criteria<=1.0 else (accuracy_criteria/100)
    modelClass = model_search(CleanedDF,target,dict_class,use_neural=use_neural,accuracy_criteria=accuracy_criteria)
    modelClass.yamldata=dict_class.getdict()
    modelClass.feature_importance_=dict_class.feature_importance if(features==None) else calculate_feature_importance(CleanedDF.drop(target,axis=1),CleanedDF[target],dict_class)
    dict_class.resetVar()
    return modelClass

def load(model_path=None):
        """
        param1: string: (required) the filepath to the stored model. Supports .pkl models.
        returns: Model file

        function loads the serialized model from .pkl format to usable format.
        """
        if model_path not in [None,""]:
            path_components = model_path.split('.')
            extension = path_components[1] if len(path_components)<=2 else path_components[-1]
            base_path=os.path.splitext(model_path)[0]
            if extension == 'pkl':
                model = dill.load(open(model_path, 'rb'))  
                if model.yamldata['model']['type'] in ['TF','tf','Tensorflow']:
                    if model.yamldata['model']['save_type']=='h5':
                        h5_path=base_path+".h5"
                        if os.path.isfile(h5_path):model.model=tf.keras.models.load_model(h5_path)
                        else: raise FileNotFoundError(f"{h5_path} file doest exists in the directory")
                    elif model.yamldata['model']['save_type']=='pb':
                        if os.path.isdir(base_path):model.model=tf.keras.models.load_model(base_path, custom_objects=ak.CUSTOM_OBJECTS)
                        else: raise FileNotFoundError(f"{base_path} Folder doest exists")
                    else:
                        raise TypeError(f"{model.yamldata['model']['save_type']}, not supported save format")
                return model
            else:
                raise TypeError(f"{extension}, file type must be .pkl")
        else:
            raise TypeError(f"{model_path}, path can't be None or Null")
        

def spill(filepath,yaml_path=None,doc=None):
    """
    param1:string : filepath and format of generated file to store. either .py or .ipynb
    param2:string : filepath of already generated YAML file 
    param3:boolean : whether generate code along with documentation

    Function calls generator functions to generate source code for the AutoAI Procedure
    """
    if yaml_path in [None,""] : raise TypeError("YAML file path can't be None")
    data=yml_reader(yaml_path)
    code_generator(data,filepath,doc)

def calculate_feature_importance(X,Y,dict_class):
    """
    param1:pd.DataFrame
    param2:pd.Series/pd.DataFrame
    param3: class Object
    return: dictionary

    Function to calculate the feature importance of the features
    """
    if X.shape[1]>2:
        score_func=f_classif if(dict_class.getdict()['problem']["type"]=='Classification') else f_regression
        fit = SelectKBest(score_func=score_func, k=X.shape[1]).fit(X,Y)
        dfscores,dfcolumns = pd.DataFrame(fit.scores_),pd.DataFrame(X.columns)
        df = pd.concat([dfcolumns,dfscores],axis=1)
        df.columns = ['features','Score']
        df['Score']=MinMaxScaler().fit_transform(np.array(df['Score']).reshape(-1,1))
        imp=AFS.MainScore(dict(df.values),dict_class)
        return imp
    else:
        print('Dataset has only {} features, required atleast 2 for feature importances'.format(X.shape[1]))
        return None