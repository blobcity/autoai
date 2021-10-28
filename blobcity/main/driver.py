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


# Takes data frame, and target only. 
# Should do automatic feature selection
# Should return a trained model
# Should output progress, in an interactive manner while training is in progress

import pickle
import numpy as np
import pandas as pd
from blobcity.store import DictClass
from blobcity.utils import getDataFrameType,dataCleaner
from blobcity.utils import AutoFeatureSelection as AFS
from blobcity.main.modelSelection import model_search
from blobcity.code_gen import yml_reader,code_generator
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,f_regression,f_classif
def train(file=None, df=None, target=None,features=None,accuracy_criteria=0.99):
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
        dataframe= getDataFrameType(file, dict_class)
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
    modelClass = model_search(CleanedDF,target,dict_class,use_neural=False,accuracy_criteria=accuracy_criteria)
    #return modelClass object
    modelClass.yamldata=dict_class.getdict()
    modelClass.feature_importance_=dict_class.feature_importance if(features==None) else calculate_feature_importance(CleanedDF.drop(target,axis=1),CleanedDF[target],dict_class)
    dict_class.resetVar()
    return modelClass
# Performs an automated model training. 


# Reads a BlobCity published model file, and loads it into memory.
# This can be combination of yml and other related artifacts of a trained model
# Need to see if a h5 file of TensorFlow, and a pickel file for other models can be combined into say a .bcm file for storage
# .bcm would be a custom format, standing for a BlobCity Model
def load(modelFile,h5_path=None):
        """
        param1: string: (required) the filepath to the stored model. Supports .pkl models.
        param2: string: the filepath to the stored h5 file, provide only if saved h5 file.
        returns: Model file

        function loads the serialized model from .pkl or .h5 format to usable format.
        """
        path_components = modelFile.split('.')
        extension = path_components[1] if len(path_components)<=2 else path_components[2]
         
        if extension == 'pkl' and h5_path in [None,""]:
            model = pickle.load(open(modelFile, 'rb'))

        """ elif os.path.splitext(h5_path)[1] == '.h5' and h5_path!=None:
            print("pkl path: {}, h5 path : {}".format(os.path.splitext(modelFile),os.path.splitext(h5_path)))
            if os.path.splitext(h5_path)[0] == os.path.splitext(modelFile)[0]:
                tfmodel = tf.keras.models.load_model(h5_path)
                model=pickle.load(open(modelFile, 'rb'))
                model.model=tfmodel
            else:
                raise ValueError("file name for pickle and h5 file should be same") """
        return model

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

