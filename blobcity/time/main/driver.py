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

import os,dill
from pickle import NONE
import numpy as np
import pandas as pd
import warnings,copy
from blobcity.store import DictClass
from blobcity.aicloud import send_yaml_to_cloud
from sklearn.preprocessing import MinMaxScaler 
from blobcity.main.modelSelection import model_search
from blobcity.code_gen import yml_reader,code_generator
from sklearn.feature_selection import SelectKBest,f_regression,f_classif
from blobcity.utils import ProType,get_dataframe_type
from blobcity.utils import timeseries_cleaner


with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
    

def train(file=None, df=None, target=None,features=None,date=None,frequency_sampling_type=None):
    """
    param1: string: dataset file path 

    param2: (optional) pandas.DataFrame object

    param3: string: target/dependent column name.

    param4: list: List of features to consider for training

    param5: string:  whether to train on GOFAI algorithms or Neural Network, available options are ['classic','neural','all']

    param6: float: range[0.1,1.0]
    
    param7: boolean: whether to consider Multicolinearity check in Auto Feature Selection

    param8: int :  Number of epoches for Neural Network training

    param9: int :  Max number of Neural Network Models to try.

    return: Model Class Object

    Performs a model search on the data proivded. A yaml file is generated once the best fit model configuration
    is discovered. The yaml file is later used for generating source code. 

    Input to the function must be one of file or data frame (df). Passing both parameters of file and df in a single
    invocation is an incorrect use.
    """
    dict_class=DictClass()
    dict_class.resetVar()
    exp_id=ProType.generate_uuid()
    if file!=None:
        dataframe= get_dataframe_type(file, dict_class)
        
    else: 
        dataframe = df
        dict_class.addKeyValue('data_read',{"type":"df","class":"df"})
        
    if(features==None):
        #featureList=AutoFeatureSelection_dt.FeatureSelection(dataframe,target,dict_class,disable_colinearity)
        CleanedDF=timeseries_cleaner(dataframe,date,target,frequency_sampling_type,dict_class)
       
   
    print(dict_class.getdict()) 
    dict_class.resetVar()
    return None