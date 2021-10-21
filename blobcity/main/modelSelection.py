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
import warnings
import itertools
from math import isnan
from tqdm import tqdm,tqdm_notebook
from blobcity.store import Model
from blobcity.config import tuner as Tuner
from blobcity.config import classifier_config,regressor_config
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ["PYTHONWARNINGS"] = "ignore"

"""
This python file consists of function to get best performing model for a given dataset.
"""
def getKFold(X):

    """
    param1: pandas.DataFrame

    return : integer 

    Function returns number of kfold to consider for Cross validation on the basis of dataset row counts
    """
    k=2
    rows=X.shape[0]
    if(rows>300 and rows<=500): k=2
    elif(rows>500 and rows <=5000 ):k=5
    elif rows>5000 : k=10
    return k

def cv_score(model,X,Y,k):

    """
    param1: sklearn/xgboost/lightgbm/catboost model 
    param2: pandas.DataFrame
    param3: pandas.DataFrame/pandas.Series
    param4: integer for kfold 
    return: float

    function gets above mentioned argument and uses cross_val_score to calculate average accuracy on specified kfolds
    """
    accuracy = cross_val_score(model, X, Y, cv = k,n_jobs=-1)
    return accuracy.mean()

def sort_score(modelScore):
    """
    param1: Dictionary
    return: Dictionary

    Function returns a sorted dictionary on the basis of values.
    """
    sorted_dict=dict(sorted(modelScore.items(), key=lambda item: item[1],reverse=True))
    return sorted_dict

def train_on_sample_data(dataframe,target,models,DictClass):
    """
    param1: pandas.DataFrame
    param2: string
    param3: Dictionary (sklearn/xgboost/lightgbm/catboost model object)
    param4: Class object
    return: Dictionary

    Function returns top 5 models with best accuracy in a dictionary.
    The models where accuracy is calculate on sampled data from the dataset.
    Accuracy is calculated using average cross validation score on specified kfold counts.
    """
    rows=dataframe.shape[0]
    sample_rate=round((500+(((rows-500)*0.2)))/rows,1)
    df=dataframe.sample(frac=sample_rate,random_state=123)
    X,Y=df.drop(target,axis=1),df[target]
    k=getKFold(X)
    modelScore={m:cv_score(models[m][0](),X,Y,k) for m in tqdm_notebook(models)}
    clean_dict = {k: modelScore[k] for k in modelScore if not isnan(modelScore[k])}
    return dict(itertools.islice(sort_score(clean_dict).items(), 5))

def train_on_full_data(dataframe,target,models,best,DictClass):
    """
    param1: pandas.DataFrame
    param2: string
    param3: Dictionary (sklearn/xgboost/lightgbm/catboost model object)
    param4: Class object
    return: Dictionary

    Function returns single best model with best accuracy in a dictionary. 
    Accuracy is calculated using average cross validation score on specified kfold counts.
    """
    X,Y=dataframe.drop(target,axis=1),dataframe[target]
    k=getKFold(X)
    modelScore={m:cv_score(models[m][0](),X,Y,k) for m in tqdm_notebook(best)}
    clean_dict = {k: modelScore[k] for k in modelScore if not isnan(modelScore[k])}
    return dict(itertools.islice(sort_score(clean_dict).items(), 1))

def model_search(dataframe,target,DictClass):
    """
    param1: pandas.DataFrame
    param2: string
    param3: Class object
    return: Class object

    Function first fetches model dictionary which consists of model object and required parameter,
    on the basis of problem type a specific class variable for dictionary is fetched.
    If the dataset has more than 500 entries then first fetch top 5 best performing models on 10% sample dataset 
    then run the top 5 models on full dataset to get single best model for hyper parameter tuning.
    The single model is then sent for hyper parameter tuning to tuneModel function of tuner Class,
    which returns a model object and model tuning parameters which are assigned to class variable in model class.
    Then update YAML dictionary with appropriate model details such has selected type and parameters.
    Function finally return a model class object.
    """
    ptype=DictClass.getdict()['problem']["type"]
    modelsList=classifier_config().models if ptype=="Classification" else regressor_config().models
    if dataframe.shape[0]>500:
        best=train_on_full_data(dataframe,target,modelsList,train_on_sample_data(dataframe,target,modelsList,DictClass),DictClass)
    else:
        best=train_on_full_data(dataframe,target,modelsList,modelsList,DictClass)
    modelResult = Tuner.tune_model(dataframe,target,best,modelsList,ptype)
    modelData=Model()
    modelData.featureList=dataframe.drop(target,axis=1).columns.to_list()
    modelData.model,modelData.params,acc,modelData.metrics,modelData.plot_data = modelResult
    DictClass.addKeyValue('model',{'type': modelData.model.__class__.__name__})
    DictClass.UpdateNestedKeyValue('model','parameters',modelResult[1])
    print("{} CV Score : {:.2f}".format(modelData.model.__class__.__name__,acc))
    return modelData

