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

import optuna
from blobcity.main import modelSelection
from sklearn.model_selection import cross_val_score
optuna.logging.set_verbosity(optuna.logging.WARNING)
"""
Python files consist of function to perform parameter tuning using optuna framework
"""
def getParamList(modelkey,modelList):
    """
    param1: dictionary
    param2: dictionary
    function initialize global variables required for parameter tuning and modelclass object.
    """
    global modelName
    global parameter
    Best1=list(modelkey.keys())[0]
    modelName,parameter=modelList[Best1][0],modelList[Best1][1]

def getParams(trial):
    """
    param1: optuna.trial
    return: dictionary

    Function fetch different parameter values associated to model using appropriate optuna.trial class.
    then finally return the dictionary of parameters.
    """
    params=dict()
    for key,value in parameter.items():
        for datatype,arg in value.items():
            if datatype == "int":
                params[key]=trial.suggest_int(key,arg[0],arg[1])
            elif datatype=="float":
                params[key]=trial.suggest_float(key,arg[0],arg[1])
            elif datatype=='str':
                params[key]=trial.suggest_categorical(key,arg)
    return params

def objective(trial):
    """
    param1: optuna.Trial
    return: float

    function trains model of randomized tuning parameter and return cross_validation score on specified kfold counts.
    the accuracy is average over the specified kfold counts.
    """
    params=getParams(trial)
    model=modelName(**params)
    score = cross_val_score(model, X, Y, n_jobs=-1, cv=cv)
    accuracy = score.mean()
    return accuracy    

def tuneModel(dataframe,target,modelkey,modelList):
    """
    param1: pandas.DataFrame
    param2: string 
    param3: dictionary
    param4: dictionary
    return: tuple(model,parameter)

    Function first fetchs required parameter details for the specific model by calling getParamList function and number of required kfold counts.
    then start a optuna study operation to fetch best tuning parameter for the model.
    then initialize the model with parameter and trains it on dataset.csv
    finally returns a tuple with consist of trained model and parameters.
    """
    global X
    global Y
    global cv
    X,Y=dataframe.drop(target,axis=1),dataframe[target]
    cv=modelSelection.getKFold(X)
    getParamList(modelkey,modelList)
    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5,n_jobs=-1)
        model = modelName(**study.best_params).fit(X,Y)
        return (model,study.best_params)
    except Exception as e:
        print(e)
        return None