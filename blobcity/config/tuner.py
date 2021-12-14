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
import optuna
import warnings
from blobcity.main import modelSelection
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score
from blobcity.utils import Progress,scaling_data
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,f1_score,precision_score,recall_score,confusion_matrix
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ["PYTHONWARNINGS"] = "ignore"
optuna.logging.set_verbosity(optuna.logging.WARNING)
"""
Python files consist of function to perform parameter tuning using optuna framework
"""
#Early stopping class
class EarlyStopper():
    iter_stop = 10
    iter_count = 0
    best_score = None
    criterion = 0.99

def early_stopping_opt(study, trial):
    """
    param1:optuna.study.Study 
    param2:optuna trial
    
    The function decides whether to stop parameter tunning based on the accuracy of the current trial. 
    Condition to stop trials: if accuracy is more than equal to 99%. 
    The second condition is to check whether accuracy is between 90%-99% and 
    whether the current best accuracy has not changed for the last ten trials.

    """
    if EarlyStopper.best_score == None: EarlyStopper.best_score = study.best_value
    if study.best_value >= EarlyStopper.criterion : study.stop()
    if study.best_value > EarlyStopper.best_score:
        EarlyStopper.best_score = study.best_value
        EarlyStopper.iter_count = 0
    else:
        if  study.best_value > 0.90 and study.best_value < 0.99:
            if EarlyStopper.iter_count < EarlyStopper.iter_stop:
                EarlyStopper.iter_count=EarlyStopper.iter_count+1  
            else:
                EarlyStopper.iter_count = 0
                EarlyStopper.best_score = None
                study.stop()
                
    return

def regression_metrics(y_true,y_pred):
    """
    param1: pandas.Series/pandas.DataFrame/numpy.darray
    param2: pandas.Series/pandas.DataFrame/numpy.darray 

    return: dictionary

    Function accept actual prediction labels from the dataset and predicted values from the model and utilizes this
    two values/data to calculate r2 score, mean absolute error, mean squared error, and root mean squared error at same time add them to result dictionary.
    Finally return the result dictionary 
    
    """
    result=dict()
    result['R2']=round(r2_score(y_true, y_pred),3)
    result['MAE']=round(mean_absolute_error(y_true, y_pred),3)
    result['MSE']=round(mean_squared_error(y_true, y_pred),3)
    result['RMSE']=round(mean_squared_error(y_true, y_pred,squared=False),3)
    return result

def classification_metrics(y_true,y_pred):
    """
    param1: pandas.Series/pandas.DataFrame/numpy.darray
    param2: pandas.Series/pandas.DataFrame/numpy.darray 

    return: dictionary

    Function accept actual prediction labels from the dataset and predicted values from the model and utilizes this
    two values/data to calculate f1 score,precision score, and recall for the classification problem. And finally 
    return them in a dictionary
    """
    result=dict()
    result['F1-Score']=round(f1_score(y_true, y_pred, average="weighted"),3)
    result['Precision']=round(precision_score(y_true, y_pred,average="weighted"),3)
    result['Recall']=round(recall_score(y_true, y_pred,average="weighted"),3)
    return result

def metricResults(y_true,y_pred,ptype,prog):
    """
    param1: model object (keras/sklearn/xgboost/catboost/lightgbm)
    param2: pandas.DataFrame
    param3: pandas.DataFrame/pandas.Series/numpy.darray
    param4: String

    return: Dictionary

    Function first perform an train test split of 80:20 split and train the selected model (with parameter tuning) 
    on training set. based on problem type call appropriate metric function either regression_metrics() or classification_metrics()
    return the resulting output(Dictionary).
    """
    results = classification_metrics(y_true,y_pred) if ptype =="Classification" else regression_metrics(y_true,y_pred)
    prog.update_progressbar(1)
    return results

def get_param_list(modelkey,modelList):
    """
    param1: dictionary
    param2: dictionary
    function initialize global variables required for parameter tuning and modelclass object.
    """
    global modelName
    global parameter
    Best1=list(modelkey.keys())[0]
    modelName,parameter=modelList[Best1][0],modelList[Best1][1]

def get_params(trial):
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
            elif datatype in ['str','bool','object']:
                params[key]=trial.suggest_categorical(key,arg)
    return params

def objective(trial):
    """
    param1: optuna.Trial
    return: float

    function trains model of randomized tuning parameter and return cross_validation score on specified kfold counts.
    the accuracy is average over the specified kfold counts.
    """
    params=get_params(trial)
    model=modelName(**params)
    score = cross_val_score(model, X, Y, cv=cv)
    accuracy = score.mean()
    prog.update_progressbar(1)
    return accuracy   

def prediction_data(y_true,y_pred,ptype,prog):
    """
    param1:pandas.Series/numpy.darray
    param2:pandas.Series/numpy.darray
    param3:string

    return:array/2D array

    Function generate data for ploting appropriate graph/diagram on the basis of problem type.
    """
    if ptype=='Classification':
        cm=confusion_matrix(y_true,y_pred)
        prog.update_progressbar(1)
        return cm
    else:
        data_pred=[y_true.values,y_pred]
        prog.update_progressbar(1)
        return data_pred        

def tune_model(dataframe,target,modelkey,modelList,ptype,accuracy,DictionaryClass,stages):
    """
    param1: pandas.DataFrame
    param2: string : target column name
    param3: string : Model Key / Class name
    param4: dictionary : model dictionary 
    param5: string : Problem type either classification or reggression
    param6: float : Value to consider for stop model fining tune on desired accuracy criteria
    param7: class object
    return: tuple(model,parameter)

    Function first fetchs required parameter details for the specific model by calling getParamList function and number of required kfold counts.
    then start a optuna study operation to fetch best tuning parameter for the model.
    then initialize the model with parameter and trains it on dataset.csv
    finally returns a tuple with consist of trained model and parameters.
    """
    global X
    global Y
    global cv
    global prog
    prog=Progress()
    X,Y=dataframe.drop(target,axis=1),dataframe[target]
    cv=modelSelection.getKFold(X)
    get_param_list(modelkey,modelList)
    EarlyStopper.criterion=accuracy
    n_trials=50
    try:
        if modelName().__class__.__name__ in ['SVC','NuSVC','LinearSVC','SVR','NuSVR','LinearSVR','KNeighborsClassifier','KNeighborsRegressor','RadiusNeighborsClassifier','RadiusNeighborsRegressor']:
            X = scaling_data(X,DictionaryClass,update=True)
                 
        prog.create_progressbar(n_trials,"Tuning {} (Stage 3 of {}) :".format(modelName().__class__.__name__,stages))
        study = optuna.create_study(direction="maximize")
        study.optimize(objective,n_trials=n_trials,callbacks=[early_stopping_opt])
        model = modelName(**study.best_params).fit(X,Y)
        metric_result=metricResults(Y,model.predict(X),ptype,prog)
        plots=prediction_data(Y,model.predict(X),ptype,prog)
        prog.update_progressbar(prog.trials)
        prog.close_progressbar()
        return (model,study.best_params,study.best_value,metric_result,plots)
    except Exception as e:
        print(e)