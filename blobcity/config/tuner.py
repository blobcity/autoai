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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from blobcity.main import modelSelection
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score
from blobcity.utils import Progress,scaling_data
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing,SimpleExpSmoothing, Holt
from blobcity.utils import * 
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,f1_score,precision_score,recall_score,confusion_matrix,mean_absolute_percentage_error
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

class EarlyStopper_time():
    iter_stop = 10
    iter_count = 0
    best_score = None
    criterion = 1 

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

def early_stopping_opt_time(study, trial):
    """
    param1:optuna.study.Study 
    param2:optuna trial
    
    The function decides whether to stop parameter tunning based on the accuracy of the current trial. 
    Condition to stop trials: if accuracy is more than equal to 99%. 
    The second condition is to check whether accuracy is between 90%-99% and 
    whether the current best accuracy has not changed for the last ten trials.

    """
    if EarlyStopper_time.best_score == None: EarlyStopper_time.best_score = study.best_value
    if study.best_value <= EarlyStopper_time.criterion : study.stop()
    if study.best_value < EarlyStopper_time.best_score:
        EarlyStopper_time.best_score = study.best_value
        EarlyStopper_time.iter_count = 0
    else:
        if  study.best_value > 1 and study.best_value < 10:
            if EarlyStopper_time.iter_count < EarlyStopper_time.iter_stop:
                EarlyStopper_time.iter_count=EarlyStopper_time.iter_count+1  
            else:
                EarlyStopper_time.iter_count = 0
                EarlyStopper_time.best_score = None
                study.stop()
                
    return

def time_metrics(y_true, y_pred):
    """
    Computes performance metrics for time series forecasting.

    Args:
        y_true (array-like): Actual observed values.
        y_pred (array-like): Forecasted/predicted values.

    Returns:
        dict: Dictionary containing R2-score, MAE, MSE, RMSE, and MAPE.

    Raises:
        ValueError: If y_true and y_pred have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    return {
        "R2": round(r2_score(y_true, y_pred), 5),
        "MAE": round(mean_absolute_error(y_true, y_pred), 5),
        "MSE": round(mean_squared_error(y_true, y_pred), 5),
        "RMSE": round(mean_squared_error(y_true, y_pred, squared=False), 5),
        "MAPE": round(mean_absolute_percentage_error(y_true, y_pred), 5),
    }

def regression_metrics(y_true, y_pred):
    """
    Computes regression performance metrics.

    Args:
        y_true (array-like): True target values (pandas Series, DataFrame, or numpy array).
        y_pred (array-like): Predicted values (pandas Series, DataFrame, or numpy array).

    Returns:
        dict: Dictionary containing R2-score, MAE, MSE, RMSE, and MAPE.

    Raises:
        ValueError: If y_true and y_pred have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    return {
        "R2": round(r2_score(y_true, y_pred), 3),
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "MSE": round(mean_squared_error(y_true, y_pred), 3),
        "RMSE": round(mean_squared_error(y_true, y_pred, squared=False), 3),
        "MAPE": round(mean_absolute_percentage_error(y_true, y_pred), 3),
    }

def classification_metrics(y_true, y_pred):
    """
    Computes classification performance metrics.

    Args:
        y_true (array-like): True labels (pandas Series, DataFrame, or numpy array).
        y_pred (array-like): Predicted labels (pandas Series, DataFrame, or numpy array).

    Returns:
        dict: Dictionary containing F1-score, Precision, and Recall (weighted average).

    Raises:
        ValueError: If y_true and y_pred have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    return {
        "F1-Score": round(f1_score(y_true, y_pred, average="weighted"), 3),
        "Precision": round(precision_score(y_true, y_pred, average="weighted"), 3),
        "Recall": round(recall_score(y_true, y_pred, average="weighted"), 3),
    }

def metricResults(y_true, y_pred, ptype, prog):
    """
    Computes evaluation metrics based on the problem type.

    Args:
        y_true (array-like): Actual target values.
        y_pred (array-like): Predicted values.
        ptype (str): Problem type - "Classification" or "Regression".
        prog (Progress): Progress bar object for tracking optimization.

    Returns:
        dict: Computed metrics based on the problem type.
    """
    if ptype in ["Classification", "Image Classification"]:
        results = classification_metrics(y_true, y_pred)
    elif ptype in ["Regression", "Timeseries"]:
        results = regression_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported problem type: {ptype}")

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
    Fetches different parameter values using the Optuna trial object.

    Args:
        trial (optuna.trial): The trial object for hyperparameter optimization.

    Returns:
        dict: A dictionary of suggested parameter values.
    """
    return {
        key: (
            trial.suggest_int(key, *arg) if datatype == "int" else
            trial.suggest_float(key, *arg) if datatype == "float" else
            trial.suggest_categorical(key, arg)
        )
        for key, value in parameter.items()
        for datatype, arg in value.items()
    }


def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial (optuna.Trial): An Optuna trial object.

    Returns:
        float: The mean cross-validation score for the model.
    """
    params = get_params(trial)  # Fetch hyperparameters
    model = modelName(**params)  # Initialize model with hyperparameters
    
    try:
        score = cross_val_score(model, X, Y, cv=cv, scoring='accuracy')  # Perform cross-validation
        accuracy = score.mean()
    except Exception as e:
        print(f"Error during cross-validation: {e}")
        return float('-inf')  # Return worst possible score in case of failure

    prog.update_progressbar(1)
    return accuracy


def prediction_data(y_true, y_pred, ptype, prog):
    """
    Generates data for plotting appropriate graphs/diagrams based on the problem type.

    Args:
        y_true (pandas.Series | numpy.ndarray): Actual values.
        y_pred (pandas.Series | numpy.ndarray): Predicted values.
        ptype (str): Problem type ('Classification', 'Image Classification', 'Timeseries', etc.).
        prog (object): Progress tracking object.

    Returns:
        array | 2D array: Confusion matrix for classification or prediction data for other problem types.
    """
    prog.update_progressbar(1)  # Update progress at the beginning

    try:
        if ptype in ['Classification', 'Image Classification']:
            return confusion_matrix(y_true, y_pred)

        elif ptype == "Timeseries":
            return [y_true.values, y_pred]

        else:  # Default case (Regression or other problem types)
            return [y_true.values, y_pred]

    except Exception as e:
        print(f"Error in prediction_data: {e}")
        return None  # Return None to handle failures gracefully
      

def tune_model(dataframe,target,modelkey,modelList,ptype,accuracy,DictionaryClass,stages):
    """
    param1: pandas.DataFrame
    param2: string : target column name
    param3: string : Model Key / Class name
    param4: dictionary : model dictionary 
    param5: string : Problem type either classification or reggression
    param6: float : Value to consider for stop model fining tune on desired accuracy criteria
    param7: class object
    return: tuple(model,parameter)dataframe

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
    if ptype!="Image Classification":X,Y=dataframe.drop(target,axis=1),dataframe[target]
    else: X,Y=dataframe,target
    cv=modelSelection.getKFold(X.shape[0])
    get_param_list(modelkey,modelList)
    EarlyStopper.criterion=accuracy
    n_trials=100
    try:
        if modelName().__class__.__name__ in ['SVC','NuSVC','LinearSVC','SVR','NuSVR','LinearSVR','KNeighborsClassifier','KNeighborsRegressor','RadiusNeighborsClassifier','RadiusNeighborsRegressor','NearestCentroid']:
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

def timeobjective(trial):
    """
    Optimizes a time-series forecasting model using Optuna.

    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter tuning.

    Returns:
        float: RMSE (Root Mean Squared Error) of the forecasted values.
    """
    params = get_params(trial)  # Fetch hyperparameters
    mdl = modelName(train_data1, **params)  # Initialize model

    try:
        mdl1 = mdl.fit(disp=0)  # Try fitting with disp=0 (for models that support it)
    except:
        mdl1 = mdl.fit()  # Fallback to default fitting

    # Generate predictions
    predictions = mdl1.forecast(len(test_data1))
    predictions = pd.Series(predictions, index=test_data1.index)

    # Calculate residuals and RMSE
    residuals = test_data1 - predictions
    rmse = round(np.sqrt(np.mean(residuals**2)), 5)

    return rmse
 
def time_tuner(train_data, test_data, modelkey, modelList, accuracy=None):
    """
    Performs hyperparameter tuning for a time-series forecasting model using Optuna.

    Args:
        train_data (pd.Series or pd.DataFrame): Training dataset.
        test_data (pd.Series or pd.DataFrame): Test dataset.
        modelkey (dict): Dictionary mapping model names to their configurations.
        modelList (dict): Dictionary containing model objects and hyperparameter ranges.
        accuracy (optional): Placeholder for compatibility; not used in this function.

    Returns:
        tuple: Contains the final trained model, best hyperparameters, best score, metric results, and plots.
    """
    # Set global train/test data variables
    global train_data1, test_data1
    train_data1, test_data1 = train_data, test_data

    # Initialize model parameters
    get_param_list(modelkey, modelList)

    # Create and optimize the Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(timeobjective, n_trials=30, callbacks=[early_stopping_opt_time])

    # Train the final model with the best parameters
    final_model = modelName(train_data1, **study.best_params).fit()

    # Generate predictions
    predictions = final_model.forecast(len(test_data1))
    predictions = pd.Series(predictions, index=test_data1.index)

    # Evaluate model performance
    metric_result = time_metrics(test_data1, predictions)

    # Generate plot data
    plots = prediction_data(test_data1, predictions, ptype="Timeseries")

    return final_model, study.best_params, study.best_value, metric_result, plots
