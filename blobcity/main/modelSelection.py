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
import numpy as np
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
from math import isnan
import warnings,itertools
from blobcity.store import Model
from sklearn.metrics import r2_score
from blobcity.config import tuner as Tuner
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from blobcity.config import classifier_config,regressor_config,time_config
from blobcity.utils import Progress,scaling_data,AutoFeatureSelection
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing,ExponentialSmoothing, Holt
from blobcity.utils import *
import warnings
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import autokeras as ak
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
"""
This python file consists of function to get best performing model for a given dataset.
"""

class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        prog.update_progressbar(1)
    def on_epoch_end(self, epoch, logs=None):
        prog.update_progressbar(1)

def getKFold(rows):

    """
    param1: pandas.DataFrame

    return : integer 

    Function returns number of kfold to consider for Cross validation on the basis of dataset row counts
    """

    if(rows>100 and rows<500):k=3
    elif(rows>500 and rows <=5000 ):k=5
    elif(rows>5000 and rows <=50000):k=7
    elif rows>50000: k=10
    else:k=3
    return k

def cv_score(model, X, Y, k):
    """
    Computes the cross-validation score for a given model.

    :param model: Scikit-learn/XGBoost/LightGBM/CatBoost model instance
    :param X: pd.DataFrame, Feature matrix
    :param Y: pd.Series or pd.DataFrame, Target variable
    :param k: int, Number of cross-validation folds
    :return: float, Mean cross-validation score

    Uses cross_val_score to calculate average accuracy on k-folds.
    Optimizes parallel execution based on model type.
    """
    
    # Restrict parallel processing for specific models
    restricted_models = {'XGBClassifier', 'XGBRegressor', 'LGBMRegressor', 'LGBMClassifier', 
                         'CatBoostRegressor', 'CatBoostClassifier'}
    n_jobs = 1 if model.__class__.__name__ in restricted_models else -1

    return cross_val_score(model, X, Y, cv=k, n_jobs=n_jobs).mean()

def sort_score(modelScore):
    """
    param1: Dictionary
    return: Dictionary

    Function returns a sorted dictionary on the basis of values.
    """
    sorted_dict=dict(sorted(modelScore.items(), key=lambda item: item[1],reverse=True))
    return sorted_dict

def eval_model(models, model_key, X, Y, k, DictionaryClass):
    """
    Evaluates a model using cross-validation.

    :param models: dict, Dictionary of model constructors
    :param model_key: str, Key to select model
    :param X: pd.DataFrame, Feature matrix
    :param Y: pd.Series or np.array, Target variable
    :param k: int, Number of cross-validation splits
    :param DictionaryClass: Class object, Used for scaling if required
    :return: float, Cross-validation score

    If the model belongs to distance-based algorithms, scales the data to speed up training.
    """
    
    # Apply scaling for distance-based models
    distance_based_models = {
        'SVC', 'NuSVC', 'LinearSVC', 'SVR', 'NuSVR', 'LinearSVR',
        'KNeighborsClassifier', 'KNeighborsRegressor', 
        'RadiusNeighborsClassifier', 'RadiusNeighborsRegressor', 'NearestCentroid'
    }
    
    if model_key in distance_based_models:
        X = scaling_data(X, DictionaryClass)

    # Handle verbosity for specific models
    verbosity_params = {
        'XGBClassifier': {'verbosity': 0},
        'XGBRegressor': {'verbosity': 0},
        'CatBoostRegressor': {'verbose': False},
        'CatBoostClassifier': {'verbose': False},
        'LGBMClassifier': {'verbose': -1},
        'LGBMRegressor': {'verbose': -1}
    }
    
    model_class = models[model_key][0]
    model = model_class(**verbosity_params.get(model_key, {}))

    return cv_score(model, X, Y, k)


def train_on_sample_data(dataframe, target, models, DictionaryClass, stages):
    """
    Trains models on a sampled subset of the data and returns the top 5 models with the best accuracy.

    :param dataframe: pandas.DataFrame, dataset
    :param target: str, target column name
    :param models: dict, mapping of model names to model objects (sklearn/xgboost/lightgbm/catboost)
    :param DictionaryClass: object, dictionary class with metadata
    :param stages: int, total number of search stages

    :return: dict, top 5 models with the best accuracy scores
    """

    # Determine sample rate based on dataset size
    rows = dataframe.shape[0]
    sample_rate = min(round((500 + ((rows - 500) * 0.2)) / rows, 1), 1.0)  # Ensuring sample_rate <= 1

    # Handle Image Classification separately
    if DictionaryClass.YAML['problem']['type'] == "Image Classification":
        dataframe = dataframe.sample(frac=1, random_state=123)  # Shuffle dataset
        df = dataframe.sample(frac=sample_rate, random_state=123)
        X, Y = AutoFeatureSelection.get_reshaped_image(df.values)
        models_List = DictionaryClass.image_models
    else:
        df = dataframe.sample(frac=sample_rate, random_state=123)
        X, Y = df.drop(target, axis=1), df[target]
        models_List = models.keys()

    # Get the number of folds for cross-validation
    k = getKFold(rows)
    
    modelScore = {}
    prog.create_progressbar(len(models_List), f"Quick Search (Stage 1 of {stages}):")

    for model_name in models_List:
        try:
            modelScore[model_name] = eval_model(models, model_name, X, Y, k, DictionaryClass)
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")  # Log errors instead of passing silently
        prog.update_progressbar(1)

    prog.close_progressbar()

    # Remove NaN values and sort models by accuracy
    clean_dict = {k: v for k, v in modelScore.items() if not np.isnan(v)}
    return dict(itertools.islice(sort_score(clean_dict).items(), 5))  # Return top 5 models

def train_on_full_data(X, Y, models, best, DictionaryClass, stages):
    """
    Trains selected models on the full dataset and returns the best model based on accuracy.

    :param X: pandas.DataFrame, feature dataset
    :param Y: pandas.Series or pandas.DataFrame, target dataset
    :param models: dict, mapping of model names to model objects (sklearn/xgboost/lightgbm/catboost)
    :param best: dict, selected model names from Stage 1
    :param DictionaryClass: object, dictionary class with metadata
    :param stages: int, total number of search stages

    :return: dict, single best model with the highest accuracy
    """

    # Get the number of folds for cross-validation
    k = getKFold(X.shape[0])
    modelScore = {}

    # Adjust best model list for Image Classification
    if DictionaryClass.YAML['problem']['type'] == "Image Classification" and len(best) > 5:
        best = DictionaryClass.image_models  # Use predefined image models

    # Initialize progress bar
    prog.create_progressbar(len(best), f"Deep Search (Stage 2 of {stages}):")

    for model_name in best:
        try:
            modelScore[model_name] = eval_model(models, model_name, X, Y, k, DictionaryClass)
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")  # Improved error handling
        prog.update_progressbar(1)

    prog.close_progressbar()

    # Remove NaN values and return the best model
    clean_dict = {k: v for k, v in modelScore.items() if not np.isnan(v)}
    return dict(itertools.islice(sort_score(clean_dict).items(), 1))  # Return the single best model

def train_on_neural(X, Y, ptype, epochs, max_neural_search, stage, ofstage):
    """
    Performs neural network model search and tuning using AutoKeras and returns the best Keras model.

    :param X: pandas.DataFrame, feature dataset
    :param Y: pandas.Series or pandas.DataFrame, target dataset
    :param ptype: str, problem type ('Classification', 'Regression', 'Image Classification')
    :param epochs: int, number of training epochs
    :param max_neural_search: int, maximum number of trials for AutoKeras search
    :param stage: int, current training stage
    :param ofstage: int, total number of stages

    :return: tuple (AutoKeras model, accuracy, metric results, prediction data)
    """

    # Initialize progress bar
    total_steps = (max_neural_search + 2) * epochs
    prog.create_progressbar(n_counters=total_steps, desc=f"Neural Networks (stage {stage} of {ofstage})")

    # Select the appropriate AutoKeras model
    if ptype == "Image Classification":
        clf = ak.ImageClassifier(overwrite=True, max_trials=max_neural_search)
    elif ptype == "Classification":
        clf = ak.StructuredDataClassifier(overwrite=True, max_trials=max_neural_search)
    else:  # Regression
        clf = ak.StructuredDataRegressor(overwrite=True, max_trials=max_neural_search)

    # Train the model
    clf.fit(X, Y, verbose=0, epochs=epochs, callbacks=[CustomCallback()], batch_size=8)

    # Evaluate the model
    loss, acc = clf.evaluate(X, Y, verbose=0)
    y_pred = clf.predict(X, verbose=0)

    # Handle classification predictions
    if ptype in ["Classification", "Image Classification"]:
        y_pred = y_pred.astype(int)  # Use int() instead of np.int (deprecated in NumPy 1.20+)

    # Handle regression metrics
    if ptype == "Regression":
        acc = r2_score(Y, y_pred)

    # Compute metrics and plot data
    results = Tuner.metricResults(Y, y_pred, ptype, prog)
    plot_data = Tuner.prediction_data(Y, y_pred, ptype, prog)

    # Finalize progress bar
    prog.update_progressbar(prog.trials)
    prog.close_progressbar()

    return clf, acc, results, plot_data


def classic_model(ptype, dataframe, target, X, Y, DictClass, modelsList, accuracy_criteria, stages):
    """
    Performs a two-stage model search and hyperparameter tuning for classification or regression.

    :param ptype: str, problem type ('Classification' or 'Regression')
    :param dataframe: pd.DataFrame, input dataset
    :param target: str, target column name
    :param X: pd.DataFrame, feature dataset
    :param Y: pd.Series or pd.DataFrame, target dataset
    :param DictClass: Class object, contains configurations and YAML data
    :param modelsList: dict, model class and configurations based on problem type
    :param accuracy_criteria: float, threshold to stop further hyperparameter tuning
    :param stages: int, total stages for progress bar visualization
    :return: tuple, consisting of trained model and evaluation results

    The function performs a two-stage model search:
    1. If dataset size > 500, it selects the top 5 models using a 10% sample and evaluates them on the full dataset.
    2. Runs the best-performing model through hyperparameter tuning using `Tuner.tune_model`.
    """

    # Stage 1: Quick Search on Sample Data (if dataset > 500 rows)
    if dataframe.shape[0] > 500:
        sampled_best_models = train_on_sample_data(dataframe, target, modelsList, DictClass, stages)
        best_model = train_on_full_data(X, Y, modelsList, sampled_best_models, DictClass, stages)
    else:
        print(f"Quick Search (Stage 1 of {stages}) is skipped.")
        best_model = train_on_full_data(X, Y, modelsList, 
                                        DictClass.image_models if ptype == "Image Classification" else modelsList, 
                                        DictClass, stages)

    # Stage 2: Hyperparameter Tuning
    modelResult = Tuner.tune_model(
        X if ptype == "Image Classification" else dataframe, 
        Y if ptype == "Image Classification" else target, 
        best_model, modelsList, ptype, accuracy_criteria, DictClass, stages
    )

    return modelResult


def classic_model_records(modelData, modelResult, DictClass):
    """
    Extracts model details from `modelResult` and updates `modelData` and YAML configuration.

    :param modelData: Class object, stores model-related attributes
    :param modelResult: tuple, contains model, parameters, accuracy, metrics, and plot data
    :param DictClass: Class object, manages YAML configuration updates
    :return: Class object with updated model attributes
    """

    # Unpack modelResult tuple into modelData attributes
    modelData.model, modelData.params, _, modelData.metrics, modelData.plot_data = modelResult

    # Update YAML configuration with model type and parameters
    DictClass.addKeyValue('model', {'type': modelData.model.__class__.__name__})
    DictClass.UpdateNestedKeyValue('model', 'parameters', modelData.params)

    # Assign scaler from DictClass
    modelData.scaler = DictClass.Scaler

    return modelData

def neural_model_records(modelData, neural_network, DictClass, ptype, dataframe, target):
    """
    Updates the Model Class object attributes with data from the neural network model.

    :param modelData: Class object storing model details
    :param neural_network: Tuple containing model, accuracy, metrics, and plot data
    :param DictClass: Class object managing YAML configuration
    :param ptype: str, Problem type ('Classification', 'Image Classification', 'Regression')
    :param dataframe: pd.DataFrame, Dataset
    :param target: str, Target column name
    :return: Updated modelData object

    Extracts data from the neural_network tuple and assigns it to modelData attributes.
    Updates the YAML configuration accordingly.
    """

    modelData.model, acc, modelData.metrics, modelData.plot_data = neural_network

    # Update model type
    DictClass.addKeyValue('model', {'type': 'TF'})

    # Handle classification-specific details
    if ptype in {"Classification", "Image Classification"}:
        n_labels = dataframe[target].nunique(dropna=False)
        classification_type = 'binary' if n_labels <= 2 else 'multiclass'

        DictClass.UpdateNestedKeyValue('model', 'classification_type', classification_type)
        DictClass.UpdateNestedKeyValue('model', 'save_type', "h5")
    
    # Handle regression-specific details
    elif ptype == 'Regression':
        DictClass.UpdateNestedKeyValue('model', 'save_type', "pb")

    return modelData

def model_search(
    dataframe=None,
    target=None,
    DictClass=None,
    disable_colinearity=False,
    model_types="all",
    accuracy_criteria=0.99,
    epochs=20,
    max_neural_search=10
):
    """
    Conducts a model search using classical and/or neural network approaches.

    :param dataframe: pd.DataFrame, Input dataset
    :param target: str, Target column name
    :param DictClass: Class object managing dataset metadata
    :param disable_colinearity: bool, Option to disable collinearity checking
    :param model_types: str, Model selection method ('classic', 'neural', 'all')
    :param accuracy_criteria: float, Accuracy threshold (0.1 to 1.0)
    :param epochs: int, Number of epochs for neural networks
    :param max_neural_search: int, Maximum number of neural models to search
    :return: Class object with the best model and details
    """
    global prog
    prog = Progress()
    modelData = Model()
    
    ptype = DictClass.getdict()['problem']["type"]
    is_classification = ptype in {"Classification", "Image Classification"}
    is_regression = ptype in {"Classification", "Regression"}

    # Get appropriate model dictionary
    modelsList = classifier_config().models if is_classification else regressor_config().models

    # Extract features & target
    if is_regression:
        X, Y = dataframe.drop(columns=[target]), dataframe[target]
    elif ptype == "Image Classification":
        X, Y = AutoFeatureSelection.get_reshaped_image(dataframe.values)

    # Handle target encoding and feature lists
    if is_classification:
        modelData.target_encode = DictClass.get_encoded_label()
    if is_regression:
        modelData.featureList = dataframe.drop(columns=[target]).columns.to_list()

    ### Stage-wise Model Training ###
    if model_types == 'classic':
        modelResult = classic_model(ptype, dataframe, target, X, Y, DictClass, modelsList, accuracy_criteria, 3)
        DictClass.accuracy = round(modelResult[2], 3)
        modelData = classic_model_records(modelData, modelResult, DictClass)
        class_name = modelData.model.__class__.__name__

    elif model_types == 'neural':
        if ptype == "Image Classification":
            X = X.reshape(DictClass.original_shape)
        
        # Check for GPU availability
        if not tf.config.list_physical_devices('GPU'):
            print("No GPU detected. Running on CPU. Consider using a GPU for faster training.")

        neural_network = train_on_neural(X, Y, ptype, epochs, max_neural_search, 1, 1)
        DictClass.accuracy = round(neural_network[1], 3)
        modelData = neural_model_records(modelData, neural_network, DictClass, ptype, dataframe, target)
        class_name = "Neural Network"

    elif model_types == 'all':
        modelResult = classic_model(ptype, dataframe, target, X, Y, DictClass, modelsList, accuracy_criteria, 4)

        if modelResult[2] < accuracy_criteria:
            # Check for GPU availability
            if not tf.config.list_physical_devices('GPU'):
                print("No GPU detected. Running on CPU.")

            neural_network = train_on_neural(X, Y, ptype, epochs, max_neural_search, 4, 4)

            if modelResult[2] > neural_network[1]:
                DictClass.accuracy = round(modelResult[2], 3)
                modelData = classic_model_records(modelData, modelResult, DictClass)
                class_name = modelData.model.__class__.__name__
            else:
                # Clean up rescale settings for neural networks
                DictClass.YAML.get('cleaning', {}).pop('rescale', None)

                if ptype == "Image Classification":
                    X = X.reshape(DictClass.original_shape)

                DictClass.accuracy = round(neural_network[1], 3)
                modelData = neural_model_records(modelData, neural_network, DictClass, ptype, dataframe, target)
                class_name = "Neural Network"
        else:
            print("Neural Network Search (Stage 4 of 4) skipped")
            DictClass.accuracy = round(modelResult[2], 3)
            modelData = classic_model_records(modelData, modelResult, DictClass)
            class_name = modelData.model.__class__.__name__

    ### Final Steps: Colinearity Check & Model Selection ###
    if not disable_colinearity and ptype != "Image Classification" and DictClass.accuracy < 0.8:
        print("Recommendation: Disable Colinearity in train function")

    accuracy_label = "CV Score" if class_name != "Neural Network" else "Accuracy Score"
    print(f"Selected Model :- {class_name} \n{accuracy_label} : {DictClass.accuracy:.2f}")

    return modelData


def time_model(dataframe, DictClass, accuracy_criteria=None):
    """
    Trains and tunes a time series model.

    :param dataframe: pd.DataFrame, Input dataset for time series modeling.
    :param DictClass: Class object containing dataset metadata.
    :param accuracy_criteria: float, Accuracy threshold for model selection (optional).
    :return: Tuned time series model results.
    """
    modelsList = time_config().models  # Fetch available time series models

    # Split dataset into training and testing sets
    train_data, test_data = spliter(dataframe)

    # Identify the best initial model
    model_key = model_search_time(train_data, test_data)

    # Perform model tuning
    model_result = Tuner.time_tuner(train_data, test_data, model_key, modelsList)

    return model_result



def model_search_time(train_data, test_data):
    """
    Identifies the best time series model based on RMSE.

    :param train_data: pandas Series, training data
    :param test_data: pandas Series, test data
    :return: Dictionary with the selected model name
    """
    global modelkey

    models = {
        "ARIMA": ARIMA(train_data, order=(1, 0, 0)),
        "SARIMAX": SARIMAX(train_data, order=(1, 0, 0), seasonal_order=(0, 0, 0, 12)),
        "ExponentialSmoothing": ExponentialSmoothing(train_data, initialization_method='estimated'),
        "SimpleExpSmoothing": SimpleExpSmoothing(train_data),
        "Holt": Holt(train_data),
    }

    errors = {}
    for name, model in models.items():
        try:
            fitted_model = model.fit()
            predictions = fitted_model.forecast(len(test_data))
            rmse = np.sqrt(mean_squared_error(test_data, predictions))
            errors[name] = rmse
        except Exception as e:
            print(f"Model {name} failed: {e}")  # Logs failed models

    if not errors:
        raise ValueError("All models failed to fit.")

    # Select model with the lowest RMSE
    selected_model = min(errors, key=errors.get)
    modelkey = {selected_model: 0}
    return modelkey