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


import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn import tree,ensemble,svm,linear_model,neighbors
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


"""
This python file consist of Class variable models to store detail regarding different model to be utilized for Regression problem
"""
class regressor_config:
    """
    Class variable model consist of a model type key with a list datatype value.
    where the list consist of model class object and dictionary of parameters specific to the model
    """
    models={
        "SVR":[
            svm.SVR,
            {
                "C":{"int":[1,5]},
                "gamma":{"str":['auto','scale']},
                "degree":{"int":[1,4]},
                "kernel":{"str":['rbf','poly','linear']}
            }  
        ],
        "NuSVR":[
            svm.NuSVR,
            {
                "nu":{'float':[0.1,1.0]},
                "gamma":{"str":['auto','scale']},
                "degree":{"int":[1,4]},
                "kernel":{"str":['rbf','poly','linear']}
            }
        ],
        "LinearSVR":[
            svm.LinearSVR,
            {
                "C":{"int":[1,5]},
                "epsilon":{'float':[1e-2,1]},
                "loss":{'str':['epsilon_insensitive', 'squared_epsilon_insensitive']},
                'tol':{'float':[1e-3,0.1]},
            }
        ],
        "DecisionTreeRegressor":[
            tree.DecisionTreeRegressor,
            {
                "criterion":{'str':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                "splitter":{'str':['random','best']},
                "max_features":{"str":["auto", "sqrt", "log2"]},
                "max_depth":{'int':[3,50]}
            }
        ],
        "RandomForestRegressor":[
            ensemble.RandomForestRegressor,
            {
                "criterion":{'str':['squared_error']},
                "n_estimators":{'int':[50,1000]},
                "max_features":{"str":["auto", "sqrt", "log2"]},
                "max_depth":{'int':[3,18]},
                'n_jobs':{'str':[-1]}
            }
        ],
        "ExtraTreesRegressor":[
            ensemble.ExtraTreesRegressor,
            {
                "criterion":{'str':['squared_error']},
                "n_estimators":{'int':[50,1000]},
                "max_features":{"str":["auto","sqrt", "log2"]},
                "max_depth":{'int':[3,18]},
                'n_jobs':{'str':[-1]}
            }
        ],
        "GradientBoostingRegressor":[
            ensemble.GradientBoostingRegressor,
            {
                "criterion":{'str':['squared_error','friedman_mse']},
                "n_estimators":{'int':[50,1000]},
                "max_features":{"str":["auto", "sqrt", "log2"]},
                "max_depth":{'int':[3,18]},
                "loss":{'str':['ls','lad','huber']},
                "learning_rate":{'float':[1e-3,0.1]}
            }
        ],
        "LinearRegression":[
            linear_model.LinearRegression,
            {
                'n_jobs':{'str':[-1]}
            }
        ],
        "Ridge":[
            linear_model.Ridge,
            {
                "alpha":{'float':[1e-3,0.1]},
                "solver":{'str':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']},
                "max_iter":{'int':[1000,5000]},
                'tol':{'float':[1e-3,0.1]},
            }
        ],
        "KNeighborsRegressor":[
            neighbors.KNeighborsRegressor,
            {
                "n_neighbors":{'int':[3,10]},
                "weights":{'str':['uniform','distance']},
                "algorithm":{'str':['auto', 'ball_tree', 'kd_tree', 'brute']},
                "p":{'int':[1,2]},
                "leaf_size":{'int':[10,50]}
            }
        ],
        "PoissonRegressor":[
            linear_model.PoissonRegressor,
            {
                "alpha":{'float':[1e-3,0.1]},
                "fit_intercept":{'bool':[True,False]},
                "max_iter":{'int':[1000,5000]},
                'tol':{'float':[1e-3,0.1]},
            }
        ],
        "SGDRegressor":[
            linear_model.SGDRegressor,
            {
                "loss":{"str":['huber','epsilon_insensitive','squared_epsilon_insensitive','squared_loss']},
                "penalty":{'str':['l2','l1','elasticnet']},
                "alpha":{'float':[1e-4,1.0]},
                "fit_intercept":{'bool':[True,False]},
                "max_iter":{'int':[500,2000]},
                "tol":{"float":[1e-3,0.1]},
                "shuffle":{'bool':[True,False]},
                "epsilon":{"float":[1e-1,1.5]},
                "learning_rate":{"str":['invscaling','constant','optimal','adaptive']},
                "early_stopping":{'bool':[True,False]}
            }
        ], 
        "ElasticNet" :[
            linear_model.ElasticNet,
            {
                "alpha":{'float':[1e-4,1.0]},
                "l1_ratio":{'float':[1e-4,1.0]},
                "max_iter":{'int':[500,2000]},
                "tol":{'float':[1e-4,1.0]},
                "selection":{'str':['cyclic', 'random']}
            }
        ]   ,   
        "AdaBoostRegressor":[
            ensemble.AdaBoostRegressor,
            {
                "n_estimators":{'int':[10, 5000]},
                "learning_rate":{'float':[1e-3,0.1]},
                "loss":{'str':['linear', 'square', 'exponential']}
            }
        ],
        "Lars":[
            linear_model.Lars,
            {
                "fit_intercept":{'bool':[True,False]},
                "n_nonzero_coefs":{"int":[500,2000]},
                "eps":{'float':[1e-6,1]},
            }
        ],     
        "Lasso":[
            linear_model.Lasso,
            {
                "alpha":{'float':[1e-4, 0.1]},
                "max_iter":{'int':[1000, 10000]},
                "tol":{'float':[1e-3, 0.1]},
                "selection":{'str':['cyclic', 'random']}
            }
        ],
        "BayesianRidge":[
            linear_model.BayesianRidge,
            {
                "alpha_1":{'float':[1e-3,0.1]},
                "alpha_2":{'float':[1e-3,0.1]},
                "lambda_1":{'float':[1e-3,0.1]},
                "lambda_2":{'float':[1e-3,0.1]}
            }
        ], 
        "LassoLars":[
            linear_model.LassoLars,
            {
                "alpha":{'float':[1e-4,0.1]},
                "fit_intercept":{'bool':[True,False]},
                "max_iter":{'int':[1000,10000]},
                "eps":{'float':[1e-6,0.1]}
            }
        ],
        "ARDRegression":[
            linear_model.ARDRegression,
            {
                "alpha_1":{'float':[1e-6,0.1]},
                "alpha_2":{'float':[1e-6,0.1]},
                "compute_score":{'bool':[True,False]},
                "lambda_1":{'float':[1e-3,0.1]},
                "lambda_2":{'float':[1e-3,0.1]},
                "n_iter":{'int':[300,10000]},
                "tol":{'float':[1e-3, 0.1]},
            }
        ],
        "XGBRegressor":[
            XGBRegressor,
            {
                "n_estimators":{'int':[100, 5000]},
                "max_depth":{'int':[3,16]},
                "learning_rate":{'float':[1e-3,0.1]},
                "booster":{'str':['gbtree', 'gblinear', 'dart']},
                'reg_alpha': {'float':[1e-3,0.1]},
                'reg_lambda': {'float':[1e-3,0.1]},
                'verbosity':{'str':[0]},
                'n_jobs':{'str':[1]}
            }
        ], 
        "GammaRegressor":[
            linear_model.GammaRegressor,
            {
                "alpha":{'float':[1e-4,1.0]},
                "max_iter":{'int':[1000,10000]},
                "tol":{'float':[1e-3,0.1]},
            }
        ],
        "RadiusNeighborsRegressor":[
            neighbors.RadiusNeighborsRegressor,
            {
                "radius":{'float':[1.0,10.0]},
                "weights":{'str':['uniform','distance']},
                "algorithm":{'str':['auto', 'ball_tree', 'kd_tree', 'brute']},
                "leaf_size":{'int':[10,50]},
                "p":{'int':[1,2]},
                "metric":{'str':['euclidean', 'manhattan', 'chebyshev', 'minkowski']}
            }
        ],
        "LGBMRegressor":[
            LGBMRegressor,
            {
                "boosting_type":{'str':['gbdt', 'dart','rf']},
                "num_leaves":{'int':[5, 30]},
                "learning_rate":{'float':[1e-4,1e-2]},
                "max_depth":{'int':[3,16]},
                "n_estimators":{'int':[100, 1000]},
                'bagging_fraction':{'float':[0.1,0.9]},
                'subsample_freq':{'int':[5,10]},
                'verbose':{'bool':[-1]},
            }
        ],
        "CatBoostRegressor":[
            CatBoostRegressor,
            {
                "learning_rate": {'float':[1e-3,1e-2]},
                "l2_leaf_reg":{'float':[1e-3, 1.0]},
                "loss_function":{'str': ["RMSE", "MultiRMSE"]},
                "iterations":{'int':[200, 1000]},
                "max_depth":{'int':[3,16]},
                "verbose":{'bool':[False]},
            }
        ],
        "PassiveAggressiveRegressor":[
            linear_model.PassiveAggressiveRegressor,
            {
                "max_iter":{'int':[1000,10000]},
                "epsilon":{'float':[1e-3,1]},
                "loss":{"str":['huber','epsilon_insensitive','squared_epsilon_insensitive','squared_loss']},
                "tol":{'float':[1e-3,0.1]}
            }
        ],
        "HuberRegressor":[
            linear_model.HuberRegressor, 
            {
                "epsilon":{'float':[1,1.5]},
                "max_iter":{'int':[100,10000]},
                "alpha":{'float':[1e-4,0.1]},
                "tol":{'float':[1e-3,0.1]} 
            }
        ],
        "HistGradientBoostingRegressor":[
            ensemble.HistGradientBoostingRegressor,
            {
                "loss":{'str':['least_squares', 'least_absolute_deviation', 'poisson']},
                "learning_rate":{'float':[1e-3,0.1]},
                "max_iter":{"int":[50,5000]},
                "max_depth":{'int':[3,18]},
                "tol":{'float':[1e-3,0.1]}
            }
        ],
        "OrthogonalMatchingPursuit":[
            linear_model.OrthogonalMatchingPursuit,
            {
                "tol":{"float":[1e-4,0.1]},
                "fit_intercept":{'bool':[True,False]},
                "n_nonzero_coefs":{'str':[0.1,0.15,0.2]}
            }
        ]
    }
