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
import sklearn as sk
import xgboost
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn import tree,ensemble,svm,linear_model,neighbors,naive_bayes

"""
This python file consist of Class variable models to store detail regarding different model to be utilized for classification problem
"""
class classifier_config:
    """
    Class variable model consist of a model type key with a list datatype value.
    where the list consist of model class object and dictionary of parameters specific to the model
    """
    
    models={
        "SVC":[
            svm.SVC,
            {
                "C":{"int":[1,3]},
                "gamma":{"str":['auto','scale']},
                "degree":{"int":[1,3]},
                "kernel":{"str":['rbf','poly','linear','sigmoid']}
            }
        ],
        "NuSVC":[
            svm.NuSVC, 
            {
                "nu":{'float':[0.1,1.0]},
                "gamma":{"str":['auto','scale']},
                "degree":{"int":[1,3]},
                "kernel":{"str":['rbf','poly','linear','sigmoid']}
            }
        ],
        "LinearSVC":[
            svm.LinearSVC, 
            {
                "C":{"int":[1,3]},
                "loss":{'str':['hinge', 'squared_hinge']},
                'tol':{'float':[1e-3,0.1]},
                "penalty":{'str':['l2']}, 
            }
        ],
        "DecisionTreeClassifier":[
            tree.DecisionTreeClassifier,
            {
                "criterion":{'str':['gini','entropy']},
                "splitter":{'str':['random','best']},
                "max_features":{"str":["auto", "sqrt", "log2"]},
                "max_depth":{'int':[3,50]}
            }
        ],
        "RandomForestClassifier":[
            ensemble.RandomForestClassifier,
            {
                "criterion":{'str':['gini','entropy']},
                "n_estimators":{'int':[100,1000]},
                "max_features":{"str":["auto", "sqrt", "log2"]},
                "max_depth":{'int':[3,50]},
                'n_jobs':{'str':[-1]}
            }
        ],
        "ExtraTreesClassifier":[
            ensemble.ExtraTreesClassifier,
            {
                "criterion":{'str':['gini','entropy']},
                "n_estimators":{'int':[100,1000]},
                "max_features":{"str":["auto", "sqrt", "log2"]},
                "max_depth":{'int':[3,50]},
                'n_jobs':{'str':[-1]}
            }
        ],
        "GradientBoostingClassifier":[
            ensemble.GradientBoostingClassifier,
            {
                "criterion":{'str':['squared_error','friedman_mse']},
                "n_estimators":{'int':[100,1000]},
                "max_features":{"str":["auto", "sqrt", "log2"]},
                "max_depth":{'int':[3,50]},
                "loss":{'str':['deviance','exponential']},
                "learning_rate":{'float':[1e-3,0.1]}
            }
        ],
        "LogisticRegression":[
            linear_model.LogisticRegression,
            {
                "penalty":{'str':['l1','l2','elasticnet']}, 
                'tol':{'float':[1e-3,0.1]},
                "C":{"int":[1,3]},
                "solver":{'str':['newton-cg','liblinear','lbfgs', 'sag', 'saga']},
                'n_jobs':{'str':[-1]}
            }
        ],
        "RidgeClassifier":[
            linear_model.RidgeClassifier,
            {
                "alpha":{'float':[1e-3,0.1]},
                "solver":{'str':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']},
                "max_iter":{'int':[1000,5000]},
                'tol':{'float':[1e-3,0.1]},
            }
        ],
        "KNeighborsClassifier":[
            neighbors.KNeighborsClassifier,
            {
                "n_neighbors":{'int':[3,10]},
                "weights":{'str':['uniform','distance']},
                "algorithm":{'str':['auto', 'ball_tree', 'kd_tree', 'brute']},
                "p":{'int':[1,2]},
                "leaf_size":{'int':[10,50]}
            }
        ],
        "XGBClassifier":[
            xgboost.XGBClassifier,
            {
                'use_label_encoder':{'bool':[False]},
                'max_depth': {'int':[3,50]},
                'n_estimators': {'int':[100,1000]},
                'learning_rate': {'float':[1e-3,0.1]},
                'reg_alpha': {'float':[1e-3,0.1]},
                'reg_lambda': {'float':[1e-3,0.1]},
                'booster':{'str':['gbtree', 'gblinear','dart']},
                'verbosity':{'str':[0]},
                'n_jobs':{'str':[-1]}
            }
        ],
        "RadiusNeighborsClassifier":[
            neighbors.RadiusNeighborsClassifier,
            {
                "radius":{'float':[1.0,10.0]},
                "weights":{'str':['uniform','distance']},
                "algorithm":{'str':['auto', 'ball_tree', 'kd_tree', 'brute']},
                "p":{'int':[1,2]},
                "metric":{'str':['euclidean', 'manhattan', 'chebyshev', 'minkowski']},
                "leaf_size":{'int':[10,50]},
                "outlier_label":{'str':['most_frequent']}
            }
        ],
        "BernoulliNB":[
            naive_bayes.BernoulliNB,
            {
                "alpha":{'float':[1e-2,1.0]},
                "fit_prior":{'bool':[True,False]}
            }
        ],
        "HistGradientBoostingClassifier":[
            ensemble.HistGradientBoostingClassifier,
            {
                "loss":{"str":['binary_crossentropy', 'categorical_crossentropy', 'auto']},
                "learning_rate":{'float':[1e-3,0.1]},
                "max_iter":{"int":[1000,5000]},
                "max_depth":{"int":[3,50]},
                "l2_regularization":{"float":[1e-3,0.1]},
                "tol":{"float":[1e-7,0.1]},
                "scoring":{"str":["accuracy", "precision", "loss"]},
            }
        ],
        "AdaBoostClassifier":[
            ensemble.AdaBoostClassifier,
            {
                "algorithm":{"str":['SAMME','SAMME.R']},
                "n_estimators":{"int":[50,100]},
                "learning_rate": {'float':[1e-3,0.1]},
            }
        ] ,
        "NearestCentroid":[
            neighbors.NearestCentroid,
            {
                "metric":{'str':['l1', 'l2', 'manhattan', 'euclidean']},
                "shrink_threshold":{'float':[1.0, 5.0]}
            }
        ],
        "SGDClassifier":[
            linear_model.SGDClassifier,
            {
                "loss":{"str":['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']},
                "penalty":{'str':['l1','l2','elasticnet']},
                "alpha":{'float':[1e-2,1.0]},
                "l1_ratio":{'float':[1e-3,0.1]},
                "tol":{"float":[1e-3,0.1]},
                "learning_rate":{'str':['optimal','constant','invscaling','adaptive']},
                "eta0":{'float':[0.0, 0.1]},
                "power_t":{'float':[0.01, 0.5]},
                "epsilon":{'float':[1e-8, 0.1]},
                'n_jobs':{'str':[-1]}
            }
        ], 
        "CategoricalNB":[
            naive_bayes.CategoricalNB,
            {
                "alpha":{'float':[1e-2,1.0]},
                "fit_prior":{'bool':[True,False]},
            }
        ],
        "MultinomialNB":[
            naive_bayes.MultinomialNB,
            {
                "alpha":{'float':[1e-2,1.0]},
                "fit_prior":{'bool':[True,False]},
            }
        ],
        "Perceptron":[
            linear_model.Perceptron,
            {
                "penalty":{"str":['l1','l2','elasticnet']},
                "alpha":{'float':[1e-4,1.0]},
                "l1_ratio":{'float':[1e-2,1.0]},
                "tol":{'float':[1e-3,0.1]},
                'n_jobs':{'str':[-1]}
            }
        ]
    }

