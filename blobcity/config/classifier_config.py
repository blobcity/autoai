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
        
        "svc":[
            svm.SVC,
            {
                "C":{"int":[1,3]},
                "gamma":{"str":['auto','scale']},
                "degree":{"int":[1,3]},
                "kernel":{"str":['rbf','poly','linear','sigmoid']}
            }
        ],
        "nusvc":[
            svm.NuSVC, 
            {
                "nu":{'float':[0.1,1.0]},
                "gamma":{"str":['auto','scale']},
                "degree":{"int":[1,3]},
                "kernel":{"str":['rbf','poly','linear','sigmoid']}
            }
        ],
        "linearsvc":[
            svm.LinearSVC, 
            {
                "C":{"int":[1,3]},
                "loss":{'str':['hinge', 'squared_hinge']},
                'tol':{'float':[1e-3,0.1]},
                "penalty":{'str':['l2']}, 
            }
        ],
        "decisiontree":[
            tree.DecisionTreeClassifier,
            {
                "criterion":{'str':['gini','entropy']},
                "splitter":{'str':['random','best']},
                "max_features":{"str":["auto", "sqrt", "log2"]},
                "max_depth":{'int':[3,50]}
            }
        ],
        "randomforest":[
            ensemble.RandomForestClassifier,
            {
                "criterion":{'str':['gini','entropy']},
                "n_estimators":{'int':[100,1000]},
                "max_features":{"str":["auto", "sqrt", "log2"]},
                "max_depth":{'int':[3,50]}
            }
        ],
        "extratrees":[
            ensemble.ExtraTreesClassifier,
            {
                "criterion":{'str':['gini','entropy']},
                "n_estimators":{'int':[100,1000]},
                "max_features":{"str":["auto", "sqrt", "log2"]},
                "max_depth":{'int':[3,50]}
            }
        ],
        "gradientboosting":[
            ensemble.GradientBoostingClassifier,
            {
                "criterion":{'str':['mse','friedman_mse']},
                "n_estimators":{'int':[100,1000]},
                "max_features":{"str":["auto", "sqrt", "log2"]},
                "max_depth":{'int':[3,50]},
                "loss":{'str':['deviance','exponential']},
                "learning_rate":{'float':[1e-3,0.1]}
            }
        ],
        "logistic":[
            linear_model.LogisticRegression,
            {
                "penalty":{'str':['l1','l2','elasticnet']}, 
                'tol':{'float':[1e-3,0.1]},
                "C":{"int":[1,3]},
                "solver":{'str':['newton-cg','liblinear','lbfgs', 'sag', 'saga']},

            }
        ],
        "ridge":[
            linear_model.RidgeClassifier,
            {
                "alpha":{'float':[1e-3,0.1]},
                "solver":{'str':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']},
                "max_iter":{'int':[1000,5000]},
                'tol':{'float':[1e-3,0.1]},
            }
        ],
        "knn":[
            neighbors.KNeighborsClassifier,
            {
                "n_neighbors":{'int':[3,10]},
                "weights":{'str':['uniform','distance']},
                "algorithm":{'str':['auto', 'ball_tree', 'kd_tree', 'brute']},
                "p":{'int':[1,2]},
                "leaf_size":{'int':[10,50]}
            }
        ],
        "xgboost":[
            xgboost.XGBClassifier,
            {
                'max_depth': {'int':[3,50]},
                'n_estimators': {'int':[100,1000]},
                'learning_rate': {'float':[1e-3,0.1]},
                'reg_alpha': {'int':[1, 1.5]},
                'reg_lambda': {'int':[1, 1.5]},
                'booster':{'str':['gbtree', 'gblinear','dart']}
            }
        ],
         "bernoullinb":[
            naive_bayes.BernoulliNB,
            {
                "alpha":{'float':[1e-2,1.0]},
                "fit_prior":{'bool':[True,False]}
            }
        ],
        "histgradientboosting":[
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
        "adaboost":[
            ensemble.AdaBoostClassifier,
            {
                "algorithm":{"str":['SAMME','SAMME.R']},
                "n_estimators":{"int":[50,100]},
                "learning_rate": {'float':[1e-3,0.1]},
            }

    }