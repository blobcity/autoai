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



"""
This python file consists of class  PyComments,  which has dictionary models and procedures utilized to add comments/meta description in Code generation.
"""
class PyComments:
    models={
        'Classification':{
            'LogisticRegression':"LogisticRegression model Descriptions\r\n",
            'RidgeClassifier':"RidgeClassifier model Descriptions\r\n",
            'SGDClassifier':"SGDClassifier model Descriptions\r\n",
            'ExtraTreesClassifier':"ExtraTreesClassifier model Descriptions\r\n",
            'RandomForestClassifier':"RandomForestClassifier model Descriptions\r\n",
            'AdaBoostClassifier':"AdaBoostClassifier model Descriptions\r\n",
            'GradientBoostingClassifier':"GradientBoostingClassifier model Descriptions\r\n",
            'HistGradientBoostingClassifier':"HistGradientBoostingClassifier model Descriptions\r\n",
            'SVC':"SVC model Descriptions\r\n",
            'NuSVC':"NuSVC model Descriptions\r\n",
            'LinearSVC':"LinearSVC model Descriptions\r\n",
            'DecisionTreeClassifier':"DecisionTreeClassifier model Descriptions\r\n",
            'KNeighborsClassifier':"KNeighborsClassifier model Descriptions\r\n",
            'RadiusNeighborsClassifier':"RadiusNeighborsClassifier model Descriptions\r\n",
            'MultinomialNB':'MultinomialNB model Descriptions\r\n',
            'CategoricalNB':'CategoricalNB model Descriptions\r\n',
            'XGBClassifier':'XGBClassifier model Descriptions\r\n',
            'NearestCentroid':'NearestCentroid model Descriptions\r\n',
            'Perceptron':'Perceptron model Descriptions\r\n'
        },
        'Regression':{
            'LinearRegression':"LinearRegression model Descriptions\r\n",
            'Ridge':"Ridge model Descriptions\r\n",
            'SGDRegressor':"SGDRegressor model Descriptions\r\n",
            'ExtraTreesRegressor':"ExtraTreesRegressor model Descriptions\r\n",
            'RandomForestRegressor':"RandomForestRegressor model Descriptions\r\n",
            'AdaBoostRegressor':"AdaBoostRegressor model Descriptions\r\n",
            'GradientBoostingRegressor':"GradientBoostingRegressor model Descriptions\r\n",
            'HistGradientBoostingRegressor':"HistGradientBoostingRegressor model Descriptions\r\n",
            'SVR':"SVR model Descriptions\r\n",
            'NuSVR':"NuSVR model Descriptions\r\n",
            'LinearSVR':"LinearSVR model Descriptions\r\n",
            'DecisionTreeRegressor':"DecisionTreeRegressor model Descriptions\r\n",
            'KNeighborsRegressor':"KNeighborsRegressor model Descriptions\r\n",
            'Lasso':"Lasso model Descriptions\r\n",
            'Lars':"Lars model Descriptions\r\n",
            'BayesianRidge':'BayesianRidge model Description\r\n',
            'LassoLars':'LassoLars model Descriptions\r\n',
            'XGBRegressor':'XGBRegressor model Descriptions\r\n',
            'ARDRegressor':'ARDRegressor model Descriptions\r\n',
            'CatBoostRegressor':'CatBoostRegressor model Descriptions\r\n',
            'GammaRegressor':'GammaRegressor model Description\r\n',
            'LGBMRegressor':'LGBMRegressor model Descriptions\r\n',
            'RadiusNeighborsRegressor':'RadiusNeighborsRegressor model Descriptions\r\n',
            'PassiveAggressiveRegressor':'PassiveAggressiveRegressor model Descriptions\r\n',
            'HuberRegressor':'HuberRegressor model Descriptions\r\n',
            'ElasticNet':'ElasticNet model Descriptions\r\n'
        }
    }

    procedure={
        'datafetch':"\n### Pandas is an open-source, BSD-licensed library providing high-performance,\n### easy-to-use data manipulation and data analysis tools.\n",
        'missing':"\n### Since the majority of the machine learning models in the Sklearn library doesn't handle string category data and Null value,\n### we have to explicitly remove or replace null values.\n### The below snippet have functions, which removes the null value if any exists.\n",
        'encoding':"\n### Converting the string classes data in the datasets\n### by encoding them to integer either using OneHotEncoding or LabelEncoding\n",
        'datasplit':"\n### The train-test split is a procedure for evaluating the performance of an algorithm.\n### The procedure involves taking a dataset and dividing it into two subsets.\n### The first subset is utilized to fit/train the model.\n### The second subset is used for prediction.\n### The main motive is to estimate the performance of the model on new data.\n",
        'metrics':"\n### Performance metrics are a part of every machine learning pipeline. \n### They tell you if you're making progress, and put a number on it. All machine learning models,\n### whether it's linear regression, or a SOTA technique like BERT, need a metric to judge performance.\n",
        'x&y':"\n### It is the process of reducing the number of input variables when developing a predictive model.\n### Used to reduce the number of input variables to reduce the computational cost of modelling and,\n### in some cases,to improve the performance of the model.\n"
    }