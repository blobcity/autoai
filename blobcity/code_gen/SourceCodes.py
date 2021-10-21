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
This python file consists of class  SourceCodes,  which has variables utilized to add source code in the generated file.
"""
class SourceCode:
    data_read={
        'csv':"# Data Fetch\r"+"file='PATH'\rdf=pd.read_csv(file)\rdf.head()\r\n",
        'xlsx':"# Data Fetch\r"+"file='PATH'\rdf=pd.read_xlsx(file)\rdf.head()\r\n",
        'df':"# Data Fetch\r"+"df='DATAFRAME_OBJECT'\rdf.head()\r\n"
    }

    problem={
        'Classification':'# Classification Problem\r\n',
        'Regression':'# Regression Problem\r\n'
    }

    imports={
        'Classification':"# imports\r"+'import numpy as np\rimport pandas as pd\rimport matplotlib.pyplot as plt\r'+
                'import seaborn as se\rimport warnings\rfrom sklearn.model_selection import train_test_split\r'+
                'from sklearn.preprocessing import LabelEncoder\r'+'from sklearn.metrics import classification_report,plot_confusion_matrix\r'
                +"warnings.filterwarnings('ignore')\r\n",
        'Regression':"# imports\r"+'import numpy as np\rimport pandas as pd\rimport matplotlib.pyplot as plt\r'+
                'import seaborn as se\rimport warnings\rfrom sklearn.model_selection import train_test_split\r'+
                'from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error\r'
                +"warnings.filterwarnings('ignore')\r\n",
    }

    columns={
        'features':"# Selected Columns\r"+'features=FEATURES\r',
        'target':"target='TARGET'\r\n"
    }

    selections={
        'X':"# X & Y\r"+"X=df[features]\r",
        'Y':"Y=df[target]\r\n"
    }

    cleaning={
        'missingValues':"# Data Cleaning\r"+"def NullClearner(value):\r\tif(isinstance(value, pd.Series) and (value.dtype in ['float64','int64'])):\r"+
            '\t\tvalue.fillna(value.mean(),inplace=True)\r\t\treturn value\r'+
            '\telif(isinstance(value, pd.Series)):\r'+
            '\t\tvalue.fillna(value.mode()[0],inplace=True)\r\t\treturn value\r'+
            '\telse:return value\r'+
            'x=X.columns.to_list()\r'+
            'for i in x:\r\tX[i]=NullClearner(X[i])\r'+
            'Y=NullClearner(Y)\r\n',
        'encode':{
            'X':"# Handling AlphaNumeric Features\r"+"X=pd.get_dummies(X)\r\n",
            'Y':"# Handling Target Encoding\r"+'def EncodeY(Y):\r'+
                "\tun_EncodedT=np.sort(pd.unique(Y), axis=-1, kind='mergesort')\r"+
                '\tY=LabelEncoder().fit_transform(Y)\r'+
                "\tEncodedT=[xi for xi in range(len(un_EncodedT))]\r"+
                "\tprint('Encoded Target: {} to {}'.format(un_EncodedT,EncodedT))\r"+
                "\treturn Y\r"+
                "Y=EncodeY(Y)\r\n"
        }
    }

    splits="# Data split for training and testing\r"+\
    "X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=123)\r\n"

    parameters="#Model Parameters\r"+"param=PARAM\r\n"

    models_init="# Model Initialization\r"+"model=MODELNAME(**param)\rmodel.fit(X_train,Y_train)\r\n"
    
    metric={
        'Classification':"# Confusion Matrix\rplot_confusion_matrix(model,X_test,Y_test)\r\n"+\
        '# Classification Report\rprint(classification_report(Y_test,model.predict(X_test)))\r\n',
        
        'Regression':"# Metrics\r\ny_pred=model.predict(X_test)\rprint('R2 Score: {:.2f}'.format(r2_score(Y_test,y_pred)))\r"+\
            "print('Mean Absolute Error {:.2f}'.format(mean_absolute_error(Y_test,y_pred)))\r"+\
            "print('Mean Squared Error {:.2f}'.format(mean_squared_error(Y_test,y_pred)))"
    }

    models={
        'Classification':{
            'LogisticRegression':"from sklearn.linear_model import LogisticRegression\r\n",
            'RidgeClassifier':"from sklearn.linear_model import RidgeClassifier\r\n",
            'SGDClassifier':"from sklearn.linear_model import SGDClassifier\r\n",
            'ExtraTreesClassifier':"from sklearn.ensemble import ExtraTreesClassifier\r\n",
            'RandomForestClassifier':"from sklearn.ensemble import RandomForestClassifier\r\n",
            'AdaBoostClassifier':"from sklearn.ensemble import AdaBoostClassifier\r\n",
            'GradientBoostingClassifier':"from sklearn.ensemble import GradientBoostingClassifier\r\n",
            'HistGradientBoostingClassifier':"from sklearn.experimental import enable_hist_gradient_boosting\r\nfrom sklearn.ensemble import HistGradientBoostingClassifier\r\n",
            'SVC':"from sklearn.svm import SVC\r\n",
            'NuSVC':"from sklearn.svm import NuSVC\r\n",
            'LinearSVC':"from sklearn.svm import LinearSVC\r\n",
            'DecisionTreeClassifier':"from sklearn.tree import DecisionTreeClassifier\r\n",
            'KNeighborsClassifier':"from sklearn.neighbors import KNeighborsClassifier\r\n",
            'CategoricalNB':"from sklearn.naive_bayes import CategoricalNB\r\n",
            'NearestCentroid':'from sklearn.neighbors import NearestCentroid\r\n',
            'RadiusNeighborsClassifier':'from sklearn.neighbors import RadiusNeighborsClassifier\r\n',
            'XGBClassifier':'from xgboost import XGBClassifier\r\n',
            'MultinomialNB':'from sklearn.naive_bayes import MultinomialNB\r\n',
            'Perceptron':'from sklearn.linear_model import Perceptron\r\n'
        },
        'Regression':{
            'LinearRegression':"from sklearn.linear_model import LinearRegression\r\n",
            'Ridge':"from sklearn.linear_model import Ridge\r\n",
            'SGDRegressor':"from sklearn.linear_model import SGDRegressor\r\n",
            'ExtraTreesRegressor':"from sklearn.ensemble import ExtraTreesRegressor\r\n",
            'RandomForestRegressor':"from sklearn.ensemble import RandomForestRegressor\r\n",
            'AdaBoostRegressor':"from sklearn.ensemble import AdaBoostRegressor\r\n",
            'GradientBoostingRegressor':"from sklearn.ensemble import GradientBoostingRegressor\r\n",
            'HistGradientBoostingRegressor':"from sklearn.experimental import enable_hist_gradient_boosting\r\nfrom sklearn.ensemble import HistGradientBoostingRegressor\r\n",
            'SVR':"from sklearn.svm import SVR\r\n",
            'NuSVR':"from sklearn.svm import NuSVR\r\n",
            'LinearSVR':"from sklearn.svm import LinearSVR\r\n",
            'DecisionTreeRegressor':"from sklearn.tree import DecisionTreeRegressor\r\n",
            'KNeighborsRegressor':"from sklearn.neighbors import KNeighborsRegressor\r\n",
            'Lasso':"from sklearn.linear_model import Lasso\r\n",
            'Lars':"from sklearn.linear_model import Lars\r\n",
            'XGBRegressor':'from xgboost import XGBRegressor\r\n',
            'BayesianRidge':'from sklearn.linear_model import BayesianRidge\r\n',
            'LassoLars':'from sklearn.linear_model import LassoLars\r\n',
            'ARDRegressor':'from sklearn.linear_model import ARDRegressor\r\n',
            'CatBoostRegressor':'from catboost import CatBoostRegressor\r\n',
            'GammaRegressor':'from sklearn.linear_model import GammaRegressor\r\n',
            'LGBMRegressor':'from lightgbm import LGBMRegressor\r\n',
            'RadiusNeighborsRegressor':'from sklearn.neighbors import RadiusNeighborsRegressor\r\n',
            'PassiveAggressiveRegressor':'from sklearn.linear_model import PassiveAggressiveRegressor\r\n',
            'HuberRegressor':'from sklearn.linear_model import HuberRegressor\r\n',
            'ElasticNet':'from sklearn.linear_model import ElasticNet\r\n',
        }
    }