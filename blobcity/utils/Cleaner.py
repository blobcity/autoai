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
This Python file consists of function to perform basic data cleaning/data preprocessing operation on most dataset.
Functions includes, Removal of Unique COlumns,High Null value ratio, Missing Value Handling, String Categorical feature Handling .
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from blobcity.utils.ProblemType import ProType
import warnings
warnings.filterwarnings("ignore")

def dataCleaner(df,features,target,DictionaryClass):
    """
    Funciton to check null occurances and handles other functions.

    param1: pandas DataFrame
    param2: target column name
    param3: Dictionary Class object

    return: pandas dataframe

    working:
    First the function identifies the problem type that is either regression or classification using ProType Class and its function checkType.
    For the Complete dataframe if any rows has more then 50% null values or missing values we will drop it.
    For the Complete dataframe if any columns has more then equal to 80% null missing values we will drop it to avoid any noise or sequed data imputation.
    Then check whether the dataframe has any null values. 
    if TRUE then : get all the columns names with null/missing values.
    for each columns with missing data call Cleaner function to Handle missing value with appropriate missing value handling strategy.
    Once Missing values are handled perform categorical value handling operation by calling Encoder() function with appropriate arguments
    """
    problemtype=ProType()
    missingdict=dict()
    DictionaryClass.addKeyValue('problem',problemtype.checkType(df[target]))
    
    updateddf=df[features].copy(deep=True)
    updateddf[target]=df[target].copy(deep=True)
    updateddf=RemoveRowsWithHighNans(updateddf)
    updateddf=RemoveHighNullValues(updateddf)
    updateddf=dropUniqueColumn(updateddf,target)
    if updateddf.isnull().values.any(): 
        cols=updateddf.columns[updateddf.isnull().any()].tolist()
        for i in cols:
            Cleaner(updateddf,i,missingdict)
        DictionaryClass.addKeyValue('cleaning',{'missingValues':missingdict})

    X_values,Y_value=updateddf.drop(target,axis=1),updateddf[target]

    EncoderResult=Encoder(DictionaryClass,X_values,Y_value,target)

    DictionaryClass.addKeyValue('features',{'X_values':X_values.columns.to_list(),'Y_values':target})

    return EncoderResult

def dropUniqueColumn(X_values,target):
    """
        param1: pandas.DataFrame 
        return : pandas.DataFrame

        Function Drop Column with Complete Unique data for example data such as ID,UniqueID etc.
        for all available feature in the dataframe it checks whether the column has unique value counts equal to number of entries
        in the dataset. and drops if exists and return dataframe once dropped.
    """    
    row_counts = len(X_values)
    for i in X_values.columns.to_list():
        if len(X_values[i].unique())==row_counts and i!=target:
            X_values.drop(i, axis=1, inplace=True)
    return X_values

def RemoveHighNullValues(dataframe):
    """
    param1: pandas.DataFrame
    return: pandas.DataFrame

    Function drops any feature with more then 80% of Null Values and return the Dataframe
    """
    thresh = len(dataframe) * .2
    dataframe.dropna(thresh = thresh, axis = 1, inplace = True)
    return dataframe

def Cleaner(df,i,missingdict):
    """
    param1: pandas DataFrame
    param2: column name
    param3: placeholder dictionary for YAML record

    Working:
     This funciton Handles missing values in the dataset by considering the datatype of each passed column to the function. 
     if the columns datatype is object to uses mode imputation.
     while if the datatype is integer ir float, and has 3 or less unique values in it mode imputation is used,else mean imputation
    
    """
    if(df[i].dtype in ["float","int"]):
        if(len(np.unique(df[i]))<=3):
            df[i].fillna(df[i].mode()[0],inplace=True)
            missingdict[i]="mode"
        else:
            df[i].fillna(df[i].mean(),inplace=True)   
            missingdict[i]="mean"
    elif(df[i].dtype=="object"):
        df[i].fillna(df[i].mode()[0],inplace=True)
        missingdict[i]="mode"

def Encoder(DictionaryClass,X,Y,target):
    """
    Function to Encode categorical data from the feature set and target sets.
    
    param1: X_values pd.dataframe type
    param2: y_values pd.series or numpy.ndarray

    return: DataFrame 

    working:
        if the feature set(X_values) has any string categorical data(object/string/category) 
        then: apply one hot encoding to the feature 
        and 
        if the target (Y_values) is a categorical data (object/category) 
        then : Apply Label Encoding Technique to the target columns

        finally return both the feature and target .
    """
    encode=dict()
    if("object" in X.dtypes.to_list() or Y.dtype=="object"):
        if("object" in X.dtypes.to_list()):
            objectTypes(X,DictionaryClass)
            X=pd.get_dummies(X)
            encode['X']='OneHotEncode' 
        if(Y.dtype=="object" ):
            encode['Y']='LabelEncoder' 
            Y=LabelEncoder().fit_transform(Y)
        dataframe=X.copy(deep=True)
        dataframe[target]=Y
        DictionaryClass.UpdateNestedKeyValue('cleaning','encode',encode)
        return dataframe
    else:
        dataframe=X.copy(deep=True)
        dataframe[target]=Y
        return dataframe

def objectTypes(X,DictionaryClass):
    """
    param1: pandas.dataframe
    param2: class object

    Function indentifies existence of String Categorical features.
    If String Categorical Feature exist record the list of features with string data in Class List Variable,
    and set boolean flag for existence to True else False
    """

    g = X.columns.to_series().groupby(X.dtypes).groups
    gd={k.name: v for k, v in g.items()}
    if 'object' in gd.keys():
        DictionaryClass.ObjectExist=True
        DictionaryClass.ObjectList= gd['object'].to_list()  
    else:
        DictionaryClass.ObjectExist= False

def RemoveRowsWithHighNans(dataframe):
    """
    param1: pandas.DataFrame
    return: pandas.DataFrame

    Function delete rows containing more than 50% NaN Values
    """
    percent = 50.0
    min_count = int(((100-percent)/100)*dataframe.shape[1] + 1)
    dataframe = dataframe.dropna( axis=0, 
                    thresh=min_count)
    return dataframe
