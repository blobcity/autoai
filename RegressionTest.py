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
This file runs a sample regression problem on the AutoAI framework. Can be used to check general functioning after adding Regression type models in the library.
"""
import blobcity as bc
import pandas as pd
file_path,target="./us_comm.csv",'Commission'
features=['month_sin', 'month_cos', 'Headcount', 'Adj Close', 'T1', 'T2']

model=bc.train(file=file_path,target=target,features=features) # function to test AutoAI Process

model.summary() # function to log summary about trained model

model.spill("./aicodegen.py") #function to test code generation 

model.generate_yaml() #function to test yaml generation

bc.spill("codefile.ipynb","./Process.yaml") #funciton to test yaml generation from specified yaml file

bc.spill("codefile.py","./Process.yaml") #funciton to test yaml generation from specified yaml file

model.plot_feature_importance()# function to plot feature_importance calculate using selectKbest functionality

model.plot_prediction() # function to plot true vs predicted values