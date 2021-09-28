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


# Takes data frame, and target only. 
# Should do automatic feature selection
# Should return a trained model
# Should output progress, in an interactive manner while training is in progress
#
# In future other formats are required instead of DataFrame. Like CSV, Excel, Log files etc. 
from blobcity.store import DictClass
from blobcity.utils import getDataFrameType
from blobcity.utils import dataCleaner
from blobcity.utils import AutoFeatureSelection as AFS
from blobcity.utils import writeYml
from blobcity.main.modelSelection import modelSearch
def train(file_path=None, target=None,features=None):
    # this should internally create and a yml file. The yml file is used for generating the code in the future.
    # this should also store a pickle / tensorflow file based on the model used
    # Data read
    #below function read tabular/Structured/Semi-Structured data based on file type and returns dataframe object.
    dc=DictClass()
    dc.resetVar()
    #data read
    dataframe=getDataFrameType(file_path,dc)
    if(features==None):
        featureList=AFS.FeatureSelection(dataframe,target,dc)
        CleanedDF=dataCleaner(dataframe,featureList,target,dc)
    else:
        CleanedDF=dataCleaner(dataframe,features,target,dc)
    #model search space
    modelClass = modelSearch(CleanedDF,target,dc)

    #YML generation
    writeYml(dc.getdict())
    #return modelClass object
    dc.resetVar()
    return modelClass
# Performs an automated model training. 


# Reads a BlobCity published model file, and loads it into memory.
# This can be combination of yml and other related artifacts of a trained model
# Need to see if a h5 file of TensorFlow, and a pickel file for other models can be combined into say a .bcm file for storage
# .bcm would be a custom format, standing for a BlobCity Model
def load(modelFile):
    print('Loading model')


