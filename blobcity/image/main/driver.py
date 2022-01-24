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

import os,copy
from blobcity.main import load as model_loader
from blobcity.main import spill as code_spill
from blobcity.aicloud import send_yaml_to_cloud
from blobcity.store import DictClass
from blobcity.main.modelSelection import model_search
from blobcity.utils import get_dataframe_type,check_subfolder_data,ProType
from blobcity.utils import uncompress_file,validate_url,file_from_url,AutoFeatureSelection

"""
This file consists of Function to deal with Image Classification problem in python.
"""
def train(file=None, target=None,model_types='classic',accuracy_criteria=0.99,resize=50):
    """
    param1: string: file path 

    param2: string: (optional)target/dependent column name.

    param3: string: whether to train on classic or tensorflow models options ['all','classic','neural']

    param4: float: range[0.1,1.0] 

    param5: int : resolution size to resize the image i.e resize X resize. for example, 50X50

    return: Model Class Object
    Performs a model search on the data proivded. A yaml file is generated once the best fit model configuration
    is discovered. The yaml file is later used for generating source code. 

    Input to the function must be one of file.
    """
    dict_class=DictClass()
    dict_class.resetVar()
    exp_id=ProType.generate_uuid()
    #data read
    if file!=None:
        dict_class.addKeyValue('problem',{'type':'Image Classification'})
        root, ext = os.path.splitext(file)
        compress_list=[".zip",".tar",".gz",'.tar.gz','.bz2']
        dict_class.addKeyValue('data_read',{'file':file})
        if not ext and ext not in compress_list and target==None:
            if validate_url(file): 
                file=file_from_url(file,dict_class) 
            else:
                dict_class.UpdateNestedKeyValue('data_read','from','Local')
            data,target=check_subfolder_data(file,dict_class)
        elif ext in compress_list:
            if validate_url(file): 
                file=file_from_url(file,dict_class) 
            else:
                dict_class.UpdateNestedKeyValue('data_read','from','Local')
            file=uncompress_file(file,dict_class)
            data,target=check_subfolder_data(file,dict_class)
        data=AutoFeatureSelection.image_processing(data,target,resize,dict_class)
            
        modelClass = model_search(dataframe=data,target='label',DictClass=dict_class,disable_colinearity=True,model_types=model_types,accuracy_criteria=accuracy_criteria,epochs=20,max_neural_search=10)
        modelClass.yamldata=dict_class.getdict()
        metrics=copy.deepcopy(modelClass.metrics)
        if modelClass.yamldata['model']['type'] in ['TF','tf','Tensorflow']:metrics['Accuracy']=dict_class.accuracy
        else:metrics['CVSCORE']=dict_class.accuracy
        send_yaml_to_cloud({'autoAIID':exp_id,'yaml':modelClass.yamldata,'metrics':metrics})
        dict_class.resetVar()
        return modelClass
    else:
        raise ValueError("{file} can't be null or empty")

def load(model_path=None):
    """
    param1: string : pickle file path 
    return: Class object 

    Function returns a de-serialized model file.
    """
    model=model_loader(model_path)
    return model

def spill(filepath=None,yaml_data=None,doc=None):
    """
    param1:string : filepath and format of generated file to store. either .py or .ipynb

    param2:string : filepath of already generated YAML file or dictionary object.
    
    param3:boolean : whether generate code along with documentation

    Function calls generator functions to generate source code for the AutoAI Procedure
    """
    code_spill(filepath=filepath,yaml_data=yaml_data,doc=doc)