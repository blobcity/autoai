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
from blobcity.main import load as model_loader
from blobcity.main import spill as code_spill
from blobcity.aicloud import send_yaml_to_cloud
from blobcity.store import DictClass
from blobcity.main.modelSelection import image_gan
from blobcity.utils import ProType,gan_image_proccessing
from blobcity.config import DCGAN

"""
This file consists of Function to deal with Image GAN  problem in python.
"""
def train(file=None,resolution_factor=3,epochs=10,batch_size=32):
    """
    param1: string: file path 

    param2: int: (optional)resolution factor to resize image (1=32px, 2=64px, 3=96px, 4=128px, etc.).

    param3: int: Number of training epochs

    param4: int : batch size for training model.
    return: Model Class Object
    Performs a model search on the data proivded. A yaml file is generated once the best fit model configuration
    is discovered. The yaml file is later used for generating source code. 

    Input to the function must be one of file.
    """
    dict_class=DictClass()
    dict_class.resetVar()
    exp_id=ProType.generate_uuid()
    dcgan= DCGAN(GENERATE_RES=resolution_factor,BATCH_SIZE=batch_size)
    #data read
    if file!=None:
        dict_class.addKeyValue('problem',{'type':'Image GAN'})
        root, ext = os.path.splitext(file)
        compress_list=[".zip",".tar",".gz",'.tar.gz','.bz2']
        dict_class.addKeyValue('data_read',{'file':file})
        data=gan_image_proccessing(file,dcgan)
        modelclass=image_gan(data,dcgan,dict_class,epochs)
        modelclass.yamldata=dict_class.getdict()
        # modelClass = model_search(dataframe=data,target='label',DictClass=dict_class,disable_colinearity=True,model_types=model_types,accuracy_criteria=accuracy_criteria,epochs=epochs,max_neural_search=max_neural_search)
        # modelClass.yamldata=dict_class.getdict()
        # metrics=copy.deepcopy(modelClass.metrics)
        # if modelClass.yamldata['model']['type'] in ['TF','tf','Tensorflow']:metrics['Accuracy']=dict_class.accuracy
        # else:metrics['CVSCORE']=dict_class.accuracy
        # send_yaml_to_cloud({'autoAIID':exp_id,'yaml':modelClass.yamldata,'metrics':metrics})
        dict_class.resetVar()
        return modelclass
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