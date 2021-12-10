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
from blobcity.store import DictClass
from blobcity.utils import get_dataframe_type,check_subfolder_data
from blobcity.utils import uncompress_file,validate_url,file_from_url

"""
This file consists of Function to deal with Image Classification problem in python.
"""
def train(file=None, df=None, target=None,model_types='classic',accuracy_criteria=0.99):
    """
    param1: string: file path 

    param2: (optional) pandas.DataFrame object

    param3: string: (optional)target/dependent column name.

    param4: string: whether to train on classic or tensorflow models options ['all','classic','neural']

    param5: float: range[0.1,1.0] 

    return: Model Class Object
    Performs a model search on the data proivded. A yaml file is generated once the best fit model configuration
    is discovered. The yaml file is later used for generating source code. 

    Input to the function must be one of file or data frame (df). Passing both parameters of file and df in a single
    invocation is an incorrect use.
    """
    dict_class=DictClass()
    dict_class.resetVar()
    #data read
    if file!=None:
        root, ext = os.path.splitext(file)
        compress_list=[".zip",".tar",".gz",'.tar.gz','.bz2']
        if not ext and ext not in compress_list and target==None:
            if validate_url(file): file=file_from_url(file) 
            data,target=check_subfolder_data(file)
            return
        elif ext in compress_list:
            if validate_url(file): 
                file=file_from_url(file)
            file=uncompress_file(file)
            data,target=check_subfolder_data(file)
            return
    else: 
        dataframe = df
        dict_class.addKeyValue('data_read',{"type":"df","class":"df"})