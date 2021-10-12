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
This file consists of functions to deal with the Code generation functionality of AutoAI.
"""
import yaml
import os
from blobcity.code_gen.PyMeta import PyComments
from blobcity.code_gen.SourceCodes import SourceCode
def yml_reader(ymlpath):
    """
    param1: string : Path of YAML file
    return: dictionary
    The function reads provided YAML file and converts it to a python dictionary.
    If the YAML file path is specified, read from it, else read from the default file and return a dictionary.
    """
    if ymlpath==None:
        return yaml.load(open('./Process.yaml', 'r'),Loader=yaml.FullLoader)
    else:
        return yaml.load(open(ymlpath, 'r'),Loader=yaml.FullLoader)


def codegen_type(filepath):
    """
    param1: string : path of file to generate with either .py or .ipynb extension.
    return: string 
    The function identifies which type of file to generate and return type.
    """
    if(filepath=="" or None):
        ftype='py'
    else:
        extension = os.path.splitext(filepath)[1]
        ftype= "py" if extension==".py" else "ipynb"
    return ftype

def write_pycode(CGpath,codes):
    """
    param1: string : path of file to generate with either .py or .ipynb extension.
    param2: string : Actual code strings to be written into the file.

    The function writes provide code string into the the .py file.
    """
    with open(CGpath,'w') as f:
        f.write(codes)

def initialize(key):
    """
    param1: string 
    return: string
    The function initializes code string including problem type comment and generic import statements associated with the problem type.
    """
    codes=SourceCode.problem[key]+SourceCode.imports[key]
    return codes

def data_read(ymlData,codes,with_doc):
    """
    param1: dictionary : AutoAI steps data
    param2: string : Code syntaxs
    param3: boolean : Whether to includer documentation/meta description for the following section
    return: string : Code syntaxs

    The function adds code syntax related to data fetching using the panda's library.
    """
    if with_doc:
        codes=codes+PyComments.procedure['datafetch']
    rtype=ymlData['data_read']['type']
    if(rtype!='df'):
        reader_code=SourceCode.data_read[rtype].replace("PATH", ymlData['data_read']['file'])
    else:
        reader_code=SourceCode.data_read[rtype]
    codes=codes+reader_code
    return codes

def features_selection(yml_data,codes,with_doc):
    """
    param1: dictionary : AutoAI steps data
    param2: string : Code syntaxs
    param3: boolean : Whether to includer documentation/meta description for the following section
    return: string : Code syntaxs

    The function adds code syntax related to feature selection using dataframe indexing.
    """
    if with_doc:
        codes=codes+PyComments.procedure['x&y']
    features=SourceCode.columns['features'].replace("FEATURES", str(yml_data['features']['X_values']))
    target=SourceCode.columns['target'].replace("TARGET", str(yml_data['features']['Y_values']))
    codes=codes+features+target+SourceCode.selections['X']+SourceCode.selections['Y']
    return codes

def cleaning(yml_data,codes,with_doc):
    """
    param1: dictionary : AutoAI steps data
    param2: string : Code syntaxs
    param3: boolean : Whether to includer documentation/meta description for the following section
    return: string : Code syntaxs

    The function adds code syntax related to the data preprocessing stages,
    such as missing value imputation and string encoding using the panda's and Sci-kit learn library.
    """
    if 'cleaning' in yml_data.keys():
        cleankeys=yml_data['cleaning'].keys()
        if 'missingValues' in cleankeys:
            if with_doc:
                codes=codes+PyComments.procedure['missing']
            codes=codes+SourceCode.cleaning['missingValues']
        if 'encode' in cleankeys:
            if with_doc:
                codes=codes+PyComments.procedure['encoding']
            encodekeys = yml_data['cleaning']['encode'].keys()
            if 'X' in encodekeys:
                codes=codes+SourceCode.cleaning['encode']['X']
            if 'Y' in encodekeys:
                codes=codes+SourceCode.cleaning['encode']['Y']
    return codes

def splits(codes,with_doc):
    """
    param1: string : Code syntaxs
    param2: boolean : Whether to includer documentation/meta description for the following section
    return: string : Code syntaxs

    The function adds code syntax related to the train and test splits.
    """
    if with_doc:
        codes=codes+PyComments.procedure['datasplit']

    return codes+SourceCode.splits

def modeler(yml_data,codes,key,with_doc):
    """
    param1: dictionary : AutoAI steps data
    param2: string : Code syntaxs
    param3: boolean : Whether to includer documentation/meta description for the following section
    return: string : Code syntaxs

    The function adds code syntax related to the Machine learning model initialization and training.
    """
    param=SourceCode.parameters.replace("PARAM", str(yml_data['model']['parameters']))
    model=SourceCode.models_init.replace("MODELNAME", str(yml_data['model']['type']))
    imports,metaDesc=SourceCode.models[key][yml_data['model']['type']],PyComments.models[key][yml_data['model']['type']]
    idx = codes.index("warnings.filterwarnings('ignore')")
    codes = codes[:idx]+imports+codes[idx:]
    if with_doc:
        codes=codes+"# "+metaDesc
        
    return codes+param+model

def model_metrics(key,codes,with_doc):
    """
    param1: dictionary : AutoAI steps data
    param2: string : Code syntaxs
    param3: boolean : Whether to includer documentation/meta description for the following section
    return: string : Code syntaxs

    The function adds code syntax related to the model evaluation/performance metrics based 
    on problem type, either classification or regression.
    """
    if with_doc:
        codes=codes+PyComments.procedure['metrics']

    return codes+SourceCode.metric[key]

def pycoder(yml_data,CGpath,doc):
    """
    param1: dictionary : AutoAI steps data
    param2: string : filepath to write the code.
    param3: boolean : Whether to includer documentation/meta description for the following section

    This is the driving function that is responsible for the sequential addition of code syntax into the file. 
    Once codes get generated according to the YAML file(AutoAI procedure), 
    the code is written into the file using the write_pycode function.
    """
    key=yml_data['problem']['type']
    codes=data_read(yml_data,initialize(key),doc)
    codes=features_selection(yml_data,codes,doc)
    codes=cleaning(yml_data,codes,doc)
    codes=splits(codes,doc)
    codes=modeler(yml_data,codes,key,doc)
    codes=model_metrics(key,codes,doc)
    write_pycode(CGpath,codes)

def code_generator(data,filepath,doc=None):
    """
    param1: dictionary : AutoAI steps data
    param2: string : filepath to write the code.
    param3: boolean : Whether to includer documentation/meta description for the following section

    """
    ftype = "py" if (filepath in ["",None]) else codegen_type(filepath)
    CGpath= f"CodeGen.{ftype}" if (filepath in ["",None]) else filepath
    if ftype=="py" and doc in [None,False]:
        pycoder(data,CGpath,doc=False)
    elif ftype=="py" and doc==True:
        pycoder(data,CGpath,doc)
    else:
        raise TypeError("file type must be .py or .ipynb")
