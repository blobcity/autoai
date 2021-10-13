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
import nbformat as nbf
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

def write_ipynbcode(CGpath,nbs):
    """
    param1: string : path of file to generate with either .py or .ipynb extension.
    param2: notebook object 

    The function writes provided notebook data into the the .ipynb file.
    """

    with open(CGpath, 'w') as f:
        nbf.write(nbs, f)

def initialize(key):
    """
    param1: string 
    return: string
    The function initializes code string including problem type comment and generic import statements associated with the problem type.
    """
    codes=SourceCode.problem[key]+SourceCode.imports[key]
    return codes

def data_read(ymlData,codes="",nb=None,with_doc=False):
    """
    param1: dictionary : AutoAI steps data
    param2: string : Code syntaxs
    param3: boolean : Whether to includer documentation/meta description for the following section
    return: string : Code syntaxs

    The function adds code syntax related to data fetching using the panda's library.
    """
    if with_doc and nb!=None:
        nb['cells'].append(nbf.v4.new_markdown_cell(PyComments.procedure['datafetch'].replace("###","")))
    elif with_doc and codes!="":
        codes=codes+PyComments.procedure['datafetch']
    rtype=ymlData['data_read']['type']
    if(rtype!='df'): reader_code=SourceCode.data_read[rtype].replace("PATH", ymlData['data_read']['file'])
    else: reader_code=SourceCode.data_read[rtype]
    if nb!=None and codes=="":
        nb['cells'].append(nbf.v4.new_code_cell(reader_code))
        return nb
    else: return codes+reader_code

def features_selection(yml_data,codes="",nb=None,with_doc=False):
    """
    param1: dictionary : AutoAI steps data
    param2: string : Code syntaxs
    param3: boolean : Whether to includer documentation/meta description for the following section
    return: string : Code syntaxs

    The function adds code syntax related to feature selection using dataframe indexing.
    """
    if with_doc and nb!=None:
        nb['cells'].append(nbf.v4.new_markdown_cell(PyComments.procedure['x&y'].replace("###","")))
    elif with_doc and nb==None:
        codes=codes+PyComments.procedure['x&y']
    
    features=SourceCode.columns['features'].replace("FEATURES", str(yml_data['features']['X_values']))
    target=SourceCode.columns['target'].replace("TARGET", str(yml_data['features']['Y_values']))
    data=features+target+SourceCode.selections['X']+SourceCode.selections['Y']
    if nb!=None and codes=="":
        nb['cells'].append(nbf.v4.new_code_cell(data))
        return nb
    else:
        return codes+data

def cleaning(yml_data,codes="",nb=None,with_doc=False):
    """
    param1: dictionary : AutoAI steps data
    param2: string : Code syntaxs
    param3: boolean : Whether to includer documentation/meta description for the following section
    return: string : Code syntaxs

    The function adds code syntax related to the data preprocessing stages,
    such as missing value imputation and string encoding using the panda's and Sci-kit learn library.
    """
    
    if 'cleaning' in yml_data.keys():
        if 'missingValues' in yml_data['cleaning'].keys():
            if with_doc and nb!=None:
                nb['cells'].append(nbf.v4.new_markdown_cell(PyComments.procedure['missing'].replace("###","")))
            elif with_doc and codes!="":
                codes=codes+PyComments.procedure['missing']

            if nb!=None and codes=="":
                nb['cells'].append(nbf.v4.new_code_cell(SourceCode.cleaning['missingValues']))
            else:
                codes=codes+SourceCode.cleaning['missingValues']

        if 'encode' in yml_data['cleaning'].keys():
            if with_doc and nb!=None:
                nb['cells'].append(nbf.v4.new_markdown_cell(PyComments.procedure['encoding'].replace("###","")))
            elif with_doc and codes!="":
                codes=codes+PyComments.procedure['encoding']

            encode_code=""
            if 'X' in yml_data['cleaning']['encode'].keys():
                encode_code=encode_code+SourceCode.cleaning['encode']['X']
            if 'Y' in yml_data['cleaning']['encode'].keys():
                encode_code=encode_code+SourceCode.cleaning['encode']['Y']
            
            if nb!=None:
                nb['cells'].append(nbf.v4.new_code_cell(encode_code))
            else:
                codes=codes+encode_code

        return nb if nb!=None else codes
    else:
        return nb if nb!=None else codes


def splits(codes="",nb=None,with_doc=False):

    """
    param1:string:
    param2:notebook object:
    param3:boolean:
    return: string/notebook object
    """
    if with_doc: 
        if nb!=None: nb['cells'].append(nbf.v4.new_markdown_cell(PyComments.procedure['datasplit'].replace("###","")))
        else:codes=codes+PyComments.procedure['datasplit']
    if nb!=None and codes=="":
        nb['cells'].append(nbf.v4.new_code_cell(SourceCode.splits))
        return nb
    else:
        return codes+SourceCode.splits

def modeler(yml_data,key,with_doc,codes="",nb=None):
    
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
    
    if nb!=None:
        nb['cells'][0]['source']=nb['cells'][0]['source']+imports
        if with_doc:
            nb['cells'].append(nbf.v4.new_markdown_cell(metaDesc.replace("###","")))
        
        nb['cells'].append(nbf.v4.new_code_cell(param))
        nb['cells'].append(nbf.v4.new_code_cell(model))
        return nb
    elif codes!="":
        idx = codes.index("warnings.filterwarnings('ignore')")
        codes = codes[:idx]+imports+codes[idx:]
        if with_doc:
            codes=codes+"# "+metaDesc
        return codes+param+model

def model_metrics(key,codes="",nb=None,with_doc=False):
    """
    param1: dictionary : AutoAI steps data
    param2: string : Code syntaxs
    param3: boolean : Whether to includer documentation/meta description for the following section
    return: string : Code syntaxs

    The function adds code syntax related to the model evaluation/performance metrics based 
    on problem type, either classification or regression.
    """
    if with_doc: 
        if nb!=None: nb['cells'].append(nbf.v4.new_markdown_cell(PyComments.procedure['metrics'].replace("###","")))
        else: codes=codes+PyComments.procedure['metrics']
    if nb!=None and codes=="":
        nb['cells'].append(nbf.v4.new_code_cell(SourceCode.metric[key]))
        return nb
    else:
        return codes+SourceCode.metric[key]

def pycoder(yml_data,CGpath,doc=False):
    """
    param1: dictionary : AutoAI steps data
    param2: string : filepath to write the code.
    param3: boolean : Whether to includer documentation/meta description for the following section

    This is the driving function that is responsible for the sequential addition of code syntax into the PY file. 
    Once codes get generated according to the YAML file(AutoAI procedure), 
    the code is written into the file using the write_pycode function.
    """
    key=yml_data['problem']['type']
    codes=data_read(yml_data,codes=initialize(key),with_doc=doc)
    codes=features_selection(yml_data,codes=codes,with_doc=doc)
    codes=cleaning(yml_data,codes=codes,with_doc=doc)
    codes=splits(codes=codes,with_doc=doc)
    codes=modeler(yml_data,key,doc,codes=codes)
    codes=model_metrics(key,codes=codes,with_doc=doc)
    write_pycode(CGpath,codes)

def ipynbcoder(yml_data,CGpath,doc=True):
    """
    param1: dictionary : AutoAI steps data
    param2: string : filepath to write the code.
    param3: boolean : Whether to includer documentation/meta description for the following section

    This is the driving function that is responsible for the sequential addition of code syntax into the IPYNB file. 
    Once codes get generated according to the YAML file(AutoAI procedure), 
    the nobebook object is written into the file using the write_ipynbcode function.
    """
    nb = nbf.v4.new_notebook()
    key=yml_data['problem']['type']
    nb['cells'].append(nbf.v4.new_code_cell(initialize(key)))
    nb=data_read(yml_data,nb=nb,with_doc=doc)
    nb=features_selection(yml_data,nb=nb,with_doc=doc) 
    nb=cleaning(yml_data,nb=nb,with_doc=doc)
    nb=splits(nb=nb,with_doc=doc)
    nb=modeler(yml_data,key,doc,nb=nb) 
    nb=model_metrics(key,nb=nb,with_doc=doc)
    write_ipynbcode(CGpath,nb)

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
    elif ftype=="ipynb" and doc==None:
        ipynbcoder(data,CGpath,doc=True)
    elif ftype=="ipynb" and doc!=None:
        ipynbcoder(data,CGpath,doc)
    else:
        raise TypeError("file type must be .py or .ipynb")
