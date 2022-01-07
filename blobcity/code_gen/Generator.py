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
from blobcity.code_gen.IpynbMeta import IpynbComments
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

def initialize(key,codes="",nb=None):
    """
    param1: string 
    return: string
    The function initializes code string including problem type comment and generic import statements associated with the problem type.
    """
    if nb!=None:
        nb['cells'].append(nbf.v4.new_markdown_cell(SourceCode.problem[key]))
        nb['cells'].append(nbf.v4.new_code_cell(SourceCode.imports[key]))
        return nb
    else:
        codes=SourceCode.problem[key]+SourceCode.imports[key]
        return codes

def data_read(ymlData,codes="",nb=None,with_doc=False):
    """
    param1: dictionary : AutoAI steps data
    param2: string : Code syntaxs
    param3: boolean : Whether to includer documentation/meta description for the following section
    return: notebook object

    The function adds code syntax related to data fetching using the panda's library.
    """
    if ymlData['problem']['type'] in ['Classification','Regression']:
        if with_doc and nb!=None:
            nb['cells'].append(nbf.v4.new_markdown_cell(IpynbComments.procedure['datafetch']))
        elif with_doc and codes!="":
            codes=codes+PyComments.procedure['datafetch']
        rtype=ymlData['data_read']['type']
        if(rtype!='df'): reader_code=SourceCode.data_read[rtype].replace("PATH", ymlData['data_read']['file'])
        else: reader_code=SourceCode.data_read[rtype]
        if nb!=None and codes=="":
            nb['cells'].append(nbf.v4.new_code_cell(reader_code))
            return nb
        else: return codes+reader_code
    elif ymlData['problem']['type']=='Image Classification': 
        paths=SourceCode.image_data['paths'].replace('PATH',str(ymlData['data_read']['file'])).replace('TARGET',str(ymlData['features']['Y_values']))
        paths=paths.replace('file','PATHZIP')
        if 'Decompress' in ymlData['data_read'].keys():
            compression="\rfile='DECOMP'\r".replace('DECOMP',ymlData['data_read']['Decompressed_path'])
        
        if nb!=None and codes=="":
            nb['cells'].append(nbf.v4.new_markdown_cell("### Initialization"))
            nb['cells'].append(nbf.v4.new_code_cell(paths+compression))
            return nb
        else: return codes+paths+compression

def sample_imager(ymlData,codes="",nb=None,with_doc=False):
    sample_image=SourceCode.image_data['Sample Image']
    if nb!=None and codes=="":
        nb['cells'].append(nbf.v4.new_markdown_cell("### Sample Image"))
        nb['cells'].append(nbf.v4.new_code_cell(sample_image))
        return nb
    else: return codes+sample_image

def features_selection(yml_data,codes="",nb=None,with_doc=False):
    """
    param1: dictionary : AutoAI steps data
    param2: string : Code syntaxs
    param3: boolean : Whether to includer documentation/meta description for the following section
    return: notebook object

    The function adds code syntax related to feature selection using dataframe indexing.
    """
    if with_doc and nb!=None:
        nb['cells'].append(nbf.v4.new_markdown_cell(IpynbComments.procedure['x&y']))
    elif with_doc and nb==None:
        codes=codes+PyComments.procedure['x&y']
    
    if yml_data['problem']['type'] in ['Classification','Regression']:
        features=SourceCode.columns['features'].replace("FEATURES", str(yml_data['features']['X_values']))
        target=SourceCode.columns['target'].replace("TARGET", str(yml_data['features']['Y_values']))
        data=features+target+SourceCode.selections['X']+SourceCode.selections['Y']
        if nb!=None and codes=="":
            nb['cells'].append(nbf.v4.new_code_cell(data))
            return nb
        else:
            return codes+data
    elif yml_data['problem']['type']=='Image Classification': 
        data=SourceCode.image_data['features']
        if nb!=None and codes=="":
            nb['cells'].append(nbf.v4.new_code_cell(data))
            return nb
        else:
            return codes+"\r### Feature Selection"+data

def decompressing_code(yml_data,codes="",nb=None,with_doc=False):
    """
    param1: dictionary : AutoAI steps data
    param2: string : Code syntaxs
    param3: boolean : Whether to includer documentation/meta description for the following section
    return: string/notebook object
    """
    if 'Decompress' in yml_data['data_read'].keys():
        if yml_data['data_read']['Decompress']=='gz':
            import_statement=SourceCode.folder_decompression['gz']['import']
            code_syntax=SourceCode.folder_decompression['gz']['code'].replace('SAVEZIP','file')
        elif yml_data['data_read']['Decompress']=='zip':
            import_statement=SourceCode.folder_decompression['zip']['import']
            code_syntax=SourceCode.folder_decompression['zip']['code'].replace('SAVEZIP','file')
        
        if nb!=None and codes=="":
            nb['cells'][1]['source']=nb['cells'][1]['source']+import_statement
            nb['cells'].append(nbf.v4.new_markdown_cell("### Decompressing"))
            nb['cells'].append(nbf.v4.new_code_cell(code_syntax))
            return nb
        else:
            idx = codes.index("warnings.filterwarnings('ignore')")
            codes = codes[:idx]+import_statement+codes[idx:]
            return codes+code_syntax
    else:
        return nb if nb!=None else codes
        
def cleaning(yml_data,codes="",nb=None,with_doc=False):
    """
    param1: dictionary : AutoAI steps data
    param2: string : Code syntaxs
    param3: boolean : Whether to includer documentation/meta description for the following section
    return: string/notebook object

    The function adds code syntax related to the data preprocessing stages,
    such as missing value imputation and string encoding using the panda's and Sci-kit learn library.
    """
    
    if 'cleaning' in yml_data.keys():
        if 'missingValues' in yml_data['cleaning'].keys():
            if with_doc and nb!=None:
                nb['cells'].append(nbf.v4.new_markdown_cell(IpynbComments.procedure['missing']))
            elif with_doc and codes!="":
                codes=codes+PyComments.procedure['missing']

            if nb!=None and codes=="":
                nb['cells'].append(nbf.v4.new_code_cell(SourceCode.cleaning['missingValues']))
            else:
                codes=codes+SourceCode.cleaning['missingValues']

        if 'encode' in yml_data['cleaning'].keys():
            if with_doc and nb!=None:
                nb['cells'].append(nbf.v4.new_markdown_cell(IpynbComments.procedure['encoding']))
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

def cleaning_image(yml_data,codes="",nb=None,with_doc=False):
    # if with_doc and nb!=None:
    #     nb['cells'].append(nbf.v4.new_markdown_cell(IpynbComments.procedure['image_cleaning']))
    # elif with_doc and codes!="":
    #     codes=codes+PyComments.procedure['image_cleaning']
    clean=SourceCode.image_data['cleaning'].replace('SIZE',str(yml_data['cleaning']['resize']))
    if nb!=None and codes=="":
        nb['cells'].append(nbf.v4.new_code_cell(clean))
        return nb
    else:
        codes=codes+clean
        return codes

def add_corr_matrix(codes="",nb=None,with_doc=False):
    """
    param1:string:
    param2:notebook object:
    param3:boolean:
    return: string/notebook object
    """
    if with_doc: 
        if nb!=None: nb['cells'].append(nbf.v4.new_markdown_cell(IpynbComments.procedure['cor_matrix']))
        else:codes=codes+PyComments.procedure['cor_matrix']
    if nb!=None and codes=="":
        nb['cells'].append(nbf.v4.new_code_cell(SourceCode.cor_matrix))
        return nb
    else:
        return codes+"\n# Correlation Matrix\n"+SourceCode.cor_matrix

def data_scaling(yml_data,codes="",nb=None,with_doc=False):
    """
    param1:string
    param2:notebook object
    param3:boolean
    return: string/notebook object
    """
    if 'cleaning' in yml_data.keys():
        if 'rescale' in yml_data['cleaning'].keys():
            if yml_data['problem']['type'] in ['Classification','Regression']:
                rescale_type=yml_data['cleaning']['rescale']
                imports=SourceCode.cleaning['rescale_import'][rescale_type]
                if with_doc: 
                    if nb!=None: nb['cells'].append(nbf.v4.new_markdown_cell(IpynbComments.procedure['rescale']))
                    else:codes=codes+PyComments.procedure['rescale']
                if nb!=None and codes=="": 
                    nb['cells'][1]['source']=nb['cells'][1]['source']+imports
                    nb['cells'].append(nbf.v4.new_code_cell(SourceCode.cleaning['rescale'][rescale_type]))
                    return nb
                else:
                    idx = codes.index("warnings.filterwarnings('ignore')")
                    codes = codes[:idx]+imports+codes[idx:]
                    return codes+SourceCode.cleaning['rescale'][rescale_type]
            elif yml_data['problem']['type']=='Image Classification':
                rescale_type=yml_data['cleaning']['rescale']
                imports=SourceCode.cleaning['rescale_import'][rescale_type]
                rescaled=SourceCode.cleaning['rescale'][rescale_type].split('\r')[1]
                if with_doc: 
                    if nb!=None: nb['cells'].append(nbf.v4.new_markdown_cell(IpynbComments.procedure['rescale']))
                    else:codes=codes+PyComments.procedure['rescale']
                if nb!=None and codes=="": 
                    nb['cells'][1]['source']=nb['cells'][1]['source']+imports
                    nb['cells'].append(nbf.v4.new_code_cell(rescaled))
                    return nb
                else:
                    idx = codes.index("warnings.filterwarnings('ignore')")
                    codes = codes[:idx]+imports+codes[idx:]
                    return codes+rescaled
        else: return nb if nb!=None and codes=="" else codes
    else: return nb if nb!=None and codes=="" else codes
        

def splits(codes="",nb=None,with_doc=False):

    """
    param1:string:
    param2:notebook object:
    param3:boolean:
    return: string/notebook object
    """
    if with_doc: 
        if nb!=None: nb['cells'].append(nbf.v4.new_markdown_cell(IpynbComments.procedure['datasplit']))
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
    if key=='Image Classification': key='Classification'
    if yml_data['model']['type'] not in ['TF','tf','Tensorflow']:
        param=SourceCode.parameters.replace("PARAM", str(yml_data['model']['parameters']))
        model=SourceCode.models_init.replace("MODELNAME", str(yml_data['model']['type']))
    else:param,model="\n",SourceCode.tf_load
        
    imports,metaDesc=SourceCode.models[key][yml_data['model']['type']],PyComments.models[key][yml_data['model']['type']]
    
    if nb!=None:
        nb['cells'][1]['source']=nb['cells'][1]['source']+imports
        if with_doc:nb['cells'].append(nbf.v4.new_markdown_cell(IpynbComments.models[key][yml_data['model']['type']]))
        nb['cells'].append(nbf.v4.new_code_cell(param+model))
        return nb
    elif codes!="":
        idx = codes.index("warnings.filterwarnings('ignore')")
        codes = codes[:idx]+imports+codes[idx:]
        if with_doc:
            codes=codes+"# "+metaDesc
        return codes+param+model

def model_metrics(yml_data,key,codes="",nb=None,with_doc=False):
    """
    param1: dictionary : AutoAI steps data
    param2: string : Code syntaxs
    param3: boolean : Whether to includer documentation/meta description for the following section
    return: string : Code syntaxs

    The function adds code syntax related to the model evaluation/performance metrics based 
    on problem type, either classification or regression.
    """
    if key=='Image Classification': key='Classification'
    if with_doc: 
        if nb!=None: nb['cells'].append(nbf.v4.new_markdown_cell(IpynbComments.procedure['metrics']))
        else: codes=codes+PyComments.procedure['metrics']

    if nb!=None and codes=="":
        if yml_data['model']['type'] not in ['TF','tf','Tensorflow']:
            nb['cells'].append(nbf.v4.new_code_cell(SourceCode.metric[key]))
        else:
            if key == 'Classification':
                tf_metric_type=SourceCode.tf_metric[key]['binary'] if yml_data['model']['classification_type']=='binary' else SourceCode.tf_metric[key]['multi']
                nb['cells'].append(nbf.v4.new_code_cell(tf_metric_type))
            else:nb['cells'].append(nbf.v4.new_code_cell(SourceCode.tf_metric[key]))
        return nb
    else:
        if yml_data['model']['type'] not in ['TF','tf','Tensorflow']:
           return codes+SourceCode.metric[key]
        else:
            if key == 'Classification':
                tf_metric_type=SourceCode.tf_metric[key]['binary'] if yml_data['model']['classification_type']=='binary' else SourceCode.tf_metric[key]['multi']
                return codes+tf_metric_type
            else:
                return codes+SourceCode.tf_metric[key]

def image_predictor(yml_data,codes="",nb=None,with_doc=False):
    # if with_doc: 
    #     if nb!=None: nb['cells'].append(nbf.v4.new_markdown_cell(IpynbComments.procedure['metrics']))
    #     else: codes=codes+PyComments.procedure['metrics']
    if yml_data['model']['type'] not in ['TF','tf','Tensorflow']:
        pred=SourceCode.image_data['Image_prediction']['classic'].replace('SIZE',str(yml_data['cleaning']['resize']))
    else:
        pred=SourceCode.image_data['Image_prediction']['tf'].replace('SIZE',str(yml_data['cleaning']['resize']))
        
    if nb!=None and codes=="":
        nb['cells'].append(nbf.v4.new_code_cell(pred))
        return nb
    else:
       return codes+pred
        

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
    codes=initialize(key,codes="")
    codes=data_read(yml_data,codes=codes,with_doc=doc)
    codes=features_selection(yml_data,codes=codes,with_doc=doc)
    codes=cleaning(yml_data,codes=codes,with_doc=doc)
    codes=add_corr_matrix(codes=codes,with_doc=doc)
    codes=data_scaling(yml_data,codes=codes,with_doc=doc)
    codes=splits(codes=codes,with_doc=doc)
    codes=modeler(yml_data,key,doc,codes=codes)
    codes=model_metrics(yml_data,key,codes=codes,with_doc=doc)
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
    nb=initialize(key,nb=nb)
    nb=data_read(yml_data,nb=nb,with_doc=doc)
    nb=features_selection(yml_data,nb=nb,with_doc=doc) 
    nb=cleaning(yml_data,nb=nb,with_doc=doc)
    nb=add_corr_matrix(nb=nb,with_doc=doc)
    nb=data_scaling(yml_data,nb=nb,with_doc=doc)
    nb=splits(nb=nb,with_doc=doc)
    nb=modeler(yml_data,key,doc,nb=nb) 
    nb=model_metrics(yml_data,key,nb=nb,with_doc=doc)
    write_ipynbcode(CGpath,nb)

def pycoder_image(yml_data,CGpath,doc=False):
    """
    param1: dictionary : AutoAI steps data
    param2: string : filepath to write the code.
    param3: boolean : Whether to includer documentation/meta description for the following section

    This is the driving function that is responsible for the sequential addition of code syntax into the PY file. 
    Once codes get generated according to the YAML file(AutoAI procedure), 
    the code is written into the file using the write_pycode function.
    """
    key=yml_data['problem']['type']
    codes=initialize(key,codes="")
    codes=data_read(yml_data,codes=codes,with_doc=doc)
    codes=decompressing_code(yml_data,codes=codes,with_doc=doc)
    codes=sample_imager(yml_data,codes=codes,with_doc=doc)
    codes=cleaning_image(yml_data,codes=codes,with_doc=doc)
    codes=features_selection(yml_data,codes=codes,with_doc=doc)
    codes=data_scaling(yml_data,codes=codes,with_doc=doc)
    codes=splits(codes=codes,with_doc=doc)
    codes=modeler(yml_data,key,doc,codes=codes)
    codes=model_metrics(yml_data,key,codes=codes,with_doc=doc)
    codes=image_predictor(yml_data,codes=codes,with_doc=doc)
    write_pycode(CGpath,codes)

def ipynbcoder_image(yml_data,CGpath,doc=True):
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
    nb=initialize(key,nb=nb)
    nb=data_read(yml_data,nb=nb,with_doc=doc)
    nb=decompressing_code(yml_data,nb=nb,with_doc=doc)
    nb=sample_imager(yml_data,nb=nb,with_doc=doc)
    nb=cleaning_image(yml_data,nb=nb,with_doc=doc)
    nb=features_selection(yml_data,nb=nb,with_doc=doc) 
    nb=data_scaling(yml_data,nb=nb,with_doc=doc)
    nb=splits(nb=nb,with_doc=doc)
    nb=modeler(yml_data,key,doc,nb=nb) 
    nb=model_metrics(yml_data,key,nb=nb,with_doc=doc)
    nb=image_predictor(yml_data,nb=nb,with_doc=doc)
    write_ipynbcode(CGpath,nb)

def code_generator(data,filepath,doc=None):
    """
    param1: dictionary : AutoAI steps data
    param2: string : filepath to write the code.
    param3: boolean : Whether to includer documentation/meta description for the following section

    """
    if data['problem']['type'] in ['Classification','Regression']:
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
    elif data['problem']['type']=="Image Classification":
        code_generator_image(data,filepath,doc)


def code_generator_image(data,filepath,doc=None):
    """
    param1: dictionary : AutoAI steps data
    param2: string : filepath to write the code.
    param3: boolean : Whether to includer documentation/meta description for the following section

    """
    ftype = "py" if (filepath in ["",None]) else codegen_type(filepath)
    CGpath= f"CodeGen.{ftype}" if (filepath in ["",None]) else filepath
    if ftype=="py" and doc in [None,False]:
        pycoder_image(data,CGpath,doc=False)
    elif ftype=="py" and doc==True:
        pycoder_image(data,CGpath,doc)
    elif ftype=="ipynb" and doc==None:
        ipynbcoder_image(data,CGpath,doc=True)
    elif ftype=="ipynb" and doc!=None:
        ipynbcoder_image(data,CGpath,doc)
    else:
        raise TypeError("file type must be .py or .ipynb")