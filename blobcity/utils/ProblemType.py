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
This Python file consists of Function to identify the problem type either Classification or Regression Type.
"""
import os
import uuid
class ProType:

    def checkType(data):

        """
         param1: class
         param2: target data
         
         This function identify type of problem to be solved either regression or classification 
         on the basis of datatype and uniquiness in target columns

         Conditions:
         1. if datatype is Object/String return problem Type as classification
         2. else check if it is integer or float type with less then equal to 100 classes then return Classification 
            else return Regression as the ProblemType
        """
        if(data.dtype in ['object']): 
            return dict({'type':'Classification'})
        else:
            target_length=data.nunique(dropna=False)
            if data.dtype in ['int64','float64','int32','float32','int16','float16'] and target_length<=20:
                return dict({'type':'Classification'})
            else:
                return dict({'type':'Regression'})
    
    def generate_uuid():
        """
        return : string 
        Function generates Universal Unique identifier for each experiment executed by the train function
        """
        try:
            uid=str(uuid.uuid4())
            os.environ['EXPID']=uid
        except  Exception as e:
            print(e)
        return uid