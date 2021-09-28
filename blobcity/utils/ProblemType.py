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

import numpy as np
class ProType:

    def __init__(self):
        pass
    def checkType(self,data):

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
        if(data.dtype in ['object']): return dict({'type':'Classification'})
        else:
            target_length =len(np.unique(data))
            if data.dtype in ['int','float'] and target_length<=50: 
                return dict({'type':'Classification'})
            else: 
                return dict({'type':'Regression'})
