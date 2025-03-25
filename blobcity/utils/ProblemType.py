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
import uuid
import pandas as pd

class ProType:
    @staticmethod
    def check_type(data: pd.Series) -> dict:
        """
        Identifies the problem type (Regression or Classification) based on data type and uniqueness in the target column.
        
        Conditions:
        1. If data type is object/string -> Classification
        2. If data type is integer or float with <= 20 unique values -> Classification
        3. Otherwise -> Regression
        """
        if data.dtype == 'object':
            return {'type': 'Classification'}
        
        target_length = data.nunique(dropna=False)
        if data.dtype in ['int64', 'int32', 'int16'] and target_length <= 20:
            return {'type': 'Classification'}
        
        return {'type': 'Regression'}

    @staticmethod
    def generate_uuid() -> str:
        """
        Generates a Universal Unique Identifier (UUID) for each experiment.
        """
        try:
            uid = str(uuid.uuid4())
            os.environ['EXPID'] = uid
            return uid
        except Exception as e:
            print(f"Error generating UUID: {e}")
            return ""
