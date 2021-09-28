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
This python file Consist of Function to Generate YAML file from complete process record/log  for OPENSOURCE CODE GENERATION.
"""
import os
import yaml
def writeYml(val):
    """
    param1: dictionary

    Funciton take dictionary object and converts the data into YAML file format.
    """
    """
    with open(os.getcwd()+'\Process.yaml', 'w') as file:
        yaml.dump(val, file,sort_keys=False)
    """
    with open(r'./Process.yaml', 'w') as file:
        yaml.dump(val, file,sort_keys=False)