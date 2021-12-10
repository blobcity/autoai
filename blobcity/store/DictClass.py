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
This is a Custom Class to store Class reference data for YAML generation and process logging.
"""
class DictClass:

    def __init__(self,ObjectExist=False,ObjectList=None,YAML=dict(),feature_importance=dict(),original_label=dict(),accuracy=0.0):
        self.ObjectExist=ObjectExist
        self.ObjectList=ObjectList
        self.YAML=YAML
        self.feature_importance=feature_importance
        self.original_label=original_label
        self.accuracy=accuracy

    def get_encoded_label(self):
        """
        return: Dictionary
        Function returns dictionary of encoded label to orignal label mapping
        """
        return self.original_label
    def addKeyValue(self, key,value):
        """
        param1:Class reference/Class object 
        param2: String key
        param3: String value

        Function adds new key value pair into the class dictionary object
        """
        self.YAML[key]=value

    def getdict(self):
        """
        return : Dictionary 

        Function returns the complete dictionary in current state
        """
        return self.YAML

    def UpdateKeyValue(self,key,value):
        """
        param1:class reference
        param2: String key
        param2: String /Dicionary

        Function updates a simple Dictionary Key value if the key exists else creates the entry
        """
        if key in self.YAML.keys():
            self.YAML[key]=value
        else:
            DictClass.addKeyValue(self, key,value)

    def UpdateNestedKeyValue(self,key,key2,value):

        """
        param1:Class reference
        param2:String key
        param3:String key
        param4:String/Dictionary

        Function Updates a nested Dictionary Value if the key exists else creates an entry for the key
        """
        if key in self.YAML.keys():
            self.YAML[key][key2]=value
        else:
            self.YAML[key]={}
            self.YAML[key][key2]=value
            
    def resetVar(self):
        """
        Function to reset class variables
        """
        self.ObjectExist=False
        self.ObjectList=None
        self.YAML={}
        self.feature_importance={}
        self.original_label=dict()
    
