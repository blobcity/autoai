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
Python file consist of Class Model to initialize/store and retrive data associated to trained machine learning model.
"""
class Model:
    params=dict()
    featureList=[]
    model=None
    def __init__(self):
        pass

    def predict(self,test):
        """
        param1: self
        param2: 2D Array
        return: List/Array

        Function returns List/Array for predicted value from the trained model.
        """
        result=self.model.predict(test)
        return result

    def parameters(self):
        """
        return: Dictionary

        funciton return dictionary consisting of tuned parameters value for the trained model.
        """
        return self.params

    def features(self):
        """
        return: List/Array

        function return List of feature used by model to train
        """
        return self.featureList