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

# Takes X_values as input, and returns the Y_value using a trained model
def predict(X_values):
    print('Predicting value')

# Gets the input features used by the model.
# This is useful when using automatic feature selection,
# and the user does not know which features were selected by the AutoAI engine
def features():
    return ['col1', 'col2']

# Saves the model to a pickle file
def save(folderLoc):
    print('Saving model + yml')

# Generates a py / ipynb file at the location specified. Path name must include .py or .ipynb.
# Correspondingly generate an appropriate file
def spill(fileLoc): 
    print('Generating code')