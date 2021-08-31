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


# Takes data frame, and target only. 
# Should do automatic feature selection
# Should return a trained model
# Should output progress, in an interactive manner while training is in progress
#
# In future other formats are required instead of DataFrame. Like CSV, Excel, Log files etc. 
def train(df, target):
    # this should internally create and a yml file. The yml file is used for generating the code in the future.
    # this should also store a pickle / tensorflow file based on the model used

    # perform a feature selection

    # train the model based on the selected features
    train(df, target, 'selected_features')

# Performs an automated model training. 
def train(df, target, X_values):
    print('Actually does the training')

# Reads a BlobCity published model file, and loads it into memory.
# This can be combination of yml and other related artifacts of a trained model
# Need to see if a h5 file of TensorFlow, and a pickel file for other models can be combined into say a .bcm file for storage
# .bcm would be a custom format, standing for a BlobCity Model
def load(modelFile):
    print('Loading model')


