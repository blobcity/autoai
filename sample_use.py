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


# Here is a sample use of blobcity for representation purpose.
# THIS CODE IS NOT EXPECTED TO RUN

import blobcity as bc;

#df = pd.read_csv('test.csv')

# Will perform a training workload keeping the target column in mind. 
# Type of problem: Regression / Classification; is figured out automatically based on nature of the target column
# An automatic feature selection will be performed.
# An overloaded function is available to manually specify the columns to use, and thereby avoid feature selection

model = bc.train('file_path', 'target_column_name')
# Will get the features used by the model.
# An automatic feature selection could have been carried out

features = model.features()

# Use a trained model to predict values. 
# This should be new / unseen data for the model.
# The invocation will return the predicted y_value
prediction = model.predict(df[features])

# Use to generate source code for using the model.
# Where possible training code will also be generated.
model.spill('/test.ipynb'); # will generate an ipynb file with train & test code
model.spill('/test.py'); # will generate a py file with train & test code
# the above functions are also expected to output any required pickle / h5 files
