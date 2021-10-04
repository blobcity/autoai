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
This Python file is test file consisting of core code functionality.
"""
import blobcity as bc
import pandas as pd
file_path,target="https://raw.githubusercontent.com/Thilakraj1998/Datasets_general/main/BreastCancer1.csv",'diagnosis'
features=['radius_mean','texture_mean','smoothness_mean','compactness_mean','concavity_mean']

df=pd.read_csv(file_path)
model=bc.train(df=df,target=target,features=features)

model.stats()
#print(model.predict([[1,1,1,1,1]]))
#print(model.features())
model.save()
