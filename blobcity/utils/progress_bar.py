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



from tqdm import tqdm_notebook
"""
Class to handle custom progress bar for model tuning process
"""
class Progress():
    pbar=None
    trials=0
    def __init__(self):
        self.trials=0
        self.pbar=None

    def create_progressbar(self,n_counters):
        """
        param1: integer : number of iteration in progress bar
        Function initializes a tqdm_notebook progress bar.
        """
        self.trials=n_counters
        self.pbar=tqdm_notebook(total=n_counters, desc="Model Tuning (Stage 3 of 3):", bar_format="{l_bar}{bar} [ time left: {remaining} ]")

    def update_progressbar(self,i):
        """
        param1: integer

        Function updates a tqdm_notebook progress bar with specified integer count.
        """
        self.pbar.update(i)

    def close_progressbar(self):
        """
        Function close and reset required variable for progress bar
        """
        self.pbar.close()
        self.trials=50
        self.pbar=None
    