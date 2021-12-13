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



from tqdm import tqdm_notebook,tqdm
"""
Class to handle custom progress bar for model tuning process
"""
class Progress():

    def __init__(self,trials=0,pbar=None):
        self.trials=trials
        self.pbar=pbar

    def isnotebook(self):
        """
        return: boolean
        Function to identify type of python utilized either ipython or Python
        """
        try:
            from IPython import get_ipython
            shell = get_ipython().__class__.__name__
            
            if shell == 'ZMQInteractiveShell':
                return True  
            elif get_ipython().__class__.__module__ == "google.colab._shell":
                return True
            elif shell == 'TerminalInteractiveShell':
                return False  
            else:
                return False 
        except NameError:
            return False 

    def create_progressbar(self,n_counters,desc=""):
        """
        param1: integer : number of iteration in progress bar
        Function initializes a tqdm_notebook progress bar.
        """
        self.trials=n_counters
        if Progress().isnotebook():
            self.pbar=tqdm_notebook(total=n_counters,desc=desc, bar_format="{l_bar}{bar} [elapsed: {elapsed}< remaining:{remaining}]")
        else:
            self.pbar=tqdm(total=n_counters, desc=desc, bar_format="{l_bar}{bar} [elapsed: {elapsed}< remaining:{remaining}]")
    
    def update_progressbar(self,i):
        """
        param1: integer

        Function updates a tqdm_notebook progress bar with specified integer count.
        """
        self.trials=self.trials-1
        self.pbar.update(i)

    def close_progressbar(self):
        """
        Function close and reset required variable for progress bar
        """
        self.pbar.close()
        self.trials=50
        self.pbar=None
    