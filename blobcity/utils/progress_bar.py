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



from tqdm import tqdm, tqdm_notebook

class Progress:
    def __init__(self, trials=0):
        self.trials = trials
        self.pbar = None

    @staticmethod
    def is_notebook() -> bool:
        """
        Identifies if the code is running in a Jupyter Notebook or a terminal.
        """
        try:
            from IPython import get_ipython
            shell = get_ipython().__class__.__name__
            return shell in ('ZMQInteractiveShell', 'google.colab._shell')
        except NameError:
            return False

    def create_progressbar(self, n_counters: int, desc: str = ""):
        """
        Initializes a tqdm progress bar with appropriate handler for notebooks.
        """
        self.trials = n_counters
        if self.is_notebook():
            self.pbar = tqdm_notebook(total=n_counters, desc=desc, bar_format="{l_bar}{bar} [elapsed: {elapsed} < remaining:{remaining}]")
        else:
            self.pbar = tqdm(total=n_counters, desc=desc, bar_format="{l_bar}{bar} [elapsed: {elapsed} < remaining:{remaining}]")

    def update_progressbar(self, step: int = 1):
        """
        Updates the progress bar.
        """
        if self.pbar:
            self.pbar.update(step)
            self.trials -= step

    def close_progressbar(self):
        """
        Closes and resets the progress bar.
        """
        if self.pbar:
            self.pbar.close()
            self.pbar = None
        self.trials = 0

    