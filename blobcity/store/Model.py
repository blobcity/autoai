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
    metrics=dict()
    def __init__(self):
        self.params=dict()
        self.featureList=[]
        self.model=None
        self.metrics=dict()
        

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

        function return dictionary consisting of tuned parameters value for the trained model.
        """
        return self.params

    def features(self):
        """
        return: List/Array

        function return List of feature used by model to train
        """
        return self.featureList

    def save(self, path_pref='./'):
        """
        param: Path Prefix or Entire Path. Supported formats are .pkl and .h5. Default is .pkl
        returns: Final filepath of stored serialized file

        function saves the model and its weights serially and returns the filepath where it is saved.
        """
        path_components = path_pref.split('.')
        if len(path_components)<=2:
            extension = path_components[1]
        else:
            extension = path_components[2]

        if extension == '/':
            final_path = os.path.join(path_pref, 'my_model.pkl')
            pickle.dump(self.model, open(final_path, 'wb'))
            print("The model is stored at {}".format(final_path))
            return final_path

        elif extension == 'pkl':
            final_path = path_pref
            pickle.dump(self.model, open(final_path, 'wb'))
            print("The model is stored at {}".format(final_path))
            return final_path

        elif extension == 'h5':
            final_path = path_pref
            try:
                self.model.save(final_path)
                print("The model is stored at {}".format(final_path))
                return final_path
            except:
                raise TypeError("Your model is not a Keras model of type .h5. Try .pkl extension.")

        else:
            raise TypeError(f"{extension} file type must be .pkl or .h5")

    def load(self, filepath):
        """
        param: (required) the filepath to the stored model. Supports .h5 or .pkl models.
        returns: Model file

        function loads the serialized model from .pkl or .h5 format to usable format.
        """
        path_components = path_pref.split('.')
        if len(path_components)<=2:
            extension = path_components[1]
        else:
            extension = path_components[2]
        
        if extension == 'pkl':
            self.model = pickle.load(open(filepath, 'rb'))
        elif extension == 'h5':
            self.model = tf.keras.models.load_model(filepath)
        return self.model

    def stats(self):
        """
        function print/log/display all the metric associated with problem type for the selected trained model.
        """
        print ("{:<10} {:<10}".format('METRIC', 'VALUE'))
 
        # print each data item.
        for key, value in self.metrics.items():
            print ("{:<10} {:<10}".format(key, value))

