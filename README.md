# BlobCity AutoAI
A framework to find, train and generate code for the best performing AI model. Works on Classification and Regression problems.

Framework is currently designed for tabluar data, and is being extended to support images, videos and natural language. 

# Getting Started
``` shell
pip install blobcity
```

``` Python
import blobcity as bc
bc.train(file="data.csv", target="Y_column")
bc.spill("my_code.ipynb")
```
`Y_column` is the name of the target column. The column must be present within the data provided. 

Automatic inference of Regression / Classification is supported by the framework.

Supported input data formats are `.csv` and `.xlsx`. Extension for other file formats is being worked on. 

The `spill` function generates the model code with exhaustive documentation. Training code is also included for basic scikit-learn models. TensorFlow and other DNN models produce only the test / final use code. 

## Use a Pandas Data Frame
``` Python
bc.train(df=my_df, target="Y_column")
```

If loading data from a Database or external system, create a DataFrame from your data source, and pass it directly to the `train` function.

## From a URL
``` Python
bc.train(file="https://example.com/data.csv", target="Y_column")
```

The `file` parameter can be a local file, or a URL. The function will load data from the specified URL. The file at the URL must be either in CSV or XLSX format. The URL should be accessible publicly without authentication. 

# Code Generation
Multiple formats of code generation is supported by the framework. The `spill` function can be used to generate both `ipynb` and `py` files. The desired type is infered from the name of the output file. The code file will be created at the path specified. Relative and absolute file paths are both supported. 

### Generate Jupyter Notebook
``` Python
bc.spill("my_code.ipynb");
```
Generates an ipynb file with full markdown documentation explain the code.

### Generate Python Code
``` Python
bc.spill("my_code.py")
```
Generates a Python code file. Code documentation is intentionally avoided by default to keep the Python code clean. 

``` Python
bc.spill("my_code.py", docs=True)
```
Pass the optional `docs` parameter to generate Python code along with full code documentation. The code documentation is included as multi-line strings. 

# Specifying Features
Framework automatically performs a feature selection. All features (except target) are considered by default for feature selection.
Framework is smart enough to remove ID / Primary key columns. 

You can manually specifiy a subset of features to be used for training. 

``` Python
bc.train("data.csv", target="Y_value", features=["col1", "col2", "col3"])
```

This does not guarantee that all specified features will be used in the final model. The framework will perform an automated feature selection from amongst these features. This only guarantees that other features if present in the data will not be considered. 

# Printing Model Stats
``` Python
model = bc.train(file='./test.csv')
model.stats()
```

# Saving the Model
``` Python
model = bc.train(file='./test.csv')
model.save('./my_model.pkl')
```

Use the `save` method to serialise and save the model instance to a Pickle file. A trained model along with all its attributes can be saved. A saved model can be loaded back in the future, and used for preditions, code generation, or viewing the model stats. 

# Loading a Saved Model
``` Python 
model = bc.load('./my_model.pkl')
```

The loaded model is an exact replica of the model, as at the time of saving the model. The training state along with other training parameters are persisted and reloaded.  