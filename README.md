<a href="https://pix.blobcity.com/I1Nk23FY"><img src="https://blobcity.com/assets/img/blobcity-logo.svg" style="width: 40%"/></a>

[![PyPI version](https://badge.fury.io/py/blobcity.svg)](https://badge.fury.io/py/blobcity)
[![Downloads](https://pepy.tech/badge/blobcity/month)](https://pepy.tech/project/blobcity)
[![Vulnerabilities](https://shields.io/snyk/vulnerabilities/github/blobcity/autoai)](https://snyk.io/product/open-source-security-management/)
[![Python](https://shields.io/pypi/pyversions/blobcity)](https://pypi.org/project/blobcity/)
[![License](https://shields.io/pypi/l/blobcity)](https://pypi.org/project/blobcity/)

[![Contributors](https://shields.io/github/contributors/blobcity/autoai)](https://github.com/blobcity/autoai)
[![Commit Activity](https://shields.io/github/commit-activity/m/blobcity/autoai)](https://github.com/blobcity/autoai)
[![Last Commit](https://shields.io/github/last-commit/blobcity/autoai)](https://github.com/blobcity/autoai)
[![Slack](https://shields.io/badge/join-slack-orange)](https://pix.blobcity.com/E2Bepr4w)

[![GitHub Stars](https://shields.io/github/stars/blobcity?style=social)](https://github.com/blobcity)
[![Twitter](https://shields.io/twitter/follow/blobcity?label=Follow)](https://twitter.com/blobcity)


# BlobCity AutoAI
A framework to find, train and generate code for the best performing AI model. Works on Classification and Regression problems. The framework is currently designed for tabular data, and is being extended to support images, videos and natural language. 

# Getting Started
``` shell
pip install blobcity
```

If not already installed, then you must also install Tensorflow. This is a required dependency, but not installed by default. 
```shell
pip install tensorflow
```

``` Python
import blobcity as bc
model = bc.train(file="data.csv", target="Y_column")
model.spill("my_code.py")
```
`Y_column` is the name of the target column. The column must be present within the data provided. 

Automatic inference of Regression / Classification is supported by the framework.

Supported input data formats are `.csv` and `.xlsx`. Extension for other file formats is being worked on. 

The `spill` function generates the model code with exhaustive documentation. Training code is also included for scikit-learn models. TensorFlow and other DNN models produce only the test / final use code. 

## Use a Pandas Data Frame
``` Python
model = bc.train(df=my_df, target="Y_column")
```

If loading data from a Database or external system, create a DataFrame from your data source, and pass it directly to the `train` function.

## From a URL
``` Python
model = bc.train(file="https://example.com/data.csv", target="Y_column")
```

The `file` parameter can be a local file, or a URL. The function will load data from the specified URL. The file at the URL must be either in CSV or XLSX format. The URL should be accessible publicly without authentication. 

# Code Generation
Multiple formats of code generation is supported by the framework. The `spill` function can be used to generate both `ipynb` and `py` files. The desired type is infered from the name of the output file. The code file will be created at the path specified. Relative and absolute file paths are both supported. 

### Generate Jupyter Notebook
``` Python
model.spill("my_code.ipynb");
```
Generates an ipynb file with full markdown documentation containing code explanations. 

### Generate Python Code
``` Python
model.spill("my_code.py")
```
Generates a Python code file. Code documentation is intentionally avoided by default to keep the Python code clean. 

``` Python
model.spill("my_code.py", docs=True)
```
Pass the optional `docs` parameter to specify if relevant source code documentation should be included in the generated code.

# Specifying Features
Framework automatically performs a feature selection. All features (except target) are considered by default for feature selection.
Framework is smart enough to remove ID / Primary key columns. 

You can manually specify a subset of features to be used for training. An automatic feature selection will still be carried out, but will be restricted to the subset of features provided. 

``` Python
bc.train("data.csv", target="Y_value", features=["col1", "col2", "col3"])
```

This does not guarantee that all specified features will be used in the final model. The framework will perform an automated feature selection from amongst these features. This only guarantees that other features if present in the data will not be considered. 

# Printing Model Stats
``` Python
model.stats()
```

Key model parameters, such as Precision, Recall, F1-Score are printed using the `stats` function. The parameters change based on the type of AutoAI problem. 

# Saving the Model
``` Python
model.save('./my_model.pkl')
```

Use the `save` method to serialise and save the model instance to a Pickle file. A trained model along with all its attributes can be saved. A saved model can be loaded back in the future, and used for predictions, code generation, or viewing the model stats. 

# Loading a Saved Model
``` Python 
model = bc.load('./my_model.pkl')
```

The loaded model is an exact replica of the model, as at the time of saving the model. The training state along with other training parameters are persisted and reloaded.

# Features and Roadmap
- [x] Classification and Regression on numeric data
- [x] Automatic feature selection
- [x] py code generation
- [x] ipynb code generation
- [ ] ipynb code generation, with exhaustive markdown documentation
- [ ] Image classification
- [ ] Opitical Character Recognition (english only)
- [ ] Video tagging with YOLO

