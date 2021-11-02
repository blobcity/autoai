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
A framework to find, train and generate code for the best performing AI model. Works on Classification and Regression problems on numerical data. This is a beta release. The framework is being actively worked upon. Please report any issues you encounter.

[![Issues](https://shields.io/github/issues/blobcity/autoai)](https://github.com/blobcity/autoai/issues)


# Getting Started
``` shell
pip install blobcity
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
Multiple formats of code generation is supported by the framework. The `spill` function can be used to generate both `ipynb` and `py` files.

``` Python
model.spill("my_code.ipynb"); #produces Jupyter Notebook file with full markdown docs

model.spill("my_code.py") #produces python code with minimal docs

model.spill("my_code.py", docs=True) #python code with full docs

model.spill("my_code.ipynb", docs=False) #Notebook file with minimal markdown
```

# Feature Selection
Framework automatically performs a feature selection. All features (except target) are considered by default for feature selection.
Framework is smart enough to remove ID / Primary key columns. 

You can manually specify a subset of features to be used for training. An automatic feature selection will still be carried out, but will be restricted to the subset of features provided. 

``` Python
model = bc.train(file="data.csv", target="Y_value", features=["col1", "col2", "col3"])
```

This does not guarantee that all specified features will be used in the final model. The framework will perform an automated feature selection from amongst these features. This only guarantees that other features if present in the data will not be considered. 

```Python
model.features() #prints the features selected by the model

model.plot_feature_importance() #shows a feature importance graph
```

# Printing Model Stats
``` Python
model.stats()
```

Print the key model parameters, such as Precision, Recall, F1-Score. The parameters change based on the type of AutoAI problem. 

# Persistance
``` Python
model.save('./my_model.pkl')

model = bc.load('./my_model.pkl')
```

You can save a trained model, and load it in the future to generate predictions. 

# Features and Roadmap
- [x] Numercial data Classification and Regression
- [x] Automatic feature selection
- [x] Code generation
- [ ] Neural Networks & Deep Learning
- [ ] Image classification
- [ ] Optical Character Recognition (english only)
- [ ] Video tagging with YOLO
- [ ] Generative AI using GAN
