<a href="https://pix.blobcity.com/I1Nk23FY"><img src="https://blobcity.com/assets/img/blobcity-logo.svg" style="width: 40%"/></a>

[![PyPI version](https://badge.fury.io/py/blobcity.svg)](https://badge.fury.io/py/blobcity)
[![Downloads](https://pepy.tech/badge/blobcity/month)](https://pepy.tech/project/blobcity)
[![Python](https://shields.io/pypi/pyversions/blobcity)](https://pypi.org/project/blobcity/)
[![License](https://shields.io/pypi/l/blobcity)](https://pypi.org/project/blobcity/)

[![Contributors](https://shields.io/github/contributors/blobcity/autoai)](https://github.com/blobcity/autoai)
[![Commit Activity](https://shields.io/github/commit-activity/m/blobcity/autoai)](https://github.com/blobcity/autoai)
[![Last Commit](https://shields.io/github/last-commit/blobcity/autoai)](https://github.com/blobcity/autoai)
[![Slack](https://shields.io/badge/join%20discussion-slack-orange)](https://pix.blobcity.com/E2Bepr4w)

[![GitHub Stars](https://shields.io/github/stars/blobcity?style=social)](https://github.com/blobcity)
[![Twitter](https://shields.io/twitter/follow/blobcity?label=Follow)](https://twitter.com/blobcity)


# BlobCity AutoAI
A framework to find the best performing AI/ML model for any AI problem. Works for Classification and Regression type of problems on numerical data. AutoAI makes AI easy and accessible to everyone. It not only trains the best-performing model but also exports high-quality code for using the trained model.

The framework is currently in beta release, with active development being still in progress. Please report any issues you encounter.

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

Data input formats supported include:
1. Local CSV / XLSX file
2. URL to a CSV / XLSX file
3. Pandas DataFrame 

``` Python
model = bc.train(file="data.csv", target="Y_column") #local file
```

``` Python
model = bc.train(file="https://example.com/data.csv", target="Y_column") #url
```

``` Python
model = bc.train(df=my_df, target="Y_column") #DataFrame
```

# Pre-processing
The framework has built-in support for several data pre-processing techniques, such as imputing missing values, column encoding, and data scaling. 

Pre-processing is carried out automatically on train data. The predict function carries out the same pre-processing on new data. The user is not required to be concerned with the pre-processing choices of the framework. 

One can view the pre-processing methods used on the data by exporting the entire model configuration to a YAML file. Check the section below on "Exporting to YAML."

# Feature Selection
```Python
model.features() #prints the features selected by the model
```

```Shell
['Present_Price',
 'Vehicle_Age',
 'Fuel_Type_CNG',
 'Fuel_Type_Diesel',
 'Fuel_Type_Petrol',
 'Seller_Type_Dealer',
 'Seller_Type_Individual',
 'Transmission_Automatic',
 'Transmission_Manual']
 ```

AutoAI automatically performs a feature selection on input data. All features (except target) are potential candidates for the X input.  

AutoAI will automatically remove ID / Primary-key columns. 

This does not guarantee that all specified features will be used in the final model. The framework will perform an automated feature selection from amongst these features. This only guarantees that other features if present in the data will not be considered. 

AutoAI ignores features that have a low importance to the effective output. The feature importance plot can be viewed. 

```Python
model.plot_feature_importance() #shows a feature importance graph
```

![Feature Importance Plot](https://cdn.blobcity.com/img/autoai-feature-importance-example.png)

There might be scenarios where you want to explicitely exclude some columns, or only use a subset of columns in the training. Manually specify the features to be used. AutoAI will still perform a feature selection within the list of features provided to improve effective model accuracy. 

``` Python
model = bc.train(file="data.csv", target="Y_value", features=["col1", "col2", "col3"])
```

# Model Search, Train & Hyper-parameter Tuning
Model search, train and hyper-parameter tuning is fully automatic. It is a 3 step process that tests your data across various AI/ML models. It finds models with high success tendency, and performs a hyper-parameter tuning to find you the best possible result. 

[Regression Models Library](https://github.com/blobcity/autoai/blob/main/blobcity/config/regressor_config.py)

[Classification Models Library](https://github.com/blobcity/autoai/blob/main/blobcity/config/classifier_config.py)


# Code Generation
High-quality code generation is why most Data Scientists choose AutoAI. The `spill` function generates the model code with exhaustive documentation. scikit-learn models export with training code included. TensorFlow and other DNN models produce only the test / final use code. 

![AutoAI Generated Code Example](https://cdn.blobcity.com/img/autoai-code-gen-example.gif)


Code generation is supported in `ipynb` and `py` file formats, with options to enable or disable detailed documentation exports.

``` Python
model.spill("my_code.ipynb"); #produces Jupyter Notebook file with full markdown docs
```
``` Python
model.spill("my_code.py") #produces python code with minimal docs
```
``` Python
model.spill("my_code.py", docs=True) #python code with full docs
```
``` Python
model.spill("my_code.ipynb", docs=False) #Notebook file with minimal markdown
```

# Predictions
Use a trained model to generate predictions on new data. 

```Python
prediction = model.predict(file="unseen_data.csv")
```

All required features must be present in the `unseen_data.csv` file. Consider checking the results of the automatic feature selection to know the list of features needed by the `predict` function.


# Stats & Accuracy
```Python
model.plot_prediction()
```

The function is shared across Regression and Classification problems. It plots a relevant chart to assess efficiency of training. 

## Actual v/s Predicted Plot (for Regression)
![Actual v/s Predicted Plot](https://cdn.blobcity.com/img/autoai-regression-plot-full.png)

Plotting only first `100` rows. You can specify `-100` to plot last 100 rows.
```Python
model.plot_prediction(100)
```
![Actual v/s Predicted Plot first 100](https://cdn.blobcity.com/img/autoai-regression-plot-100.png)


## Confusion Matrix (for Classification)
![AutoAI Generated Code Example](https://cdn.blobcity.com/img/autoai-confusion-matrix.png)

## Numercial Stats
``` Python
model.stats()
```

Print the key model parameters, such as Precision, Recall, F1-Score. The parameters change based on the type of AutoAI problem. 

# Persistence
``` Python
model.save('./my_model.pkl')
```
```Python
model = bc.load('./my_model.pkl')
```

You can save a trained model, and load it in the future to generate predictions. 

# Accelerated Training
Leverage BlobCity AI Cloud for fast training on large datasets. Reasonable cloud infrastructure included for free.

[![BlobCity AI Cloud](https://shields.io/badge/Run%20On-BlobCity-orange)](https://pix.blobcity.com/pgMuJMLv)
[![CPU](https://shields.io/badge/CPU-Free-blue)](https://pix.blobcity.com/pgMuJMLv)
[![GPU](https://shields.io/badge/GPU-%2475%2Fmonth-green)](https://pix.blobcity.com/pgMuJMLv)


# Features and Roadmap
- [x] Numercial data Classification and Regression
- [x] Automatic feature selection
- [x] Code generation
- [ ] Neural Networks & Deep Learning
- [ ] Image classification
- [ ] Optical Character Recognition (english only)
- [ ] Video tagging with YOLO
- [ ] Generative AI using GAN
