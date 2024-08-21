import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "blobcity",
    version = "0.0.13",
    author = "BlobCity, Inc",
    author_email = "support@blobcity.com",
    url = "https://github.com/blobcity/autoai",
    description="Python based framework for Automatic AI for Regression and Classification over numerical data. Performs model search, hyper-parameter tuning, and high-quality Jupyter Notebook code generation.",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    license = "Apache License 2.0",
    packages=setuptools.find_packages(), 
    classifiers =[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',                # Minimum version requirement of the package
    py_modules=["blobcity"],             # Name of the python package
    # package_dir={'':'autoai/blobcity'},     # Directory of the source code of the package
    install_requires=[
        "dill==0.3.4",
        "cliff==3.6.0",
        "joblib==1.0.0",
        "nbformat==5.1.3",
        "requests==2.26.0",
        "numpy==1.21.0",
        "pandas==1.5.3",
        "httplib2==0.20.0",
        "PyYAML==6.0",
        "scikit-learn==0.24",
        "seaborn==0.10.0",
        "optuna==2.10.1",
        "imbalanced-learn == 0.8.0",
        "xgboost==1.4.0",
        "catboost==0.26",
        "lightgbm==3.2.0",
        "opencv-python==4.10.0.84",
        "statsmodels==0.14.0",
        "autokeras==1.0.20",
        "ipython",
        "openpyxl==3.1.5"
    ] 
)