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
        "nbformat==5.10.4",
        "requests==2.32.3",
        "numpy==1.22.0",
        "pandas==2.0.3",
        "httplib2==0.20.0",
        "PyYAML==6.0",
        "scikit-learn==1.3.2",
        "seaborn==0.10.0",
        "optuna==2.10.1",
        "imbalanced-learn == 0.8.0",
        "xgboost==1.6.0",
        "catboost==1.2.5",
        "lightgbm==3.2.0",
        "opencv-python==4.10.0.84",
        "statsmodels==0.14.0",
        "autokeras==1.0.20",
        "ipython==8.12.3",
        "openpyxl==3.1.5",
        "matplotlib==3.7.2",
        "scipy",
        "setuptools==69.5.1",
        "tensorflow==2.13.0",
        "tensorflow_intel==2.13.0",
        "tqdm==4.66.4"
    ] 
)