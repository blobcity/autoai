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
This Python file consists of function to perform basic data cleaning/data preprocessing operation on most dataset.
Functions includes, Removal of Unique COlumns,High Null value ratio, Missing Value Handling, String Categorical feature Handling .
"""
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tarfile import is_tarfile
import os,tarfile,requests,warnings
from zipfile import ZipFile, is_zipfile
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from blobcity.utils.ProblemType import ProType
from blobcity.utils.progress_bar import Progress
from blobcity.store.DictClass import DictClass
from scipy.stats import kruskal
from statsmodels.tsa.stattools import kpss,adfuller
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ["PYTHONWARNINGS"] = "ignore"
    from statsmodels.tsa.stattools import kpss,adfuller
    
def data_cleaner(df, features, target, DictionaryClass=None):
    """
    Cleans the dataset by handling missing values, dropping high-null columns/rows, 
    and encoding categorical features.

    :param df: pandas.DataFrame
    :param features: List of feature column names
    :param target: Target column name
    :param DictionaryClass: Optional class object for tracking transformations
    :return: Processed pandas.DataFrame
    """

    missing_dict = {}

    # Determine problem type (regression/classification)
    if DictionaryClass:
        DictionaryClass.addKeyValue('problem', ProType.check_type(df[target]))

    # Create a copy of the dataframe with selected features and target
    updated_df = df[features].copy()
    if target in df.columns:
        updated_df[target] = df[target].copy()

    # Drop rows and columns with excessive missing values
    updated_df = remove_rows_with_high_nans(updated_df)
    updated_df = remove_high_null_values(updated_df)
    updated_df = drop_unique_column(updated_df, target)

    # Handle missing values
    if updated_df.isnull().values.any():
        missing_cols = updated_df.columns[updated_df.isnull().any()]
        for col in missing_cols:
            cleaner(updated_df, col, missing_dict)

        if DictionaryClass:
            DictionaryClass.addKeyValue('cleaning', {'missingValues': missing_dict})

    # Separate features and target
    X_values = updated_df.drop(columns=[target])
    Y_value = updated_df[target] if target in updated_df.columns else None

    # Encode categorical values
    encoder_result = encoder(DictionaryClass, X_values, Y_value, target)

    # Store processed feature information
    if DictionaryClass:
        DictionaryClass.addKeyValue('features', {'X_values': X_values.columns.tolist(), 'Y_values': target})

    return encoder_result

def drop_unique_column(X_values, target):
    """
    Drops columns where all values are unique, except for the target column.

    :param X_values: pandas.DataFrame 
    :param target: Target column name
    :return: pandas.DataFrame with unique columns removed
    """
    unique_cols = [col for col in X_values.columns if X_values[col].nunique() == len(X_values) and col != target]
    if len(X_values.columns) > 2:
        X_values = X_values.drop(columns=unique_cols)
    return X_values

def remove_high_null_values(dataframe, threshold=0.5):
    """
    Drops columns with more than `threshold`% missing values.

    :param dataframe: pandas.DataFrame
    :param threshold: Percentage threshold for dropping (default=80%)
    :return: pandas.DataFrame
    """
    min_non_null = len(dataframe) * (1 - threshold)
    return dataframe.dropna(thresh=min_non_null, axis=1)

def cleaner(df, col, missing_dict):
    """
    Handles missing values based on column data type.

    :param df: pandas.DataFrame
    :param col: Column name
    :param missing_dict: Dictionary to record imputation type
    """
    if df[col].dtype in ["float64", "int64"]:
        strategy = "mode" if df[col].nunique() <= 3 else "mean"
        df[col].fillna(df[col].mode()[0] if strategy == "mode" else df[col].mean(), inplace=True)
    else:
        strategy = "mode"
        df[col].fillna(df[col].mode()[0], inplace=True)

    missing_dict[col] = strategy
    
def encoder(DictionaryClass, X, Y=None, target=""):
    """
    Encodes categorical features and target variable.

    :param DictionaryClass: Class object
    :param X: Feature set (pandas.DataFrame)
    :param Y: Target variable (pandas.Series or numpy.ndarray)
    :param target: Target column name
    :return: Encoded pandas.DataFrame
    """
    encode = {}

    if X.select_dtypes(include=['object']).shape[1] > 0:
        object_types(X, DictionaryClass)
        X = pd.get_dummies(X)
        encode['X'] = 'OneHotEncode'

    if Y is not None and Y.dtype == 'object':
        encode['Y'] = 'LabelEncoder'
        original_labels = np.sort(pd.unique(Y))
        Y = LabelEncoder().fit_transform(Y)
        DictionaryClass.original_label = dict(enumerate(original_labels))

    DictionaryClass.UpdateNestedKeyValue('cleaning', 'encode', encode)
    
    df_encoded = X.copy()
    if Y is not None:
        df_encoded[target] = Y

    return df_encoded

def object_types(X, DictionaryClass):
    """
    Identifies object-type (categorical) columns.

    :param X: pandas.DataFrame
    :param DictionaryClass: Class object
    """
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    DictionaryClass.ObjectExist = bool(object_cols)
    DictionaryClass.ObjectList = object_cols

def remove_rows_with_high_nans(dataframe, threshold=0.5):
    """
    Drops rows where more than `threshold`% of values are NaN.

    :param dataframe: pandas.DataFrame
    :param threshold: Percentage threshold for dropping (default=50%)
    :return: pandas.DataFrame
    """
    min_count = int((1 - threshold) * dataframe.shape[1]) + 1
    return dataframe.dropna(thresh=min_count)

def scaling_data(dataframe, DictionaryClass, update=False):
    """
    Scales the dataset using StandardScaler or MinMaxScaler.

    :param dataframe: pandas.DataFrame
    :param DictionaryClass: Class object
    :param update: Boolean to update DictionaryClass
    :return: Scaled pandas.DataFrame
    """
    scaler = MinMaxScaler() if DictionaryClass.ObjectExist else StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)

    if update:
        DictionaryClass.UpdateNestedKeyValue('cleaning', 'rescale', scaler.__class__.__name__)
        DictionaryClass.Scaler = scaler

    return X_scaled

def uncompress_file(file, DictionaryClass):
    """
    Checks if the file exists and calls the decompressing function.
    
    :param file: File path as a string
    :param DictionaryClass: Class object
    :return: Decompressed file path as a string
    """
    if os.path.isfile(file):
        return decompress(file, DictionaryClass)
    raise FileNotFoundError(f"Provided path {file} does not exist")

def decompress(file, DictionaryClass):
    """
    Decompresses a zip or gz file and returns the decompressed file path.
    
    :param file: File path as a string
    :param DictionaryClass: Class object
    :return: Decompressed file path as a string
    """
    try:
        extract_dir = f"./{os.path.splitext(os.path.basename(file))[0]}"
        prog = Progress()

        if is_zipfile(file):
            with ZipFile(file, "r") as zip_ref:
                members = zip_ref.namelist()
                prog.create_progressbar(n_counters=len(members), desc="Decompressing:")
                for member in members:
                    zip_ref.extract(member=member, path=extract_dir)
                    prog.update_progressbar(1)
                prog.close_progressbar()
            DictionaryClass.UpdateNestedKeyValue('data_read', 'decompress', 'zip')
        elif tarfile.is_tarfile(file):
            with tarfile.open(file, mode="r:gz") as tar:
                members = tar.getmembers()
                prog.create_progressbar(n_counters=len(members), desc="Decompressing:")
                for member in members:
                    tar.extract(member=member, path=extract_dir)
                    prog.update_progressbar(1)
                prog.close_progressbar()
            DictionaryClass.UpdateNestedKeyValue('data_read', 'decompress', 'gz')

        print(f"File has been decompressed to folder {extract_dir}")
        DictionaryClass.UpdateNestedKeyValue('data_read', 'decompressed_path', extract_dir)
        return extract_dir
    except Exception as e:
        print(f"Error: {e}")
        return ""

def file_from_url(url, DictionaryClass):
    """
    Downloads a file from a URL and returns its local system path.
    
    :param url: URL as a string
    :param DictionaryClass: Class object
    :return: Local file path as a string
    """
    try:
        download_path = f"./{os.path.basename(os.path.splitext(url)[0])}{os.path.splitext(url)[-1]}"
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        with tqdm.wrapattr(open(download_path, "wb"), "write", miniters=1, total=total, desc="Downloading:") as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)
        DictionaryClass.UpdateNestedKeyValue('data_read', 'from', 'URL')
        DictionaryClass.UpdateNestedKeyValue('data_read', 'downloaded_path', download_path)
        return download_path
    except Exception as e:
        print(f"Error: {e}")
        return ""

def check_subfolder_data(file, DictionaryClass):
    """
    Verifies that all files in subdirectories of `file` are images and extracts target classes.

    :param file: Directory path as a string
    :param DictionaryClass: Class object with `addKeyValue` method
    :return: Tuple (string, list) containing the file path and identified targets
    :raises TypeError: If any non-image file is found
    """
    valid_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPEG'}
    
    try:
        targets = [d for d in os.listdir(file) if os.path.isdir(os.path.join(file, d))]
        print(f"Identified targets: {targets}")

        for category in targets:
            category_path = os.path.join(file, category)
            for img in os.listdir(category_path):
                if not os.path.splitext(img)[1] in valid_extensions:
                    raise TypeError("Some files have different formats")
    
        DictionaryClass.addKeyValue('features', {'Y_values': targets})
        return file, targets
    
    except Exception as e:
        print(f"Error: {e}")
        raise

def quick_image_processing(path: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads and processes an image for model prediction.

    Args:
        path (str): Path to the image file.
        size (int): Target resolution for resizing (size x size).

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - Flattened image array suitable for model input.
            - Original resized image (RGB format).

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the size is not a positive integer.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer.")

    data = cv2.imread(path)
    if data is None:
        raise ValueError(f"Failed to read image: {path}")

    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(data, (size, size))
    
    return img_resize.flatten(), img_resize

def parse_time(df, date):
    """
    Converts the date column to datetime format.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        date (str): The column name containing date values.

    Returns:
        pd.DataFrame: DataFrame with the parsed datetime column.

    Raises:
        TypeError: If date format is unsupported.
    """
    try:
        df[date] = pd.to_datetime(df[date], infer_datetime_format=True, errors='coerce')
        if df[date].isna().any():
            raise ValueError("Some dates could not be parsed.")
        return df
    except Exception as e:
        raise TypeError(f"Unsupported Date Format: {e}")

def frequency_checker(df, date, dictclass):
    """
    Determines the frequency of time intervals in the dataset.

    Args:
        df (pd.DataFrame): The input dataframe.
        date (str): The column name containing datetime values.
        dictclass (object): An object with a 'time_frequency' attribute.

    Returns:
        pd.DataFrame: DataFrame with datetime index set.
    """
    df = df.copy()
    df[date] = pd.to_datetime(df[date], errors='coerce')
    
    df['Month'] = df[date].dt.month
    df['Year'] = df[date].dt.year
    df['Hour'] = df[date].dt.hour
    df['Days'] = df[date].dt.day_name()

    unique_days = df['Days'].nunique()
    unique_hours = df['Hour'].nunique()
    unique_months = df['Month'].nunique()

    if unique_hours > 1 and unique_hours <= 24:
        dictclass.time_frequency = "H"
    elif unique_days > 1 and unique_days <= 7:
        dictclass.time_frequency = "D"
    elif unique_months > 1 and unique_months <= 12:
        dictclass.time_frequency = "M"
    else:
        dictclass.time_frequency = "Y"

    return df.set_index(date)

class StationarityTest:
    def __init__(self, significance_level=0.05):
        """
        Initializes the StationarityTest class.

        Args:
            significance_level (float): Threshold for p-value significance. Default is 0.05.
        """
        self.significance_level = significance_level
        self.test_results = []

    def adf_test(self, timeseries):
        """Performs the Augmented Dickey-Fuller (ADF) test for stationarity."""
        p_value = adfuller(timeseries, autolag='AIC')[1]
        self.test_results.append(p_value < self.significance_level)

    def kpss_test(self, timeseries):
        """Performs the KPSS test for stationarity."""
        p_value = kpss(timeseries)[1]
        self.test_results.append(p_value >= self.significance_level)

    def seasonality_test(self, timeseries):
        """Checks for seasonality using the Kruskal-Wallis H test."""
        seasonal = False
        idx = np.arange(len(timeseries)) % 12
        _, p_value = kruskal(timeseries, idx)
        if p_value <= self.significance_level:
            seasonal = True
        self.test_results.append(seasonal)

    def is_stationary(self, timeseries):
        """Runs all tests and returns True if the series is stationary."""
        self.test_results.clear()
        self.adf_test(timeseries)
        self.kpss_test(timeseries)
        self.seasonality_test(timeseries)
        return max(self.test_results, key=self.test_results.count)

def frequency_sampling(df, dictclass, sampling_type):
    """
    Resamples the DataFrame based on the specified sampling type.

    Args:
        df (pd.DataFrame): Input DataFrame with a datetime index.
        dictclass (object): An object with 'time_frequency' attribute and 'addKeyValue' method.
        sampling_type (str): Desired resampling frequency. Options: ["day", "week", "month", "quarterly", "year"].

    Returns:
        pd.DataFrame: Resampled DataFrame.
    """
    mapping = {"day": "H", "week": "D", "month": "M", "quarterly": "Q", "year": "M"}

    if sampling_type not in mapping:
        raise ValueError(f"Invalid option: {sampling_type}")

    if dictclass.time_frequency not in ["H", "D", "M"][:list(mapping).index(sampling_type) + 1]:
        raise ValueError(f"Incompatible time frequency: {dictclass.time_frequency}")

    df_resampled = df.resample(mapping[sampling_type]).mean()
    dictclass.addKeyValue("cleaning", {"downsample": sampling_type})
    
    return df_resampled

def split_data(df, train_ratio=0.9):
    """
    Splits a DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): Input DataFrame.
        train_ratio (float, optional): Proportion of data to use for training (default is 0.9).

    Returns:
        tuple: (train_data, test_data)
    """
    train_size = int(np.round(train_ratio * len(df)))
    return df.iloc[:train_size, :], df.iloc[train_size:, :]

def timeseries_cleaner(X, date, target, sampling_type, dict_class):
    """
    Cleans and processes a time series dataset by handling missing values, parsing dates, 
    checking stationarity, and performing frequency sampling.

    :param X: pandas.DataFrame
    :param date: Column name for date/time
    :param target: Column name for target variable
    :param sampling_type: Type of resampling (if applicable)
    :param dict_class: Class object for tracking transformations
    :return: Processed pandas.DataFrame
    """

    # Select only relevant columns
    X = X[[date, target]].copy()

    # Handle missing values
    X = remove_rows_with_high_nans(X)
    X = remove_high_null_values(X)

    # Process date column and check frequency
    X = parse_time(X, date, dict_class)
    X = frequency_checker(X, date, dict_class)

    # Perform stationarity test
    StationarityTest().results(X[target].values)

    # Apply frequency sampling if specified
    if sampling_type:
        X = frequency_sampling(X, date, dict_class, sampling_type)

    # Final check for high-null columns
    return remove_high_null_values(X)