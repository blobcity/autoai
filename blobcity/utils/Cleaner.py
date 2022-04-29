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
    
def dataCleaner(df,features,target,DictionaryClass=None):
    """
    Funciton to check null occurances and handles other functions.

    param1: pandas DataFrame
    param2: target column name
    param3: Dictionary Class object

    return: pandas dataframe

    working:
    First the function identifies the problem type that is either regression or classification using ProType Class and its function checkType.
    For the Complete dataframe if any rows has more then 50% null values or missing values we will drop it.
    For the Complete dataframe if any columns has more then equal to 80% null missing values we will drop it to avoid any noise or sequed data imputation.
    Then check whether the dataframe has any null values. 
    if TRUE then : get all the columns names with null/missing values.
    for each columns with missing data call Cleaner function to Handle missing value with appropriate missing value handling strategy.
    Once Missing values are handled perform categorical value handling operation by calling Encoder() function with appropriate arguments
    """
    missingdict=dict()
    if DictionaryClass!=None:DictionaryClass.addKeyValue('problem',ProType.checkType(df[target]))
    updateddf=df[features].copy(deep=True)
    if target in df.columns.to_list(): updateddf[target]=df[target].copy(deep=True)
    updateddf=RemoveRowsWithHighNans(updateddf)
    updateddf=RemoveHighNullValues(updateddf)
    updateddf=dropUniqueColumn(updateddf,target)
    if updateddf.isnull().values.any(): 
        cols=updateddf.columns[updateddf.isnull().any()].tolist()
        for i in cols:
            Cleaner(updateddf,i,missingdict)
        if DictionaryClass!=None:DictionaryClass.addKeyValue('cleaning',{'missingValues':missingdict})

    X_values,Y_value=updateddf.drop(target,axis=1),updateddf[target]
    if target in updateddf.columns.to_list():EncoderResult=Encoder(DictionaryClass,X_values,Y_value,target)
    else:EncoderResult=Encoder(DictionaryClass,X_values,None,target)
    if DictionaryClass!=None:DictionaryClass.addKeyValue('features',{'X_values':X_values.columns.to_list(),'Y_values':target})

    return EncoderResult

def dropUniqueColumn(X_values,target):
    """
        param1: pandas.DataFrame 
        return : pandas.DataFrame

        Function Drop Column with Complete Unique data for example data such as ID,UniqueID etc.
        for all available feature in the dataframe it checks whether the column has unique value counts equal to number of entries
        in the dataset. and drops if exists and return dataframe once dropped.
    """    
    row_counts = len(X_values)
    for i in X_values.columns.to_list():
        if len(X_values[i].unique())==row_counts and i!=target:
            X_values.drop(i, axis=1, inplace=True)
    return X_values

def RemoveHighNullValues(dataframe):
    """
    param1: pandas.DataFrame
    return: pandas.DataFrame

    Function drops any feature with more then 80% of Null Values and return the Dataframe
    """
    thresh = len(dataframe) * .5
    dataframe.dropna(thresh = thresh, axis = 1, inplace = True)
    return dataframe

def Cleaner(df,i,missingdict):
    """
    param1: pandas DataFrame
    param2: column name
    param3: placeholder dictionary for YAML record

    Working:
     This funciton Handles missing values in the dataset by considering the datatype of each passed column to the function. 
     if the columns datatype is object to uses mode imputation.
     while if the datatype is integer ir float, and has 3 or less unique values in it mode imputation is used,else mean imputation
    
    """
    if(df[i].dtype in ["float","int"]):
        if(len(np.unique(df[i]))<=3):
            df[i].fillna(df[i].mode()[0],inplace=True)
            missingdict[i]="mode"
        else:
            df[i].fillna(df[i].mean(),inplace=True)   
            missingdict[i]="mean"
    elif(df[i].dtype=="object"):
        df[i].fillna(df[i].mode()[0],inplace=True)
        missingdict[i]="mode"

def Encoder(DictionaryClass,X,Y=None,target=""):
    """
    Function to Encode categorical data from the feature set and target sets.
    
    param1: X_values pd.dataframe type
    param2: y_values pd.series or numpy.ndarray

    return: DataFrame 

    working:
        if the feature set(X_values) has any string categorical data(object/string/category) 
        then: apply one hot encoding to the feature 
        and 
        if the target (Y_values) is a categorical data (object/category) 
        then : Apply Label Encoding Technique to the target columns

        finally return both the feature and target .
    """
    encode=dict()
    if("object" in X.dtypes.to_list() or Y.dtype=="object"):
        if("object" in X.dtypes.to_list()):
            objectTypes(X,DictionaryClass)
            X=pd.get_dummies(X)
            encode['X']='OneHotEncode' 
        dataframe=X.copy(deep=True)
        if(Y.dtype=="object"):
            encode['Y']='LabelEncoder' 
            original_labels=np.sort(pd.unique(Y), axis=-1, kind='mergesort')
            Y=LabelEncoder().fit_transform(Y)
            encoded_label=[xi for xi in range(len(original_labels))]
            encodes={encoded_label[i]:original_labels[i] for i in range(len(original_labels))}
            if DictionaryClass!=None:DictionaryClass.original_label=encodes
        dataframe[target]=Y
        if DictionaryClass!=None:DictionaryClass.UpdateNestedKeyValue('cleaning','encode',encode)
        return dataframe
    else:
        dataframe=X.copy(deep=True)
        dataframe[target]=Y
        return dataframe

def objectTypes(X,DictionaryClass):
    """
    param1: pandas.dataframe
    param2: class object

    Function indentifies existence of String Categorical features.
    If String Categorical Feature exist record the list of features with string data in Class List Variable,
    and set boolean flag for existence to True else False
    """

    g = X.columns.to_series().groupby(X.dtypes).groups
    gd={k.name: v for k, v in g.items()}
    if 'object' in gd.keys():
        if DictionaryClass!=None:
            DictionaryClass.ObjectExist=True
            DictionaryClass.ObjectList= gd['object'].to_list()  
    else:
        if DictionaryClass!=None:DictionaryClass.ObjectExist= False

def RemoveRowsWithHighNans(dataframe):
    """
    param1: pandas.DataFrame
    return: pandas.DataFrame

    Function delete rows containing more than 50% NaN Values
    """
    percent = 50.0
    min_count = int(((100-percent)/100)*dataframe.shape[1] + 1)
    dataframe = dataframe.dropna( axis=0, 
                    thresh=min_count)
    return dataframe

def scaling_data(dataframe,DictionaryClass,update=False):
    """
    param1: pandas.DataFrame
    param2: Class object
    param3: boolean
    return: pandas.DataFrame

    Function applies StandardScaler or MinMaxScaler on provided dataframe depending on existence of Object type feature.
    """
    
    scaler=MinMaxScaler() if DictionaryClass.ObjectExist else StandardScaler()
    X=scaler.fit_transform(dataframe) 
    if isinstance(dataframe,pd.DataFrame):X=pd.DataFrame(data = X,columns = dataframe.columns)
    if update:
        DictionaryClass.UpdateNestedKeyValue('cleaning','rescale',scaler.__class__.__name__)
        DictionaryClass.Scaler=scaler
    return X

def uncompress_file(file,DictionaryClass):
    """
    param1: string
    param2: Class object
    return: string
    
    Function checks if the file exists and call the decompressing functions.
    """
    if os.path.isfile(file):
        return decompress(file,DictionaryClass)
    else:
        raise FileNotFoundError(f"provided path {file} does not exist")
    
def decompress(file,DictionaryClass):
    """
    param1: string
    param2: Class object
    return: string

    Function decompress file of zip or gz format and return the decompressed file path.
    """
    try:
        ogpath=os.path.splitext(file)
        extract_dir="./"+os.path.basename(ogpath[0])
        prog=Progress()
        if is_zipfile(file):
            with ZipFile(file,"r") as zip_ref:
                members=zip_ref.namelist()
                prog.create_progressbar(n_counters=len(members),desc="Decompressing :")
                for file in members:
                    zip_ref.extract(member=file,path=extract_dir)
                    prog.update_progressbar(1)
                prog.close_progressbar()
            DictionaryClass.UpdateNestedKeyValue('data_read','decompress','zip')
        elif is_tarfile(file):
            tar = tarfile.open(file, mode="r:gz")
            members=tar.getmembers()
            prog.create_progressbar(n_counters=len(members),desc="Decompressing :")
            for member in members:
                tar.extract(member=member,path=extract_dir)
                prog.update_progressbar(1)
            prog.close_progressbar()
            DictionaryClass.UpdateNestedKeyValue('data_read','decompress','gz')
        print(f"file has been decompressed to folder {extract_dir}")
        DictionaryClass.UpdateNestedKeyValue('data_read','decompressed_path',extract_dir)
    except Exception as e:print(e)
    return extract_dir

def file_from_url(url,DictionaryClass):
    """
    param1: String
    param2: Class object
    return: string

    Function downloads file from the Network bound files and return the local system path of the downloaded file
    """
    try:
        ogpath=os.path.splitext(url)
        download_path="./"+os.path.basename(ogpath[0])+ogpath[-1]
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        with tqdm.wrapattr(open(download_path, "wb"), "write", miniters=1,total=total,desc="Downloading :") as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)
        DictionaryClass.UpdateNestedKeyValue('data_read','from','URL')
        DictionaryClass.UpdateNestedKeyValue('data_read','downloaded_path',download_path)
        return download_path
    except Exception as e: print(e)

def check_subfolder_data(file,DictionaryClass):
    """
    param1:string
    param2: Class object
    return: Tuple(string,list)

    Function check individual files in the directory to verify whether all the files are of format image and return target/class
    identified from it.
    """
    targets = os.listdir(file)
    print(f"identified target are :{targets}")
    check_status=True
    for category in targets:
        path=os.path.join(file, category)
        if not os.path.isfile(path):
            for img in os.listdir(path):
                try:
                    extension = os.path.splitext(img)[1]
                    check_status= check_status if extension in ['.png',".PNG",".jpg",".jpeg",'.JPEG'] else False 
                    if not check_status:break
                except Exception as e:print(e)
        else: check_status=False
        if not check_status:break
    DictionaryClass.addKeyValue('features',{'Y_values':targets})
    if not check_status: raise TypeError("some files have different formats")
    return (file,targets)

def quick_image_processing(path,size):
    """
    param1: string: path of image file
    param2: int: image resize resolution
    return: numpy.darray

    Function flattens a image appropriate for model prediction
    """
    data = cv2.imread(path)
    data=cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
    img_resize=cv2.resize(data,(size,size))
    img_data=[img_resize.flatten()]
    return (img_data,data)


def timeseries_cleaner(X,date,target,samplingtype,dictclass):
    X=X.loc[0:X.shape[0],[date,target]].copy(deep=True) 
    updateddf=RemoveRowsWithHighNans(X)
    updateddf=RemoveHighNullValues(X)
    updateddf=parsetime(X,date,dictclass)
    updateddf=FrequencyChecker(updateddf,date,dictclass)
    X = updateddf[target].values
    s = StationarityTest()
    s.results(X)
    updateddf=frequencysampling(updateddf,date,dictclass,samplingtype) if samplingtype!=None else updateddf
    updateddf=RemoveHighNullValues(updateddf)
    return updateddf


def parsetime(df,date,dictclass):
    try:
        df[date]= pd.to_datetime(df[date])
        return df
    except:
        try:
            df[date] = pd.to_datetime(df[date],format="%d.%m.%Y")

        except:
            raise TypeError("Unsupported Date Format")
        
def FrequencyChecker(df,date,dictclass):
    df_copy=df.copy(deep=True)
    df['Month']=df[date].dt.month
    df['Year']=df[date].dt.year
    df['Hour']=df[date].dt.hour
    df['Days']=df[date].dt.day_name()
    d=df.Days.nunique()
    h=df.Hour.nunique()
    m=df.Month.nunique()
    if(h>1 and h<=24):
        dictclass.time_frequency="H"
        
    elif(d>1 and d<=7):
        dictclass.time_frequency="D"
        
    elif(m>1 and m<=12):
        dictclass.time_frequency="M"
        
    else:
        dictclass.time_frequency="Y"
    df_copy=df_copy.set_index(date)
    return df_copy
   
class StationarityTest:
    def __init__(self, SignificanceLevel=.05,test=[]):
        self.SignificanceLevel = SignificanceLevel 
        self.test=test
    def ADF_Stationarity_Test(self, timeseries):
        adfTest = adfuller(timeseries, autolag='AIC')
        self.pValue = adfTest[1]
        if (self.pValue<self.SignificanceLevel):
            self.test.append(True)
        else:
            self.test.append(False)
            
    def kpss_test(self,timeseries):
        statistic, p_value, n_lags, critical_values = kpss(timeseries)
        if (p_value < self.SignificanceLevel):
            self.test.append(False)
        else:
            self.test.append(True)
    def seasonality_test(self,timeseries):
        seasoanl = False
        idx = np.arange(len(timeseries)) % 12
        H_statistic, p_value = kruskal(timeseries, idx)
        if p_value <= self.SignificanceLevel:
            seasonal = True
        self.test.append(seasonal)
    def results(self,timeseries):
        StationarityTest.ADF_Stationarity_Test(self, timeseries)
        StationarityTest.kpss_test(self,timeseries)
        StationarityTest.seasonality_test(self,timeseries)
        
        result=max(self.test, key=self.test.count)
       
        return result

def frequencysampling(df,date,dictclass,samplingtype):
    downsample=""
    #df=df.set_index(date)
    if (samplingtype=="day" and dictclass.time_frequency=="H"):
        dff=df.resample('H').mean()
        downsample="day"
        
    elif(samplingtype=="week" and dictclass.time_frequency in ["H","D"]):
        dff=df.resample("D").mean()
        downsample="week" 
    elif (samplingtype=="month" and dictclass.time_frequency in ["H","D"]):
        dff=df.resample('M').mean()
        downsample="month"    
        
    elif (samplingtype=="quarterly" and dictclass.time_frequency in ["H","D","M"]):
        dff=df.resample("Q").mean()
        downsample="quarterly"
        
    elif (samplingtype=="year" and dictclass.time_frequency in ["H","D","M"]):
        different_locale=df.resample('M').mean()
        downsample="year"

    elif (samplingtype not in ["year","quaterly"," month","week","day",None]):
        raise ValueError(f"{samplingtype} is not a valid option, valid options are ['year','quaterly','month','week','day',None]")
    if downsample not in [None,""]:
        dictclass.addKeyValue("cleaning",{"downsample":downsample})

    return dff

def spliter(df):
    size=df.shape[0]
    trainsize=int(np.round(90*size/100))
    train_data=df.iloc[0:trainsize,:].squeeze()
    test_data=df.iloc[trainsize:,:].squeeze()
    return train_data, test_data 


