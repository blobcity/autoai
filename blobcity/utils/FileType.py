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


import os
import io
import re
import urllib
import requests
import pandas as pd
import httplib2
from requests.models import HTTPError

class DataFrameHandler:
    def __init__(self, dc=None):
        self.dc = dc

    def get_dataframe(self, file_path):
        """
        Reads a file (local or URL) and returns a pandas DataFrame.
        """
        extension = os.path.splitext(file_path)[1]
        types = extension.lstrip('.')
        try:
            df = self._read_file(file_path, extension)
        except HTTPError:
            df = self._read_from_url(file_path, extension)
        
        if self.dc:
            self.dc.addKeyValue('data_read', {"type": types, "file": file_path, "class": "df"})
        return df

    def _read_file(self, file_path, extension):
        read_funcs = {
            ".csv": pd.read_csv,
            ".xlsx": pd.read_excel,
            ".parquet": pd.read_parquet,
            ".json": pd.read_json,
            ".pkl": pd.read_pickle
        }
        return read_funcs.get(extension, lambda x: None)(file_path)

    def _read_from_url(self, file_path, extension):
        response = requests.get(file_path)
        file_object = io.StringIO(response.content.decode('utf-8'))
        return self._read_file(file_object, extension)

    def write_dataframe(self, dataframe, path):
        """
        Writes a pandas DataFrame to a specified file path.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(dataframe)}")
        
        extension = os.path.splitext(path)[1].lstrip('.')
        if extension not in ['csv', 'xlsx', 'json']:
            raise TypeError(f"Unsupported file format: {extension}")
        
        self._save_dataframe(dataframe, path, extension)
    
    def _save_dataframe(self, dataframe, path, ftype):
        save_funcs = {
            'csv': dataframe.to_csv,
            'xlsx': dataframe.to_excel,
            'json': lambda p: dataframe.to_json(p, orient="index")
        }
        save_funcs[ftype](path, index=False)
        print(f"Saved at path {path}")

class URLValidator:
    DOMAIN_FORMAT = re.compile(
        r"(?:^(\w{1,255}):(.{1,255})@|^)"  # HTTP basic authentication [optional]
        r"(?:(?:(?=\S{0,253}(?:$|:))"  # Fixed unbalanced parenthesis
        r"((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
        r"(?:[a-z0-9]{1,63})))"
        r"|localhost)"  # Accept "localhost" as well
        r"(:\d{1,5})?",  # Port [optional]
        re.IGNORECASE
    )
    SCHEME_FORMAT = re.compile(r"^(http|https|ftp|ftps)$", re.IGNORECASE)

    @staticmethod
    def validate(url: str):
        """
        Validates whether a given string is a properly formatted URL.
        """
        url = url.strip()
        if not url:
            raise ValueError("No URL specified")
        
        result = urllib.parse.urlparse(url)
        scheme, domain = result.scheme, result.netloc
        
        if not scheme or not re.fullmatch(URLValidator.SCHEME_FORMAT, scheme):
            raise ValueError(f"Invalid URL scheme: {scheme}")
        if not domain or not re.fullmatch(URLValidator.DOMAIN_FORMAT, domain):
            raise ValueError(f"Malformed domain: {domain}")
        
        return URLValidator.check_url_existence(url)

    @staticmethod
    def check_url_existence(url):
        """
        Checks if the URL exists.
        """
        h = httplib2.Http()
        resp = h.request(url, 'HEAD')
        if int(resp[0]['status']) < 400:
            return True
        else:
            raise HTTPError(f"{url} does not exist")
