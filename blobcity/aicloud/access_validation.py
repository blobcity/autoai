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
import requests

def set_token(token):
    """
    param1: string : access token for AI Cloud 

    Function validated access token set by user with the AI Cloud server api
    """
    API_ENDPOINT="https://cloud-api.blobcity.com/rest/v1/user/validate-access-key"
    res=requests.post(url=API_ENDPOINT,headers={'Authorization':"Bearer "+token})
    if res.status_code==200:
        if res.json()['ack']==1:
            os.environ['TOKEN']=token
            print("Access Token has been set!")
        elif res.json()['ack']==0:
            raise Exception("Invalid Token")
    else:raise Exception("API access denied")