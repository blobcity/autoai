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

def send_yaml_to_cloud(yaml_data):
    """
    param1: dictionary : full configuration of AutoAI process

    Function upload the yaml configuration data to the user account over BlobCity AI Cloud to maintain records of experiment.
    """
    token=os.environ.get('TOKEN',404)
    if token!=404:
        head={'Authorization':'Bearer '+token}
        try:
            API_ENDPOINT="https://cloud-api.blobcity.com/rest/v1/user/submit-autoAI"
            res=requests.post(API_ENDPOINT,json=yaml_data,headers=head)
            if res.status_code==200:
                if res.json()['ack']==1:print("AutoAI YAML configuration uploaded to BlobCity AI Cloud")
                elif res.json()['ack']==0:print("Not able to save the YAML config to the BlobCity AI Cloud currently")
            else:print("Some error occurred while sending yaml data")
        except Exception as e:
            print(e)