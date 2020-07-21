# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:27:09 2020

@author: rhs
"""

import yaml
import pykognition as pykog
import os
import pandas as pd
 

#from package_name import no_ssl_verification


_root = os.getcwd()
input_path = os.path.join(_root, "inputEmotions", "")
output_path = os.path.join(_root, "outputEmotions", "")

# reading credentials
_credentials = yaml.load(open(os.path.join(os.getcwd(), 'Credentials.yaml')))
_personal_access_key = _credentials['amazon']['personal_access_key']
_secret_access_key = _credentials['amazon']['secret_access_key']


local_images = os.listdir(input_path)
local_images = local_images[:5]



ifa = pykog.ImageFaceAnalysis(_personal_access_key, _secret_access_key)

ifa.initialize(inputPath = input_path, imageFileList = local_images)

expdf = ifa.get(attributes = ['emotions', features])
#%%


ioa = pykog.ImageObjectAnalysis(_personal_access_key, _secret_access_key)
ioa.initialize(inputPath = input_path, imageFileList = local_images)

objdf = ioa.get()




#%%