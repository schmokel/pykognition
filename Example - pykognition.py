# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:27:09 2020

@author: rhs
"""

import yaml
import pykognition as pykog
import os
import pandas as pd
import re


_root = os.getcwd()
_ifa_datadir = os.path.join(_root, "inputEmotions", "")
input_path = "/inputEmotions"
# reading credentials
_credentials = yaml.load(open(os.path.join(os.getcwd(), 'Credentials.yaml')))
_personal_access_key = _credentials['amazon']['personal_access_key']
_secret_access_key = _credentials['amazon']['secret_access_key']


local_images = []
for root, dirs, files in os.walk(_root + input_path):
    for file in files:
        if(file.endswith(".jpg")):
            local_images.append(os.path.join(root, file))
            
            
            
#local_images = os.listdir(input_path)
#local_images = local_images[:5]

ifa = pykog.ImageFaceAnalysis(_personal_access_key, _secret_access_key)

ifa.initialize(imageFileList = local_images)

#ifa_df = ifa.get(attributes = ['emotions', 'features', 'age'])
ifa_df = ifa.get(attributes = ['emotions'])


out = r"C:/Users/rasmu/Documents/Repositories/pykognition/outputEmotions/"
ifa.draw(outputPath = out)

#%%

ioa = pykog.ImageObjectAnalysis(_personal_access_key, _secret_access_key)
ioa.initialize(inputPath = input_path, imageFileList = local_images)

ioa_df = ioa.get()