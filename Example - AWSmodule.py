# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:27:09 2020

@author: rhs
"""
#%%

import yaml
import pykognition.faceAnalysis as fa
import os
import pandas as pd

#%%

#from package_name import no_ssl_verification


_root = os.getcwd()
input_path = _root + "\\inputEmotions\\"
output_path = _root + "\\outputEmotions\\"

# reading credentials
_credentials = yaml.load(open(os.getcwd() + '\\Credentials.yaml'))
_personal_access_key = _credentials['amazon']['personal_access_key']
_secret_access_key = _credentials['amazon']['secret_access_key']

#%%
local_images = os.listdir(input_path)
local_images = local_images[:10]

ifa = fa.ImageFaceAnalysis(_personal_access_key, _secret_access_key)

ifa.initialize(inputPath = input_path, imageFileList = local_images)



df = ifa.get(attributes = ['emotions', 'age', 'features'])




#%%
