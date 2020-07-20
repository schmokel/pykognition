# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:27:09 2020

@author: rhs
"""

import yaml
import pykognition.faceDetection as fd
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

fa = fd.ImageFaceAnalysis(_personal_access_key, _secret_access_key)

fa.initialize(inputPath = input_path, imageFileList = local_images)



df = fa.get(attributes = ['emotions'])




#%%