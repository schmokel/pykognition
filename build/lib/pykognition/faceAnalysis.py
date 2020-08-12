# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:10:49 2020

@author: rhs
"""

import boto3
import pandas as pd
from functools import reduce
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont


class ImageFaceAnalysis:
    
    
    def __init__(self, personal_acces_key, secret_access_key):
        self.personal_acces_key = personal_acces_key
        self.secret_access_key = secret_access_key
        self.funcDict = {
            "emotions": self._getEmotions,
            "age": self._getAge,
            'features': self._getFeatures}
        self.imageList = None
        self.response = None


    
    
    
    def client(self, region = 'us-east-1'):
        return boto3.client('rekognition', region, 
                            aws_access_key_id = self.personal_acces_key,
                            aws_secret_access_key = self.secret_access_key)
    
    
    def initialize(self, inputPath, imageFileList, region = 'us-east-1'):
        self.imageList = imageFileList
        self.response = [self._get_response(inputPath, image, region) for image in imageFileList]
        
        
        
    def _get_response(self, inputPath, imageFile, region = 'us-east-1'):
        return self.client(region = region).detect_faces(Image={
            'Bytes': self._open_image(inputPath, imageFile).read()}, Attributes = ['ALL'])
        

           
    def _open_image(self, inputPath, imagefile):
        return open(inputPath + imagefile, 'rb')
    
    
    
    def getResponse(self):
        return self.response



    def _getEmotions(self):
        holder_labels = []
        for n in range(len(self.imageList)):
            
            ## If no labels detected, still save the info:
                if len(self.response[n]['FaceDetails']) == 0:
                    temp_dict = {}
                    temp_dict["image_name"] = self.imageList[n]
                    holder_labels.append(temp_dict)   
                
                else:
                    
                    label_counter = 1
                    
                    for label in self.response[n]['FaceDetails']:
                        
                        temp_dict = {}
                        temp_dict["image_name"] = self.imageList[n]
                        temp_dict["face_id"] = label_counter
                        temp_dict["Emotion"] = max(label['Emotions'], key=lambda x:x['Confidence'])['Type']
                        temp_dict['Emotion_conf'] = max(label['Emotions'], key=lambda x:x['Confidence'])['Confidence']
                        label_counter +=1 # update for the next label
                        holder_labels.append(temp_dict)
                        
        return pd.DataFrame(holder_labels)
    
    
    def _getAge(self):
        temp = []
        
        
        for n in range(len(self.imageList)):
            if len(self.response[n]) == 0:
                temp_dict = {}
                temp_dict["image_name"] = self.imageList[n]
                temp.append(temp_dict)   
                
            else: 
                
                label_counter = 1
                
                for label in self.response[n]['FaceDetails']:
                    temp_dict = {}
                    temp_dict['image_name'] = self.imageList[n]
                    temp_dict['face_id'] = label_counter
                    temp_dict['AgeRange_low'] = label['AgeRange']['Low']
                    temp_dict['AgeRange_high'] = label['AgeRange']['High']
                    label_counter += 1
                    temp.append(temp_dict)
        
        return pd.DataFrame(temp)
    
    
    def _getFeatures(self):
            
        temp = []
        
        
        for n in range(len(self.imageList)):
            if len(self.response[n]) == 0:
                temp_dict = {}
                temp_dict["image_name"] = self.imageList[n]
                temp.append(temp_dict)   
                
            else: 
                
                label_counter = 1
                
                for label in self.response[n]['FaceDetails']:
                    temp_dict = {}
                    temp_dict['image_name'] = self.imageList[n]
                    temp_dict['face_id'] = label_counter
                    temp_dict['Beard'] = label['Beard']['Value']
                    temp_dict['Beard_conf'] = label['Beard']['Confidence']
                    temp_dict['Eyeglasses'] = label['Eyeglasses']['Value']
                    temp_dict['Eyeglasses_conf'] = label['Eyeglasses']['Confidence']
                    temp_dict['EyesOpen'] = label['EyesOpen']['Value']
                    temp_dict['EyeOpen_conf'] = label['EyesOpen']['Confidence']
                    temp_dict['Gender'] = label['Gender']['Value']
                    temp_dict['Gender_conf'] = label['Gender']['Confidence']
                    temp_dict['MouthOpen'] = label['MouthOpen']['Value']
                    temp_dict['MouthOpen_conf'] = label['MouthOpen']['Confidence']
                    temp_dict['Mustache'] = label['Mustache']['Value']
                    temp_dict['Mustache_conf'] = label['Mustache']['Confidence']
                    temp_dict['Smile'] = label['Smile']['Value']
                    temp_dict['Smile_conf'] = label['Smile']['Confidence']
                    temp_dict['Sunglasses'] = label['Sunglasses']['Value']
                    temp_dict['Sunglasses_conf'] = label['Sunglasses']['Confidence']
                    label_counter += 1
                    temp.append(temp_dict)
                    
        return pd.DataFrame(temp)
    
    
    def _getBaseData(self):
        temp = []
        
        for n in range(len(self.imageList)):
            if len(self.response[n]) == 0:
                temp_dict = {}
                temp_dict["image_name"] = self.imageList[n]
                temp.append(temp_dict)   
                
            else: 
                
                label_counter = 1
                
                for label in self.response[n]['FaceDetails']:
                    temp_dict = {}
                    temp_dict['image_name'] = self.imageList[n]
                    temp_dict['face_id'] = label_counter
                    temp_dict['face_conf'] = label['Confidence']
                    label_counter += 1
                    temp.append(temp_dict)
                    
        return pd.DataFrame(temp)
        


                    
   
    def get(self, attributes = []):
        
        df_list = [self._dataExtractor(n) for n in attributes]
        df_list.append(self._getBaseData())
        
        return self._reduce_data(df_list = df_list, join_cols = ['image_name', 'face_id'])

    def _dataExtractor(self, attribute):
        return self.funcDict[attribute]()
                
    
    def _reduce_data(self, df_list, join_cols):
        return reduce(lambda left,right: pd.merge(left,right, on = join_cols), df_list)

    
    
    #def getEmotions(self):
    #    return [self._getEmotions() for image in self.imageList]
        

