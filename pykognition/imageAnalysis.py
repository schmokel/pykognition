#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Apr 10 13:10:49 2020

@author: rhs
"""

import boto3
import pandas as pd
from functools import reduce
import os 
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont
import re

from .base import BaseImageDataHandler



class ImageFaceAnalysis(BaseImageDataHandler):
    """
     The ImageFaceAnalysis class implements AWS Rekognition API endpoint for detecting and analyzing faces in images.
     For more information about the API, visit: https://docs.aws.amazon.com/rekognition/latest/dg/faces.html

    """
    
    
    def __init__(self, personal_acces_key, secret_access_key):

        super().__init__(personal_acces_key, secret_access_key)
        self.funcDict = {
            "emotions": self._getEmotions,
            "age": self._getAge,
            'features': self._getFeatures}
        self.imageList = None
        self.response = None
        
    
    

    def initialize(self, imageFileList, region = 'us-east-1'):
        """
        Initializes the actual analysis. 
            
            

        Parameters
        ----------
        imageFileList : list, 
            The list of images to run through the API 
            Provide full path if images are not in same folder.
        Region : string, Default: 'us-east-1'
            Your AWS region
            

        Returns
        -------
        Class object
            
        
        Example
        -------

        """

        self.imageList = imageFileList
        self.response = [self._get_response(image, region) for image in self.imageList]

 

    
    def get(self, attributes = 'all'):

        """
        Get the face attributes in a flat data format (not-nested).

            

        Parameters
        ----------
        attributes : list ('emotions', 'age', 'features'), default = 'all'



        Returns
        -------
        Pandas DataFrame
            
        
        Example
        -------

        """

        if attributes == 'all':
            attributes = ['emotions', 'age', 'features']

        df_list = [self._dataExtractor(n) for n in attributes]
        df_list.append(self._getBaseData())
        
        return self._reduce_data(df_list = df_list, join_cols = ['imageName', 'faceID'])

        
    def getResponse(self):
        
        
        """
        Returns entire response from the AWS Rekognition API in nested format

 
        Returns
        -------
        List of dicts
            
        
        Example
        -------

        """
        return self.response

    

    def draw(self, outputPath, images = None, conf_threshold = 0, font_size = 16):
        """
        Returns boxes with face ID, emotions and confidence level


        Parameters
        ----------
        outputPath: str
            Folder location for images with boxes
        images: str, optional, Default = None
            choose subset of images by providing image-names. If none, takes all images as input
        conf_threshold: int, optional, default = 0
            Only draw boxes with confidence level above the threshold
        font_size: int, optional, default = 16
            Size of the font


        Returns
        -------
            Images with boxes

        Example
        -------

        """

        if images is None:
            iterlist = self.imageList
        else:
            iterlist = images

        clean_imageNames = [re.split(' |/|\\\\', pathNames)[-1] for pathNames in iterlist]

        
        for imageFile in range(len(iterlist)):
            with open(iterlist[imageFile], 'rb') as image:
            
                draw_image = Image.open(image)
    
                imgWidth, imgHeight = draw_image.size  
                draw = ImageDraw.Draw(draw_image) 
            id_counter = 0
            for label in self.response[imageFile]['FaceDetails']:  
                id_counter += 1  
                box = label['BoundingBox']
                left = imgWidth * box['Left']
                top = imgHeight * box['Top']
                width = imgWidth * box['Width']
                height = imgHeight * box['Height']
                        

                points = (
                    (left,top),
                    (left + width, top),
                    (left + width, top + height),
                    (left , top + height),
                    (left, top)
                
                )
                
                maxConfEmotion = max(label['Emotions'], key=lambda x:x['Confidence'])

                if int(maxConfEmotion['Confidence']) >= conf_threshold:
                    draw.line(points, fill='#00d400', width=2)
                    usr_font = ImageFont.truetype("arial.ttf", font_size)
                    text_position = (left, top)
                    box_label = "FID: {id}, {emotion}: {conf}".format(id = str(id_counter), emotion = maxConfEmotion['Type'], conf =  str(int(maxConfEmotion['Confidence'])))
                    draw.text(text_position, box_label, fill='RED', font = usr_font)
                    
                else:
                    continue

            draw_image.save(outputPath + clean_imageNames[imageFile])
        
        
            
    
    
    
    
        
    def _get_response(self, imageFile, region):
        return super().client(region = region).detect_faces(Image={
            'Bytes': open(imageFile, 'rb').read()}, Attributes = ['ALL'])
        



    def _getEmotions(self):
        holder_labels = []
        for n in range(len(self.imageList)):
            
            ## If no labels detected, still save the info:
                if len(self.response[n]['FaceDetails']) == 0:
                    temp_dict = {}
                    temp_dict["imageName"] = self.imageList[n]
                    holder_labels.append(temp_dict)   
                
                else:
                    
                    label_counter = 1
                    
                    for label in self.response[n]['FaceDetails']:
                        
                        temp_dict = {}
                        temp_dict["imageName"] = self.imageList[n]
                        temp_dict["faceID"] = label_counter
                        temp_dict["Emotion"] = max(label['Emotions'], key=lambda x:x['Confidence'])['Type']
                        temp_dict['Emotion_conf'] = max(label['Emotions'], key=lambda x:x['Confidence'])['Confidence']
                        label_counter +=1 # update for the next label
                        holder_labels.append(temp_dict)
                        
        return pd.DataFrame(holder_labels)
    
    
    def _getAge(self):
        temp = []
        
        
        for n in range(len(self.imageList)):
            if len(self.response[n]['FaceDetails']) == 0:
                temp_dict = {}
                temp_dict["imageName"] = self.imageList[n]
                temp.append(temp_dict)   
                
            else: 
                
                label_counter = 1
                
                for label in self.response[n]['FaceDetails']:
                    temp_dict = {}
                    temp_dict['imageName'] = self.imageList[n]
                    temp_dict['faceID'] = label_counter
                    temp_dict['AgeRange_low'] = label['AgeRange']['Low']
                    temp_dict['AgeRange_high'] = label['AgeRange']['High']
                    label_counter += 1
                    temp.append(temp_dict)
        
        return pd.DataFrame(temp)
    
    
    def _getFeatures(self):
            
        temp = []
        
        
        for n in range(len(self.imageList)):
            if len(self.response[n]['FaceDetails']) == 0:
                temp_dict = {}
                temp_dict["imageName"] = self.imageList[n]
                temp.append(temp_dict)   
                
            else: 
                
                label_counter = 1
                
                for label in self.response[n]['FaceDetails']:
                    temp_dict = {}
                    temp_dict['imageName'] = self.imageList[n]
                    temp_dict['faceID'] = label_counter
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
            if len(self.response[n]['FaceDetails']) == 0:
                temp_dict = {}
                temp_dict["imageName"] = self.imageList[n]
                temp.append(temp_dict)   
                
            else: 
                
                label_counter = 1
                
                for label in self.response[n]['FaceDetails']:
                    temp_dict = {}
                    temp_dict['imageName'] = self.imageList[n]
                    temp_dict['faceID'] = label_counter
                    temp_dict['faceConf'] = label['Confidence']
                    label_counter += 1
                    temp.append(temp_dict)
                    
        return pd.DataFrame(temp)
        
        


    def _dataExtractor(self, attribute):
        '''
        Indsæt basedata-handler i denne del. Slet fra de enkelte extractor-funtioner. 
        byg de andre som funktion der returnerer dict.
        '''
        return self.funcDict[attribute]()
                
    
    def _reduce_data(self, df_list, join_cols):
        return reduce(lambda left,right: pd.merge(left,right, on = join_cols), df_list)
    

    
            


class ImageObjectAnalysis(BaseImageDataHandler):

    """
     The ImageObjectAnalysis class implements AWS Rekognition API endpoint for detecting and analyzing objects and scenes in images.
     For more information about the API, visit: https://docs.aws.amazon.com/rekognition/latest/dg/labels.html

    """

    def __init__(self, personal_acces_key, secret_access_key):
        super().__init__(personal_acces_key, secret_access_key)


    def initialize(self, imageFileList, region = 'us-east-1'):
        """
        Initializes the actual analysis. 
            
            

        Parameters
        ----------
        imageFileList : list, 
            The list of images to run through the API 
            Provide full path if images are not in same folder.
        Region : string, Default: 'us-east-1'
            Your AWS region
            

        Returns
        -------
        Class object
            
        
        Example
        -------

        """
        self.imageList = imageFileList
        self.response = [self._get_response(image, region) for image in self.imageList]
        



    def get(self):
        """
        Get the objects and scenes in a flat data format (not-nested).


        Returns
        -------
        Pandas DataFrame
            
        
        Example
        -------

        """

        temp = []

        for n in range(len(self.imageList)):
        ## If no labels detected, still save the info:
            if len(self.response[n]['Labels']) == 0:
                #print ("No Labels Detected")
                temp_dict = {}
                temp_dict["imageName"] = self.imageList[n]
                temp_dict["objectID"] = None
                temp_dict["ObjectName"] = None
                temp_dict["objectConf"] = None
                temp.append(temp_dict)   
            
            else:
                
                label_counter = 1
                
                for label in self.response[n]['Labels']:
                    #print (label['Name'] + ' : ' + str(label['Confidence']))
                    temp_dict = {}
                    temp_dict["imageName"] = self.imageList[n]
                    #temp_dict["full_detect_labels_response"] = response
                    temp_dict["objectID"] = label_counter
                    temp_dict["object"] = label['Name']
                    temp_dict["objectConf"] = label['Confidence']
                    if len(label['Parents']) != 0:
                        temp_dict["parent"] = label['Parents'][0]['Name']
                    else:
                        temp_dict['parent'] = None
                    label_counter +=1 # update for the next label
                    temp.append(temp_dict)
                    
        return pd.DataFrame(temp)

        
    def _get_response(self, imageFile, region):
        return super().client(region = region).detect_labels(Image={
            'Bytes': open(imageFile,'rb').read()})

