# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:11:45 2020

@author: rhs
"""
import boto3

class BaseImageDataHandler:
    
    def __init__(self, personal_acces_key, secret_access_key):
        self.personal_acces_key = personal_acces_key
        self.secret_access_key = secret_access_key

    def client(self, region):
        return boto3.client('rekognition', region, 
                            aws_access_key_id = self.personal_acces_key,
                            aws_secret_access_key = self.secret_access_key)
    

        
    
    
        
        
        
        
