
Welcome to pykognition's documentation!
=========================================

For installation and description visit the `github-repo <https://github.com/schmokel/pykognition>`_.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   pykognition


Example usage:
--------------

.. code-block:: python

   import yaml
   import pykognition as pykog
   import os
   import pandas as pd

   _root = os.getcwd()
   _ifa_datadir = os.path.join(_root, "inputEmotions", "")
   input_path = "/inputEmotions"
   # reading credentials
   _credentials = yaml.load(open(os.path.join(_root, 'Credentials.yaml')))
   _personal_access_key = _credentials['amazon']['personal_access_key']
   _secret_access_key = _credentials['amazon']['secret_access_key']

   #The simpler os.listdir           
   #local_images = os.listdir(input_path)

   # or recursively reading files and files in subfolders in 'input_path'
   local_images = []
   for root, dirs, files in os.walk(_root + input_path):
      for file in files:
         if(file.endswith(".jpg")):
               local_images.append(os.path.join(root, file))
               



   ######################################
   #                                    #
   # Face and emotion detection-class   #
   #                                    #
   ######################################

   #input PAK and SAK
   ifa = pykog.ImageFaceAnalysis(_personal_access_key, _secret_access_key)

   #initialize, that is actually running, the analysis
   ifa.initialize(imageFileList = local_images)

   #Extract  all attributes
   ifa_df = ifa.get(attributes = ['emotions', 'features', 'age'])

   #or just emotions
   ifa_df = ifa.get(attributes = ['emotions'])


   #get images with face-boxes
   help(ifa.draw)
   out = _root + "/outputEmotions/"
   ifa.draw(outputPath = out, conf_threshold = 80, font_size = 24)

   #Saving the data to a spreadsheet, simply do:
   pd.to_csv(ifa.get(attributes = ['emotions', 'features', 'age']))

   #%%

   #################################
   #                               #
   # Object detection-class (BETA) #
   #                               #
   #################################

   ioa = pykog.ImageObjectAnalysis(_personal_access_key, _secret_access_key)
   ioa.initialize(imageFileList = local_images)

   ioa_df = ioa.get()





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

