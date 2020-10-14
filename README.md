# pykognition
Python wrapper for AWS Rekognition API

Pykognition is a Python wrapper for the Amazon Web Service (AWS) Rekognition API, which provides industry-grade face and emotion detection. 
For facial detection, the algorithm provides a score for the predicted probability that the image includes a face (or multiple faces). Each face is categorized in a FaceDetail object, which carries a host of metadata such as predicted age, gender, and the emotion predicted to be displayed by each face. 

The emotion classifications provided by the algorithm are: Happy, Sad, Angry, Confused, Disgusted, Surprised, Calm, Fear, and Unknown. Each emotion classification is accompanied by a confidence score ranging up to 99.9%. 

Pykognition simplifies the process of classifying images with the Rekognition API. Once the researcher establishes an AWS account, they only need to insert their access tokens and an input path where the images are stored. The ‘ifa’ function (short for Image Face Analysis) sends images for classification to the Rekognition API and returns both the classification  and metadata.


## Prerequisites
* Python 3 (Is tested on 3.7 but might work on other versions)
* Access to AWS Rekogntion API


## Installation
In order to install the FbAdLibrarin you can either build from source or install via pip from the wheel distributable

In terminal/CMD:

Installing from  wheel-file
```bash
pip install "dist/pykognition-0.1.0-py3-none-any.whl"
```

Building from source:

```bash
python setup.py install 
```

You have now installed the pykognition 


## Usage

For usage and further help see the [documentation](docs/build/html/index.html)  



## Contributing
Please refer to the project's style for submitting patches and additions. In general, we follow the "fork-and-pull" Git workflow.

1. Fork the repo on GitHub
2. Clone the project to your own machine
3. Commit changes to your own branch
4. Push your work back up to your fork
5. Submit a Pull request so that we can review your changes  



## License
pykognitionis published under the GNU General Public License v3.0  
Read the license [here](LICENSE)

