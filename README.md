# Learn the American Manual Alphabet (AMA) 

## Introduction
 
## Requirements
This project was done using Python 3.8. The following packages were used:

* Extracting
  * [MediaPipe](https://google.github.io/mediapipe/getting_started/python)
  * [OpenCV](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
  * [Pandas](https://pandas.pydata.org/)
  * [alive-progress](https://github.com/rsalmei/alive-progress)
* Training
  * [scikit-learn](https://scikit-learn.org/stable/)
  * [TensorFlow 2](https://www.tensorflow.org/install?hl=en) or [CatBoost](https://catboost.ai/)
  * [Joblib](https://joblib.readthedocs.io/)
* Running local inference
  * [NumPy](https://numpy.org/)

## Dataset acquirement
### Dataset sources
Two on kaggle published datasets from [SigNN Team](https://www.kaggle.com/signnteam) were used.

* [ASL Sign Language Alphabet Pictures \[Minus J, Z\]](https://www.kaggle.com/datasets/signnteam/asl-sign-language-pictures-minus-j-z)
* [ASL Sign Language Alphabet Videos \[J, Z\]](https://www.kaggle.com/datasets/signnteam/asl-sign-language-alphabet-videos-j-z)

The first one only contains images from the alphabet excluding J and Z.
The second dataset contains video files of the letters J and Z, because these signs involve movements.

### Extraction
To extract the landmarks, the solution [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) is used.
Passing an image to MediaPipe it results a list of hand landmarks.  

<img title="21 hand landmarks" alt="21 hand landmarks" src="docs/hand_landmarks.png">

The figure above shows the resulting hand landmarks [(MediaPipe Hands)](https://google.github.io/mediapipe/solutions/hands#hand-landmark-model).

This project includes two script to extract landmarks from either image- or video-files.
You can set the number of workers, to accelerate the extraction.
Every worker processes one letter in the dataset and yields a CSV file.

These resulting 26 files (A.csv, B.csv, ..., Z.csv) then can be merged into one single CSV file and used for training a model.

## Training

## Local inference

## Web Demo
### Dependencies
### Build