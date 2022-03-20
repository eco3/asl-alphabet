# Learn the American Manual Alphabet (AMA)
## Introduction
//TODO 

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

<a href="https://google.github.io/mediapipe/solutions/hands#hand-landmark-model">
    <img title="21 hand landmarks" alt="21 hand landmarks" src="docs/hand_landmarks.png">
</a>

The figure above shows the resulting hand landmarks [(MediaPipe Hands)](https://google.github.io/mediapipe/solutions/hands#hand-landmark-model).

This project includes two script to extract landmarks from either image- or video-files.
You can set the number of workers, to accelerate the extraction.
Every worker processes one letter in the dataset and yields a CSV file.

If the extraction encounters an image or video with a left hand, it mirrors the x-axis of the landmarks, so it behaves like a right hand.

These resulting 26 files (A.csv, B.csv, ..., Z.csv) then can be merged into one single CSV file and used for training a model.

## Training
This project includes Jupyter Notebooks to train two different models.
Both notebooks take the same extracted dataset CSV file.

* [train_catboost.ipynb](train/train_catboost.ipynb) trains a CatBoostClassifier.
* [train_neuralnetwork.ipynb](train/train_neuralnetwork.ipynb) trains a Multilayer perceptron using TensorFlow 2.

The CatBoostClassifier converges quickly and yields great accuracy.
However, while developing this project, there was this idea to include a model into a single webpage,
ideally with no Python backend. So I decided to train a Multilayer perceptron with TensorFlow. The trained
model then can be converted for the [TensorFlow.js](https://www.tensorflow.org/js) library and included directly in 
JavaScript without the need of a Python backend server.

## Local inference
You can run your trained models by either running [run_asl_catboost.py](run_asl_catboost.py) or [run_asl_neuralnetwork.py](run_asl_neuralnetwork.py).

## Web Demo
//TODO

### Dependencies
The following dependencies are used for the web demo:

* [@mediapipe/camera_utils](https://www.npmjs.com/package/@mediapipe/camera_utils)
* [@mediapipe/drawing_utils](https://www.npmjs.com/package/@mediapipe/drawing_utils)
* [@mediapipe/hands](https://www.npmjs.com/package/@mediapipe/hands)
* [@tensorflow/tfjs](https://www.npmjs.com/package/@tensorflow/tfjs)
* [splitting](https://www.npmjs.com/package/splitting)
* [Bootstrap](https://getbootstrap.com/) - as CDN
* [Gallaudet TrueType](https://www.lifeprint.com/asl101/pages-layout/gallaudettruetypefont.htm) - a beautiful font displaying the letter signs 

The modules are compiled using [webpack.js.](https://webpack.js.org/), the source files can be found [here](https://github.com/eco3/asl-alphabet/tree/demo/demo).
