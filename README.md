# Grammatical Facial Expression Recognition

This repository consists of python code for grammatical facial expression recognition computed on The Grammatical expression dataset available on UCI machine learning repository (). It makes use of a customized deep neural network architecture, designed specially for recognizing corelation between different face patterns.The entire framework, for both training and prediction is based upon Tensorflow open source library.

The repository contents are as follows:
Main code.py  : main deep learning framework for both training and prediction
Normalization.py : Used to remove any zero entities from dataset and perfrom Z score standardization on the entire dataset.
Equal_csv:  Folder consisting of csv files of various markers present in the online dataset, containing equal number of positive and negative samples. This is used by the normalization py code to preform the required pre-processing 