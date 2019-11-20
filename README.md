# Experimental evaluation of feature extraction techniques for animal breed recognition
Introduction To Computer Vision Project

## Project Description
In this project our aim is to hand-craft different feature descriptors/detectors, and then conduct an exhaustive analysis of their performance differences through evaluation on the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).

Our pipeline consists of two parts: Feature descriptors and Classifiers.

### Feature descriptors
We have hand-written 3 types of feature descriptors: HOG, FAST, SIFT. 

### Classifiers
We have used cv2 SVM and CatBoost library.

## Project structure
- Folder _dog_dataset_ contains Stanford Dogs Dataset. It contains 120 folders (one for each class, i.e. breed).
Each folder has a name of following structure:
    - n\[unique_number]-\[breed_name]
- Folder _commons_ contains functions that are used in all feature descriptors.
    - custom_pickle_loader: functions to save/load huge files as pickles
    - dataset_parser: function for reading Dogs Dataset
    - svm: SVM classifier
- Folders _HOG_, _FAST_ and _SVM_ contain all code required to train and test the feature descriptors.
