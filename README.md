# ML group project: The Survival of The Titanic
by Rifat Mahammod, Abid Razavi, Robert Shaw and Rajni Sandhu

## Introduction 

This project uses the passenger data from the Titanic to correctly predict the passengers who died or survived as a result of the sinking ship. The project uses a range of different supervised machine learning models to attempt at an accurate result. Different metrics are used to evaluate the results from each model in order to investigate the advantages and limitations of different classification models

## Software Implementation

The structure of the code involves five main files of the project as described below.

(1) exploration.py: This python file was created for preliminary analysis of the data. Running this file would produce printed tables and values, as well as plots of the data. This first hand analysis was done to understand the relationship between the different features. 

(2) preprocessing.py: The analysed data is then processed before machine learning models are implemented. Certain features were dropped or filled with values. Non-numerical features were mapped to numerical values.

(3) methods.py: This file contains the functions for the models used in this project.

(4) evaluation.py: This file contains the functions of the different metrics used to evaluate the models such as K-fold cross validation, confusion matrix and ROC curves.

(5) main.py: In this file, all the previous function are imported and compiled together to obtain the model predictions, accuracies and results from cross evaluation metrics.

### Machine Learning Models

Four machine learning models were used in this project. 

* Logistic Regression: TO DO

* Linear Discriminant Analysis: TO DO

* K-Nearest Neighbours: Uses the SciKit Learn API to implement the training classifier. 

* Random Forest Classifier: Uses the SciKit Learn API to implement the training classifier. 

### Code

You can download a copy of all the files in this repository by cloning the git repository:

git clone https://github.com/RifatM97/ML-group-project.git

## Reproducing the Results

To be able to run this project a working python environment is required. It is recommended to set up an environment through the Anaconda terminal using the conda packages. To run the code in full follow the steps below:

* Install required modules using modules using Pip:`pip install -r requirements.txt`
  
* Run main.py:`python main.py`