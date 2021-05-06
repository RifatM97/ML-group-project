import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import statistics as stats
import preprocessing as prep
import evaluation as eval
import methods
import random2
import exploratory as expl
import time
import comparison as comp


def preprocessing(df):
    """Carry out preprocessing of the data to fill missing values, remove an outlier and 
        convert ordinal variables to one hot vectors"""
    prep.sex2binary(df)
    prep.fill_na_age(df)
    prep.fillembarked3(df)
    df = prep.extractTitles(df)
    #remove outlier for Fare
    df = df[df.Fare != max(df.Fare)]
    return prep.convert2onehot(df, 'Sex', 'Embarked', 'Title'), df

def explore(df):
    """This function is used to produce all exploratory plots"""
    expl.variable_info(df)
    expl.plot_heatmap(df)
    expl.plot_overlay_hist(df)

    
def main(ifname, knn=False, forest=False, logistic=False, fisher=False, model_comparison=False):

    # set cmd panda view and import data
    pd.set_option('display.max_columns', None)
    alldata = prep.import2df(ifname)

    # fill in missing data and convert categories to one hot
    alldata, alldata_discrete = preprocessing(alldata)
    
    #Produce some exploratory plots of the data
    explore(alldata_discrete)

    # split 80:10:10 train-validation-test
    x = alldata.drop('Survived', axis=1)
    y = alldata["Survived"]
    X_train, Y_train, X_test, Y_test = prep.partition(x, y)
    
    if knn == True:
        # Checking KNN vs number of K-neighbors to identify optimum K 
        eval.accuracy_v_param(X_train,Y_train,X_test,Y_test)
        # Running the KNN model
        knn_start_time = time.time()
        knn_prediction = methods.KNN_predict(X_train, Y_train, X_test, 20)
        knn_runtime = time.time() - knn_start_time
        print("------Results for KNN--------")
        comp.model_performance(knn_prediction, Y_test, knn_runtime)
        cv_val = eval.kfoldCV(alldata, f=3, model="knn", print_result=True)

    if forest == True:
        # # Run random forest model with 100 n_estimators
        forest_start_time = time.time()
        forest_prediction = methods.randomForest(X_train, Y_train, X_test, n_estimators=100)
        forest_runtime = time.time() - forest_start_time
        print("------Results for Random Forest--------")
        comp.model_performance(forest_prediction, Y_test, forest_runtime)
        cv_val = eval.kfoldCV(alldata, f=3, model="forest", print_result=True) 
    
    if logistic == True:
        # Run Logistic Regression model
        logistic_start_time = time.time()
        logistic = methods.LogisticRegression()
        logistic_prediction = logistic.weighting(X_train,Y_train, X_test)
        logistic_runtime = time.time() - logistic_start_time
        print("------Results for Logistic Regression--------")
        comp.model_performance(logistic_prediction, Y_test, logistic_runtime)
        cv_val = eval.kfoldCV(alldata, f=3, model="logistic", print_result=True) 


    if fisher == True:
        # # Running Fisher LDA
        fisher_start_time = time.time()
        fisher_prediction = methods.fishers_LDA(X_train, Y_train, X_test, plot_hist = True)
        fisher_runtime = time.time() - fisher_start_time
        print("------Results for Fisher LDA--------")
        comp.model_performance(fisher_prediction, Y_test, fisher_runtime)
        cv_val = eval.kfoldCV(alldata, f=3, model="fisher", print_result=True)

    if model_comparison == True:
        #Assess Accuracy and Expected loss against sample
        #Takes 2 minutes to run
        #comp.metric_v_sample(x,y)
        
        # Confusion matrices
        comp.plot_cm_comparison(forest_prediction, knn_prediction, fisher_prediction, logistic_prediction, Y_test)

        # K-Fold mean accuracy vs number of folds
        #This takes approximately 8 minutes 
        #comp.accuracy_v_fold(alldata)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("Please specify data file data\\train.csv") 
    else:
        # assumes that the first argument is the name of the input data/path
        if len(sys.argv) == 2:
            main(ifname=sys.argv[1], knn=True, forest=True, logistic=True, fisher=True, model_comparison=True)
        else:
            # assumes that the second argument is the name of the model to be run
            model = sys.argv[2]
            if model == "knn":
                main(ifname=sys.argv[1], knn=True)
            elif model == "forest":
                main(ifname=sys.argv[1], forest=True) 
            elif model == "logistic":
                main(ifname=sys.argv[1], logistic=True)  
            elif model == "fisher":
                main(ifname=sys.argv[1], fisher=True)            

