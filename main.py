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
    prep.sex2binary(df)
    #prep.fillagewithmean(df)
    prep.fill_na_age(df)
    prep.fillembarked3(df)
    df = prep.extractTitles(df)
    #remove outlier for Fare
    df = df[df.Fare != max(df.Fare)]
    return prep.convert2onehot(df, 'Sex', 'Embarked', 'Title'), df

def explore(df):
    expl.variable_info(df)
    expl.plot_heatmap(df)
    expl.plot_overlay_hist(df)

    
def main():

    # set cmd panda view and import data
    pd.set_option('display.max_columns', None)
    alldata = prep.import2df(r'C:\Users\user\ML-group-project.git\ML-group-project\data\train.csv')

    # fill in missing data and convert categories to one hot
    alldata, alldata_discrete = preprocessing(alldata)

    #Produce some exploratory plots of the data
    #explore(alldata_discrete)

    # split 80:10:10 train-validation-test
    x = alldata.drop('Survived', axis=1)
    y = alldata["Survived"]
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = prep.partition(x, y)
    
    # Checking KNN vs number of K-neighbors to identify optimum K 
    #eval.accuracy_v_param(X_train,Y_train,X_test,Y_test)
    # Running the KNN model
    #knn_start_time = time.time()
    #knn_prediction = methods.KNN_predict(X_train, Y_train, X_test, 30)
    #knn_runtime = time.time() - knn_start_time
    #print("------Results for KNN--------")
    #comp.model_performance(knn_prediction, Y_test, knn_runtime)


    # # Run random forest model with 100 n_estimators
    #forest_start_time = time.time()
    #forest_prediction = methods.randomForest(X_train, Y_train, X_test, n_estimators=100)
    #forest_runtime = time.time() - forest_start_time
    #print("------Results for Random Forest--------")
    #comp.model_performance(forest_prediction, Y_test, forest_runtime)

    # # Run Logistic Regression model
    #logistic_start_time = time.time()
    #logistic = methods.LogisticRegression()
    #logistic_prediction = logistic.weighting(X_train,Y_train, X_test)
    #logistic_runtime = time.time() - logistic_start_time
    #print("------Results for Logistic Regression--------")
    #comp.model_performance(logistic_prediction, Y_test, logistic_runtime)

    # # Running Fisher LDA
    #fisher_start_time = time.time()
    #fisher_prediction = methods.fishers_LDA(X_train, Y_train, X_test, plot_hist = "yes")
    #fisher_runtime = time.time() - fisher_start_time
    #plt.show()
    #print("------Results for Fisher LDA--------")
    #comp.model_performance(fisher_prediction, Y_test, fisher_runtime)

    # # Model accuracies vs % Training Sample
    #Takes aproximately 3 minutes to run
    #comp.accuracy_v_sample(x,y)
    #plt.show()

    # # Model accuracies vs % Training Sample
    eval.loss_v_sample(x,y)
    plt.show()
    
    # # Confusion matrices
    # TODO Currently the scale is not the same on all 4 plots. 
    #   This can be modified so that the colours correspond to the same values for all 4 plots
    #comp.plot_cm_comparison(forest_prediction, knn_prediction, fisher_prediction, logistic_prediction, Y_test)
    
    # # K-Fold Cross Validation 
    #cv_val = eval.kfoldCV(alldata, f=3, model="knn", print_result="yes")
    #cv_val = eval.kfoldCV(alldata, f=3, model="fisher", print_result="yes")
    #cv_val = eval.kfoldCV(alldata, f=3, model="forest", print_result="yes") 
    #cv_val = eval.kfoldCV(alldata, f=3, model="logistic", print_result="yes") 

    # # K-Fold mean accuracy vs number of folds
    #TODO This takes too long. May have to drop it
    comp.accuracy_v_fold(alldata)


    ### TODO WE SHOULD SAVE ALL THE FIGURES IN ONE FILE ###


if __name__ == "__main__":
    main()


