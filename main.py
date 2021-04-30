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

    
def model_performance(model_prediction, Y_test):
    print("Accuracy:",round(eval.accuracy(model_prediction,Y_test), 4))
    print("Expected loss:",round(eval.expected_loss(Y_test,model_prediction,eval.confusion_matrix(model_prediction,Y_test)), 4))
    print("Misclassification error:", round(eval.misclassification_error(Y_test,model_prediction), 4))
    print("True Positives:", round(eval.recall(eval.confusion_matrix(model_prediction,Y_test)), 4))
    print("False Negatives:", round(eval.false_positive_ratio(eval.confusion_matrix(model_prediction,Y_test)), 4))

def main():

    # set cmd panda view and import data
    pd.set_option('display.max_columns', None)
    alldata = prep.import2df('data/train.csv')

    # fill in missing data and convert categories to one hot
    alldata, alldata_discrete = preprocessing(alldata)

    #Produce some exploratory plots of the data
    explore(alldata_discrete)

    # split 80:10:10 train-validation-test
    x = alldata.drop('Survived', axis=1)
    y = alldata["Survived"]
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = prep.partition(x, y)
    
    # Checking KNN vs number of K-neighbors to identify optimum K 
    #eval.accuracy_v_param(X_train,Y_train,X_test,Y_test)

    # Running the KNN model
    knn_prediction = methods.KNN_predict(X_train, Y_train, X_test, 30)
    print("------Results for KNN--------")
    model_performance(knn_prediction, Y_test)

    # # Run random forest model with 100 n_estimators
    forest_prediction = methods.randomForest(X_train, Y_train, X_test, n_estimators=100)
    print("------Results for Random Forest--------")
    model_performance(forest_prediction, Y_test)

    # # Run Logistic Regression model
    logistic = methods.LogisticRegression()
    logistic_prediction = logistic.weighting(X_train,Y_train, X_test)
    print("------Results for Logistic Regression--------")
    model_performance(logistic_prediction, Y_test)

    # # Running Fisher LDA
    fisher_prediction = methods.fishers_LDA(X_train, Y_train, X_test, plot_hist = "yes")
    plt.show()
    print("------Results for Fisher LDA--------")
    model_performance(fisher_prediction, Y_test)

    # # Model accuracies vs % Training Sample
    eval.accuracy_v_sample(x,y)
    plt.show()
    
    # # Confusion matrices
    # TODO Currently the scale is not the same on all 4 plots. 
    #   This can be modified so that the colours correspond to the same values for all 4 plots
    #   Also would be nice to have the axis labels as 'True positive etc.' rather than 0 and 1
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Confusion matrices')
    sns.heatmap(eval.confusion_matrix(forest_prediction, Y_test), annot=True, ax=axs[0, 0], xticklabels=False)
    axs[0, 0].set_title('Random Forest')
    sns.heatmap(eval.confusion_matrix(knn_prediction, Y_test), annot=True, ax=axs[0, 1], xticklabels=False)
    axs[0, 1].set_title('KNN')
    sns.heatmap(eval.confusion_matrix(fisher_prediction, Y_test), annot=True, ax=axs[1, 0])
    axs[1, 0].set_title('Fishers LDA')
    sns.heatmap(eval.confusion_matrix(logistic_prediction, Y_test), annot=True, ax=axs[1, 1], yticklabels=False)
    axs[1, 1].set_title('Logistic Regression')
    plt.show()
    
    # # Timing each model
    # TODO Is it possible to wrap these around the functions the first time we run them rather than running again
    print("------Time taken for each model to train and predict------")
    eval.model_timing(X_train,Y_train,X_test)

    # # K-Fold mean accuracy vs number of folds
    #plt.figure()
    #eval.accuracy_v_fold(alldata, model="knn")
    #eval.accuracy_v_fold(alldata, model="forest")
    #eval.accuracy_v_fold(alldata, model="logistic")
    #eval.accuracy_v_fold(alldata, model="fisher")
    #plt.legend()

    ### TODO WE SHOULD SAVE ALL THE FIGURES IN ONE FILE ###


if __name__ == "__main__":
    main()


