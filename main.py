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


def preprocessing(df):
    prep.sex2binary(df)
    #prep.fillagewithmean(df)
    prep.fill_na_age(df)
    prep.fillembarked3(df)
    df = prep.extractTitles(df)
    #remove outlier for Fare
    df = df[df.Fare != max(df.Fare)]
    return prep.convert2onehot(df, 'Sex', 'Embarked', 'Title'), df
    #return df.drop(['Sex', 'Embarked', 'Title'], axis=1)

def explore(df):
    expl.variable_info(df)
    expl.plot_heatmap(df)
    expl.plot_overlay_hist(df)

def main():

    # set cmd panda view and import data
    pd.set_option('display.max_columns', None)
    alldata = prep.import2df('data/train.csv')
    #alldata = prep.import2df('data/train.csv')

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
    #knn_prediction = methods.KNN_predict(X_train, Y_train, X_test, 30)
    #print("------Results for KNN--------")
    #print(knn_prediction)
    #print("KNN accuracy:",eval.accuracy(knn_prediction,Y_test))
    #print("KNN expected loss:",eval.expected_loss(Y_test,knn_prediction,eval.confusion_matrix(knn_prediction,Y_test)))
    #print("Cross entropy error:",eval.cross_entropy_error(Y_test,methods.KNN_prob(X_train, Y_train, X_test, 30))) #(NAN error)
    #print("KNN misclassification error:", eval.misclassification_error(Y_test,knn_prediction))
    #print("True Positives:", eval.recall(eval.confusion_matrix(knn_prediction,Y_test)))
    #print("False Negatives:", eval.false_positive_ratio(eval.confusion_matrix(knn_prediction,Y_test)))

    # # Run random forest model with 100 n_estimators
    # forest_prediction = methods.randomForest(X_train, Y_train, X_test, n_estimators=100)
    # print("------Results for Random Forest--------")
    # print(forest_prediction)
    # print("Random Forest accuracy:", eval.accuracy(forest_prediction,Y_test))
    # print("Random Forest expected loss:",eval.expected_loss(Y_test,forest_prediction,eval.confusion_matrix(forest_prediction,Y_test)))
    # print("Random Forest misclassification error:", eval.misclassification_error(Y_test,forest_prediction))
    # print("True Positives:", eval.recall(eval.confusion_matrix(forest_prediction,Y_test)))
    # print("False Negatives:", eval.false_positive_ratio(eval.confusion_matrix(forest_prediction,Y_test)))

    # # Run Logistic Regression model
    # logistic = methods.LogisticRegression()
    # logistic_prediction = logistic.weighting(X_train,Y_train, X_test)
    # print("------Results for Logistic Regression--------")
    # print(logistic_prediction)
    # print("Logistic Regression accuracy:",eval.accuracy(logistic_prediction,Y_test))
    # print("Logistic Regression expected loss:",eval.expected_loss(Y_test,logistic_prediction,eval.confusion_matrix(logistic_prediction,Y_test)))
    # print("Logstic Regression misclassification error:", eval.misclassification_error(Y_test,logistic_prediction))
    # print("True Positives:", eval.recall(eval.confusion_matrix(logistic_prediction,Y_test)))
    # print("False Negatives:", eval.false_positive_ratio(eval.confusion_matrix(logistic_prediction,Y_test)))

    # # Running Fisher LDA
    fisher_pred = methods.fishers_LDA(X_train, Y_train, X_test)
    plt.show()
    print("------Results for Fisher LDA--------")
    print(fisher_pred)
    print("LDA accuracy:", eval.accuracy(fisher_pred,Y_test))
    print("LDA expected loss:",eval.expected_loss(Y_test, fisher_pred, eval.confusion_matrix(fisher_pred, Y_test)))
    print("LDA misclassification error:", eval.misclassification_error(Y_test,fisher_pred))
    print("True Positives:", eval.recall(eval.confusion_matrix(fisher_pred,Y_test)))
    print("False Negatives:", eval.false_positive_ratio(eval.confusion_matrix(fisher_pred,Y_test)))

    # # Model accuracies vs % Training Sample
    # plt.figure()
    # eval.accuracy_v_sample(x,y,model="logistic")
    # eval.accuracy_v_sample(x,y,model="forest")
    # eval.accuracy_v_sample(x,y,model="knn")
    # eval.accuracy_v_sample(x,y,model="fisher") 
    # plt.legend()
    
    # # Confusion matrices
    # eval.confusion_matrix(logistic_prediction,Y_test)
    # plt.figure()
    # sns.heatmap(eval.confusion_matrix(logistic_prediction, Y_test), annot=True)
    # plt.title("Logistic Regression confusion matrix")
    # eval.confusion_matrix(forest_prediction,Y_test)
    # plt.figure()
    # sns.heatmap(eval.confusion_matrix(forest_prediction, Y_test), annot=True)
    # plt.title("Random Forest confusion matrix")
    # eval.confusion_matrix(knn_prediction,Y_test)
    # plt.figure()
    # sns.heatmap(eval.confusion_matrix(knn_prediction, Y_test), annot=True)
    # plt.title("KNN confusion matrix")
    # eval.confusion_matrix(fisher_pred,Y_test)
    # plt.figure()
    # sns.heatmap(eval.confusion_matrix(fisher_pred, Y_test), annot=True)
    # plt.title("LDA confusion matrix")
    
    # # Timing each model
    # print("------Time taken for each model to train and predict------")
    # eval.model_timing(X_train,Y_train,X_test)

    # # K-Fold Cross Validation (NOT WORKING FOR SOME REASON)
    # cv_val = eval.kfoldCV(alldata, f=3, k=30, model="knn") #KNN
    # print("KNN Result from each fold:", cv_val)
    # print("Mean:", stats.mean(cv_val))
    # print("Standard deviation:", stats.stdev(cv_val))
    # cv_val = eval.kfoldCV(alldata, f=20, k=30, model="fisher") #LDA
    # print("LDA Result from each fold:", cv_val)
    # print("Mean:", stats.mean(cv_val))
    # print("Standard deviation:", stats.stdev(cv_val))
    # cv_val = eval.kfoldCV(alldata, f=3, k=30, model="forest") #Forest
    # print("Random Forest Result from each fold:", cv_val)
    # print("Mean:", stats.mean(cv_val))
    # print("Standard deviation:", stats.stdev(cv_val))
    # cv_val = eval.kfoldCV(alldata, f=3, k=30, model="logistic") #Logistic
    # print("Logistic regression Result from each fold:", cv_val)
    # print("Mean:", stats.mean(cv_val))
    # print("Standard deviation:", stats.stdev(cv_val))

    # # K-Fold mean accuracy vs number of folds
    # plt.figure()
    # eval.accuracy_v_fold(alldata, model="knn")
    # eval.accuracy_v_fold(alldata, model="forest")
    # #eval.accuracy_v_fold(alldata, model="logistic")
    # eval.accuracy_v_fold(alldata, model="fisher")
    # plt.legend()

    # ROC Curves
    # eval.ROC_curves(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, model="knn")
    # eval.ROC_curves(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, model="forest")
    # eval.ROC_curves(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, model="logistic") #(VERYYYYYY SLOW)
    # eval.ROC_curves(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, model="fisher")

    # fisher_pred = methods.fishers_LDA(X_train, Y_train, X_test)
    # print(fisher_pred)
    #plt.show()


    # TODO score (don't submit this we need to do our own evaluations) add evaluation techniques here
    ### WE SHOULD SAVE ALL THE FIGURES IN ONE FILE ###


if __name__ == "__main__":
    main()