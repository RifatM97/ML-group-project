import seaborn as sns
import evaluation as eval
import preprocessing as prep
import numpy as np
import matplotlib.pyplot as plt
import methods
from statistics import mean


def model_performance(model_prediction, Y_test, runtime):
    print("Accuracy:",round(eval.accuracy(model_prediction,Y_test), 4))
    print("Expected loss:",round(eval.expected_loss(Y_test,model_prediction,eval.confusion_matrix(model_prediction,Y_test)), 4))
    print("Misclassification error:", round(eval.misclassification_error(Y_test,model_prediction), 4))
    print("True Positives:", round(eval.recall(eval.confusion_matrix(model_prediction,Y_test)), 4))
    print("False Negatives:", round(eval.false_positive_ratio(eval.confusion_matrix(model_prediction,Y_test)), 4))
    print("Model run time:","--- %s seconds ---" % round(runtime, 4))


def plot_cm_comparison(forest_prediction, knn_prediction, fisher_prediction, logistic_prediction, Y_test):
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

    
# Accuracy vs Training sample size
def accuracy_v_sample(x,y):
    """Function plots the accuracy against the sample size of different models. The inputs is the chosen 
    model. The output is the plot"""
    size = np.arange(0.11,0.9,0.1)
    KNN_accuracy_score = []
    forest_accuracy_score = []
    logistic_accuracy_score = []
    fisher_accuracy_score = []
    for i in size:
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test=prep.partition(x,y,train_portion=i)
     
        KNN_predict = methods.KNN_predict(X_train, Y_train, X_test, 30)
        KNN_accuracy_score.append(eval.accuracy(KNN_predict,Y_test))

        forest_predict = methods.randomForest(X_train, Y_train, X_test,100)
        forest_accuracy_score.append(eval.accuracy(forest_predict,Y_test))

        logistic = methods.LogisticRegression()
        logistic_predict = logistic.weighting(X_train, Y_train, X_test)
        logistic_accuracy_score.append(eval.accuracy(logistic_predict,Y_test))

        fisher_predict = methods.fishers_LDA(X_train, Y_train, X_test)
        fisher_accuracy_score.append(eval.accuracy(fisher_predict,Y_test))

    # plotting routine
    plt.figure()
    plt.plot(size, KNN_accuracy_score, label="KNN")
    plt.plot(size, forest_accuracy_score, label="Random Forest")
    plt.plot(size, logistic_accuracy_score, label="Logistic")
    plt.plot(size, fisher_accuracy_score, label="Fishers LDA")
    plt.xlabel("Training sample proportion")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Training Sample")
    plt.legend()


    # Accuracy vs Folds
def accuracy_v_fold(x):
    """Function takes a chosen model to plot an accuracy vs number of folds plot. The accuracies are
    the average values for the all the accuracies from each fold."""

    logistic_cross_vals = []
    knn_cross_vals = []
    forest_cross_vals = []
    fisher_cross_vals = []
    folds = np.arange(2, 10, 2)
    for i in folds:       
        logistic_cv_val = eval.kfoldCV(x, f=i, model="logistic")
        logistic_cross_vals.append(mean(logistic_cv_val))
        knn_cv_val = eval.kfoldCV(x, f=i, k=30, model="knn")
        knn_cross_vals.append(mean(knn_cv_val))
        forest_cv_val = eval.kfoldCV(x, f=i, n_estimators=100, model="forest")
        forest_cross_vals.append(mean(forest_cv_val))
        fisher_cv_val = eval.kfoldCV(x, f=i, model="fisher")
        fisher_cross_vals.append(mean(fisher_cv_val))
            
    # plotting routine
    plt.figure()
    plt.plot(folds, logistic_cross_vals, label="Logistic Regression")
    plt.plot(folds, knn_cross_vals, label="KNN")
    plt.plot(folds, forest_cross_vals, label="Random Forest")
    plt.plot(folds, fisher_cross_vals, label="FishersLDA")
    plt.xlabel("Folds")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs K-folds")
    plt.legend()
    plt.show()