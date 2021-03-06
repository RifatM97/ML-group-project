import seaborn as sns
import evaluation as eval
import preprocessing as prep
import numpy as np
import matplotlib.pyplot as plt
import methods
from statistics import mean


def model_performance(model_prediction, Y_test, runtime):
    """This function prints a number of performance metrics to the console using the prediction made by the model"""
    print("Accuracy:",round(eval.accuracy(model_prediction,Y_test), 4))
    print("Misclassification error:", round(eval.misclassification_error(Y_test,model_prediction), 4))
    print("Expected loss:",round(eval.expected_loss(Y_test,model_prediction,eval.confusion_matrix(model_prediction,Y_test)), 4))
    print("True Positives:", round(eval.recall(eval.confusion_matrix(model_prediction,Y_test)), 4))
    print("False Negatives:", round(eval.false_positive_ratio(eval.confusion_matrix(model_prediction,Y_test)), 4))
    print("Model run time:","--- %s seconds ---" % round(runtime, 4))


def plot_cm_comparison(forest_prediction, knn_prediction, fisher_prediction, logistic_prediction, Y_test):
    """This function plots a confusion matrix for all four models"""
    fig, axs = plt.subplots(2, 2)
    #fig.suptitle('Confusion matrices')
    sns.heatmap(eval.confusion_matrix(forest_prediction, Y_test), annot=True, ax=axs[0, 0], xticklabels=False, vmax=110)
    axs[0, 0].set_title('Random Forest')
    sns.heatmap(eval.confusion_matrix(knn_prediction, Y_test), annot=True, ax=axs[0, 1], yticklabels=False, vmax=110)
    axs[0, 1].set_title('KNN')
    sns.heatmap(eval.confusion_matrix(fisher_prediction, Y_test), annot=True, ax=axs[1, 0], vmax=110)
    axs[1, 0].set_title('Fishers LDA')
    sns.heatmap(eval.confusion_matrix(logistic_prediction, Y_test), annot=True, ax=axs[1, 1], yticklabels=False, vmax=110)
    axs[1, 1].set_title('Logistic Regression')
    plt.savefig('plots\confusion_matrices.png')

    
# Accuracy vs Training sample size
def metric_v_sample(x,y, k, n_est):
    """Function plots the accuracy and expected loss against the sample size of different models. 
    The inputs is the data. 
    The output is the plots"""
    #Determine the size for each training sample
    size = np.arange(0.11,0.9,0.1)
    #Create empty lists to fill with accuracy and expected loss separately
    KNN_accuracy_score = []
    KNN_loss = []
    forest_accuracy_score = []
    forest_loss = []
    logistic_accuracy_score = []
    logistic_loss = []
    fisher_accuracy_score = []
    fisher_loss = []
    for i in size:
        #For each training sample size, split the full data into test and train data
        X_train, Y_train, X_test, Y_test=prep.partition(x,y,train_portion=i)
     
        #Run the KNN model
        KNN_predict = methods.KNN_predict(X_train, Y_train, X_test, k)
        #For each training sample size, fill the relevant list with the accuracy or expected loss
        KNN_accuracy_score.append(eval.accuracy(KNN_predict,Y_test))
        KNN_loss.append(eval.expected_loss(Y_test, KNN_predict, eval.confusion_matrix(KNN_predict, Y_test)))

        #Run the Random Forest model
        forest_predict = methods.randomForest(X_train, Y_train, X_test,n_est)
        #For each training sample size, fill the relevant list with the accuracy or expected loss
        forest_accuracy_score.append(eval.accuracy(forest_predict,Y_test))
        forest_loss.append(eval.expected_loss(Y_test, forest_predict, eval.confusion_matrix(forest_predict, Y_test)))

        #Run the Logistic Regression model
        logistic = methods.LogisticRegression()
        #For each training sample size, fill the relevant list with the accuracy or expected loss
        logistic_predict = logistic.weighting(X_train, Y_train, X_test)
        logistic_accuracy_score.append(eval.accuracy(logistic_predict,Y_test))
        logistic_loss.append(eval.expected_loss(Y_test, logistic_predict, eval.confusion_matrix(logistic_predict, Y_test)))

        #Run the Fishers LDA model
        fisher_predict = methods.fishers_LDA(X_train, Y_train, X_test)
        #For each training sample size, fill the relevant list with the accuracy or expected loss
        fisher_accuracy_score.append(eval.accuracy(fisher_predict,Y_test))
        fisher_loss.append(eval.expected_loss(Y_test, fisher_predict, eval.confusion_matrix(fisher_predict, Y_test)))

    #Plot accuracy vs training sample
    plt.figure()
    #Plot the results for each model separately
    plt.plot(size, KNN_accuracy_score, label="KNN")
    plt.plot(size, forest_accuracy_score, label="Random Forest")
    plt.plot(size, logistic_accuracy_score, label="Logistic")
    plt.plot(size, fisher_accuracy_score, label="Fishers LDA")
    plt.xlabel("Training sample proportion")
    plt.ylabel("Accuracy")
    #plt.title("Accuracy vs Training Sample")
    plt.legend()
    plt.savefig('plots\Accuracy_v_trainingsample.png')

    #Plot Expected loss vs training sample
    plt.figure()
    plt.plot(size, KNN_loss,label="KNN")
    plt.plot(size, forest_loss,label="Random Forest")
    plt.plot(size, logistic_loss,label="Logistic")
    plt.plot(size, fisher_loss,label="Fishers LDA")
    plt.xlabel("Training sample proportion")
    plt.ylabel("Expected Loss")
    #plt.title("Expected Loss vs Training Sample")
    plt.legend()
    plt.savefig('plots\Exp_loss_v_trainingsample.png')


# Accuracy vs Folds
def accuracy_v_fold(x, k, n_est):
    """Function takes a chosen model to plot an accuracy vs number of folds plot. The accuracies are
    the average values for the all the accuracies from each fold."""

    #Create empty lists to fill for each model
    logistic_cross_vals = []
    knn_cross_vals = []
    forest_cross_vals = []
    fisher_cross_vals = []
    #Specify how many folds to explore
    folds = np.arange(2, 10, 2)
    for i in folds:
        #For each number of folds run each of the models and
        #Record the mean accuracy for each number of folds

        #KNN model
        knn_cv_val = eval.kfoldCV(x, f=i, k=k, model="knn")
        knn_cross_vals.append(mean(knn_cv_val))   
        
        #Random Forest model
        forest_cv_val = eval.kfoldCV(x, f=i, n_estimators=n_est, model="forest")
        forest_cross_vals.append(mean(forest_cv_val))
        
        #Logistic Regression model
        logistic_cv_val = eval.kfoldCV(x, f=i, model="logistic")
        logistic_cross_vals.append(mean(logistic_cv_val))
        
        #Fishers LDA model
        fisher_cv_val = eval.kfoldCV(x, f=i, model="fisher")
        fisher_cross_vals.append(mean(fisher_cv_val))
            
    # Plotting the mean accuracy against the number of folds
    plt.figure()
    # Plot for each model separately
    plt.plot(folds, knn_cross_vals, label="KNN")
    plt.plot(folds, forest_cross_vals, label="Random Forest")
    plt.plot(folds, logistic_cross_vals, label="Logistic Regression")
    plt.plot(folds, fisher_cross_vals, label="FishersLDA")
    plt.xlabel("Folds")
    plt.ylabel("Accuracy")
    #plt.title("Accuracy vs K-folds")
    plt.legend()
    plt.savefig('plots\Accuracy_v_kfolds.png')
