import random
from statistics import stdev
from statistics import mean
import numpy as np
import methods
import matplotlib.pyplot as plt
import preprocessing as prep
import methods
import random2

# Partitioning the input data into Train-Validation-Testing sets


# Creating a function for to find the accuracy 
def accuracy(Y_predict, Y):
    """Function calculates the accuracy of the prediction"""
    
    assert len(Y) == len(Y_predict)
    correct = sum(Y_predict == Y)
    return correct/len(Y)


# Confusion matrix evaluation 
def confusion_matrix(Y_predict, Y):
    """Function takes the model predictions and actual values to find accuracy"""

    K = len(np.unique(Y))
    cm = np.zeros((K,K))
    for i in range(len(Y)):
        cm[Y[i]][Y_predict[i]] += 1 
    return cm

def accuracy_v_param(X_train,Y_train,X_test,Y_test):

    K_values = np.arange(1,51)
    accuracy_score = []
    for k in K_values: # from K=1 to K=20
        y_predict = methods.KNN_predict(X_train, Y_train, X_test, k)
        accuracy_score.append(accuracy(y_predict,Y_test))
        
    plt.figure()
    plt.plot(K_values, accuracy_score)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs number of K neighbours")

# Accuracy vs Training sample size
def accuracy_v_sample(x,y,model="knn"):
    """Function plots the accuracy against the sample size of different models. The inputs is the chosen 
    model. The output is the plot"""
    size = np.arange(0.1,0.9,0.1)
    accuracy_score = []
    for i in size:
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test=prep.partition(x,y,train_portion=i)
        
        if model == "knn":
            y_predict = methods.KNN_predict(X_train, Y_train, X_test, 30)
            accuracy_score.append(accuracy(y_predict,Y_test))
            
        elif model == "forest":
            y_predict = methods.randomForest(X_train, Y_train, X_test,100)
            accuracy_score.append(accuracy(y_predict,Y_test))
            
        elif model == "logistic":
            logistic = methods.LogisticRegression()
            y_predict = logistic.weighting(X_train, Y_train, X_test)
            accuracy_score.append(accuracy(y_predict,Y_test))

        elif model == "fisher":
            y_predict = methods.fishers_LDA(X_train, Y_train, X_test)
            accuracy_score.append(accuracy(y_predict,Y_test))
                                  
    # plotting routine
    plt.plot(size, accuracy_score,label=model)
    plt.xlabel("Training sample")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Training Sample")

# Precision of the predictions
def precision(cm):
    """The ratio of correct positive predictions to the total predicted positives."""

    return cm[1][1]/(cm[1][1] + cm[0][1])

# True positives 
def recall(cm):
    """The ratio of correct positive predictions to the total positives examples. 
    This is also called the true positive rate."""
    return cm[1][1]/(cm[1][1] + cm[1][0])
  
# False positives
def false_positive_ratio(cm):
    """The false positive rate is the ratio between the false positives
    the total number of actual negative events"""

    return cm[0][1]/(cm[0][1] + cm[0][0])

# Expected error evaluation function
def expected_loss(targets, predicts, lossmtx):
    """
    How close predicted values are to the true values.
    ----------
    targets - The actual survival values
    predicts - the predictions of the survival
    lossmtx - confusion matrix

    Returns
    -------
    error - An estimate of the expected loss between true and predicted target
    """

    # flatten both arrays and ensure they are array objects
    targets = np.array(targets).flatten()
    predicts = np.array(predicts).flatten()
    class0 = (targets == 0) #dead
    class1 = np.invert(class0)
    predicts0 = (predicts == 0) 
    predicts1 = np.invert(predicts0)
    class0loss = lossmtx[0,0]*np.sum(class0 & predicts0) + lossmtx[0,1]*np.sum(class1 & predicts1)
    class1loss = lossmtx[1,0]*np.sum(class0 & predicts0) + lossmtx[1,1]*np.sum(class1 & predicts1)
    N = len(targets)
    error = (class0loss + class1loss)/N
    return error

# Cross entropy loss evaluation function (NEEDS TO BE REVIEWED AS IT GIVES NAN ERROR)
def cross_entropy_error(targets, predict_probs):
    """
    Evaluate how closely predicted values match the true
    values in a cross-entropy sense.
    ----------
    targets - The actual survival values
    predicts - the predictions of the survival
    lossmtx - confusion matrix

    Returns
    -------
    error - The cross-entropy error between true and predicted target
    """
    # flatted
    targets = np.array(targets).flatten()
    predict_probs = np.array(predict_probs[:,1]).flatten()
    N = len(targets)
    error = - np.sum(targets*np.log(predict_probs) + (1-targets)*np.log(1-predict_probs))/N
    return error

### misclassification error function

def misclassification_error(targets, predicts):
    """Function finds the minimum-misclassification error between true and predicted target. 
    It cant be considered as 1 minus the accuracy. """

    # flatten both arrays and ensure they are array objects
    targets = np.array(targets).flatten()
    predicts = np.array(predicts).flatten()
    N = targets.size
    error = 1 - np.sum(targets == predicts)/N
    return error

### K-fold cross validation function

def cross_validation_split(dataset, folds):
    """Function splits the data in chosen folds. The output is splitted data"""

    dataset_split = []
    df_copy = dataset
    fold_size = int(df_copy.shape[0] / folds)
    
    # for loop to save each fold
    for i in range(folds):
        fold = []
        # while loop to add elements to the folds
        while len(fold) < fold_size:
            # select a random element
            r = random2.randrange(df_copy.shape[0])
            # determine the index of this element 
            index = df_copy.index[r]
            # save the randomly selected line 
            fold.append(df_copy.loc[index].values.tolist())
            # delete the randomly selected line from
            # dataframe not to select again
            df_copy = df_copy.drop(index)
        # save the fold     
        dataset_split.append(np.asarray(fold))
        
    return dataset_split 
    
def kfoldCV(dataset, f=5, k=30, n_estimators=100, model="knn"):
    """Function runs chosen model into each fold and tests the model on different 
    sections. Inputs is the chosen dataset, number of folds, model name and model parameters.
    The output is an array of accuracy values for each fold."""

    data=cross_validation_split(dataset,f)
    result=[]
    # determine training and test sets 
    for i in range(f):
        r = list(range(f))
        r.pop(i)
        for j in r :
            if j == r[0]:
                cv = data[j]
            else:    
                cv=np.concatenate((cv,data[j]), axis=0)
        
        # apply the selected model
        # default is logistic regression
        if model == "logistic":
            logistic = methods.LogisticRegression()
            test = logistic.weighting(cv[:,0:4],cv[:,4],data[i][:,0:4])
        elif model == "knn":
            test = methods.KNN_predict(cv[:,0:4],cv[:,4],data[i][:,0:4],k)
        elif model == "forest":
            test = methods.randomForest(cv[:,0:4],cv[:,4],data[i][:,0:4],n_estimators)
        elif model == "fisher":
            test = methods.fishers_LDA(cv[:,0:4],cv[:,4],data[i][:,0:4])
            
        # calculate accuracy    
        acc=(test == data[i][:,4]).sum()
        result.append(acc/len(test))
        
    return result

# Accuracy vs Folds
def accuracy_v_fold(x,model="knn"):
    """Function takes a chosen model to plot an accuracy vs number of folds plot. The accuracies are
    the average values for the all the accuracies from each fold."""

    cross_vals = []
    folds = np.arange(2,20)
    for i in folds:
        
        if model == "logistic":
            cv_val = kfoldCV(x, f=i, k=5, model="logistic")
            cross_vals.append(mean(cv_val))
        elif model == "knn":
            cv_val = kfoldCV(x, f=i, k=5, model="knn")
            cross_vals.append(mean(cv_val))
        elif model == "forest":
            cv_val = kfoldCV(x, f=i, k=5, model="forest")
            cross_vals.append(mean(cv_val))
        elif model == "forest":
            cv_val = kfoldCV(x, f=i, k=5, model="fisher")
            cross_vals.append(mean(cv_val))
            
    # plotting routine
    plt.plot(folds, cross_vals)
    plt.xlabel("Folds")
    plt.ylabel("Accuracy")

# Model timing 
def model_timing(X_train, Y_train, X_test):
    """Time taken for each model to train on training data and predict on testing data"""
    import time
    
    # logistic regression time
    start_time = time.time()
    logistic = methods.LogisticRegression()
    logistic_prediction = logistic.weighting(X_train, Y_train, X_test)
    print("Logistic Regression:","--- %s seconds ---" % (time.time() - start_time))

    # knn model time
    start_time = time.time()
    knn_prediction = methods.KNN_predict(X_train, Y_train, X_test,30)
    print("KNN:","--- %s seconds ---" % (time.time() - start_time))

    # random forest model time
    start_time = time.time()
    forest_prediction = methods.randomForest(X_train, Y_train, X_test,n_estimators=100)
    print("Forest:","--- %s seconds ---" % (time.time() - start_time))

    # Fisher's LDA model time
    start_time = time.time()
    fisher_prediction = methods.fishers_LDA(X_train, Y_train, X_test)
    print("LDA:","--- %s seconds ---" % (time.time() - start_time))



# KNN threshold 
def KNN_threshold(X_train, Y_train, x, threshold=0.5):
    """Function uses threshold input for model prediction"""

    y_probs = methods.KNN_prob(X_train, Y_train, x, 30)
    predictions = y_probs >= threshold
    return np.multiply(predictions[:,1],1)

# Forest threshold 
def forest_threshold(X_train, Y_train, x, threshold=0.5):
    """Function uses threshold input for model prediction"""

    y_probs = methods.randomForest_prob(X_train,Y_train, x, 100)
    predictions = y_probs >= threshold
    return np.multiply(predictions[:,1],1)
    
# Calculate the recall and false positive rate for threshold options
def get_roc(X_train, Y_train, x, y,model="knn"):
    tpr = []
    fpr = []
    # Define decision thresholds between 0-1
    thresholds = np.linspace(0,1, 400)
    for threshold in thresholds:

        if model == "knn":
            Y_predict = KNN_threshold(X_train, Y_train, x,threshold=threshold)
            cm = confusion_matrix(Y_predict, y)
            tpr.append(recall(cm))
            fpr.append(false_positive_ratio(cm))

        elif model == "forest":
             Y_predict = forest_threshold(X_train, Y_train, x,threshold=threshold)
             cm = confusion_matrix(Y_predict, y)
             tpr.append(recall(cm))
             fpr.append(false_positive_ratio(cm))
            
        elif model == "logistic":
            logistic = methods.LogisticRegression()
            Y_predict = logistic.weighting(X_train, Y_train, x, threshold=threshold)
            cm = confusion_matrix(Y_predict, y)
            tpr.append(recall(cm))
            fpr.append(false_positive_ratio(cm))
        
        elif model == "fisher":
             Y_predict = fisher_threshold(X_train, Y_train, x,threshold=threshold)
             cm = confusion_matrix(Y_predict, y)
             tpr.append(recall(cm))
             fpr.append(false_positive_ratio(cm))
            
    return fpr, tpr
  
# Cutoff point of the best threshold by maximising the true positive rate and minimising the false positive rate
def get_cutoff(fpr, tpr):
    # Define decision thresholds between 0-1
    thresholds = np.linspace(0,1, 400)
    optimal_idx = np.argmax(np.array(tpr) - np.array(fpr))
    optimal_threshold = thresholds[optimal_idx]
    return optimal_idx, optimal_threshold

def ROC_curves(X_train, Y_train, X_valid, Y_valid, X_test, Y_test,model="forest"):
    """ROC curves for each model"""

    train_roc = get_roc(X_train, Y_train, X_train, Y_train,model)
    valid_roc = get_roc(X_train, Y_train, X_valid, Y_valid,model)
    test_roc = get_roc(X_train, Y_train, X_test, Y_test,model)

    train_cutoff = get_cutoff(train_roc[0], train_roc[1])   
    valid_cutoff = get_cutoff(valid_roc[0], valid_roc[1])
    test_cutoff = get_cutoff(test_roc[0], test_roc[1])

    plt.figure()
    plt.plot(train_roc[0], train_roc[1], label="Train", c='r')
    plt.plot(valid_roc[0], valid_roc[1], label="Valid", c='b')
    plt.plot(test_roc[0], test_roc[1], label="Test", c='g', linestyle='dashed')

    plt.scatter(train_roc[0][train_cutoff[0]], train_roc[1][train_cutoff[0]], label="Train cutoff: {}".format(train_cutoff[1]), c='r')
    plt.scatter(valid_roc[0][valid_cutoff[0]], valid_roc[1][valid_cutoff[0]], label="Valid cutoff: {}".format(valid_cutoff[1]), c='b')
    plt.scatter(test_roc[0][test_cutoff[0]], test_roc[1][test_cutoff[0]], label="Test cutoff: {}".format(test_cutoff[1]), c='g', linestyle='dashed')
    plt.title(model)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.legend()


