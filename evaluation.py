import random

# Partitioning the input data into Train-Validation-Testing sets
def partition(x,y,train_portion = None):
    """ Partitions the data into train-validatin-test. 
    Inputs  - x : the titanic dataset
            - y : the survived columns 
    Outputs - The data splitted in 3 different parts
    
    """
    # Divide datasets
    random.seed(40) # same seed for consistent workflow
    
    # Decide training portion
    if train_portion is None:
        train_portion = 0.8
    else:
        train_portion = train_portion
    
    valid_portion = 0.1
    test_portion = 0.1

    # Converting the df to numpy arrays
    y = y.to_numpy()
    x = x.to_numpy()

    ### randomise the data set
    idx = [i for i in range(len(x))]
    random.shuffle(idx)
    train_idx, valid_idx, test_idx = np.split(idx,[int(train_portion*len(x)), int((train_portion + test_portion)*len(x))])

    X_train = x[train_idx]
    Y_train = y[train_idx]

    X_valid = x[valid_idx]
    Y_valid = y[valid_idx]

    X_test = x[test_idx]
    Y_test = y[test_idx]

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

# Creating a function for to find the accuracy 
def accuracy(Y_predict, Y):
    """Function calculates the accuracy of the prediction"""
    
    assert len(Y) == len(Y_predict)
    correct = sum(Y_predict == Y)
    return correct/len(Y)


# Confusion matrix evaluation 
def confusion_matrix(Y_predict, Y) :
    """Function takes the model predictions and actual values to find accuracy"""

    K = len(np.unique(Y))
    cm = np.zeros((K,K))
    for i in range(len(Y)):
        cm[Y[i]][Y_predict[i]] += 1 
    return cm

# Accuracy vs Training sample size
def accuracy_v_sample(model="knn"):
    """Function plots the accuracy against the sample size of different models. The inputs is the chosen 
    model. The output is the plot"""
    size = np.arange(0.1,0.9,0.1)
    accuracy_score = []
    for i in size:
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test=partition(x,y,train_portion=i)
        
        if model == "knn":
            y_predict = KNN_predict(X_train, Y_train, X_test, 5)
            accuracy_score.append(accuracy(y_predict,Y_test))
            
        elif model == "forest":
            y_predict = randomForest(X_train, Y_train, X_test,100)
            accuracy_score.append(accuracy(y_predict,Y_test))
            
        elif model == "logistic":
            ### THIS IS ABID'S FUNCTION (CHANGE ACCORDING TO THE DEFINITON)
            y_predict = logistic.fit(X_train, Y_train, X_test)
            accuracy_score.append(accuracy(y_predict,Y_test))
            
    # plotting routine
    plt.plot(size, accuracy_score)
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