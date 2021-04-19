from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Random Forest Classifier - class membership prediction
def randomForest(x_train, y_train, x_test, n_estimators):
    random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=3)
    random_forest.fit(x_train, y_train)
    return random_forest.predict(x_test)

# KNN Classifier - class membership prediction
def KNN_predict(x_train, y_train, test, K):
    'This function uses the K-Nearest Neighbours (kNN) classifier. The function trains the model on the training set'
    'and predicts the survival of the testing set'
    
    # KNN classifier
    my_classifier = KNeighborsClassifier(n_neighbors=K)
    
    # training model
    KNN = my_classifier.fit(x_train,y_train)
      
    # predict
    prediction = my_classifier.predict(test)
    
    return prediction

# KNN Classifier - class membership probability
def KNN_prob(x_train, y_train, test, K):
    "This function uses the K-Nearest Neighbours (kNN) classifier. The function trains the model on the training set"
    "and predicts the probability of survival of the testing set"
    
    # KNN classifier
    my_classifier = KNeighborsClassifier(n_neighbors=K)
    
    # training model
    KNN = my_classifier.fit(x_train,y_train)
      
    # predict
    prediction = my_classifier.predict_proba(test)
    
    return prediction


#TODO enter your methods here 
def fishers_LDA(x_train, y_train):    
    data = pd.merge(x_train, y_train, left_index=True, right_index=True)
    target_col="Survived"
    # read in the data
    dataframe = data
    #print the names of all columns to the console
    print("dataframe.columns = %r" % (dataframe.columns,) )
    #obtain the number of rows in the data
    N = dataframe.shape[0]
    input_cols = list(dataframe.columns)
    # target data should not be part of the inputs
    input_cols.remove("Survived")
    #print the names of the columns that will be used in the model to the console
    print("input_cols = %r" % (input_cols,))
    #we assume there are only 2 classes in the target variable
     # get the class values as a pandas Series object
    class_values = dataframe["Survived"]
    classes = class_values.unique()
    #print the unique classes to the console
    print("classes = %r" % (classes,))
    # In case the classes are not integers, this is a simple encoding from class to integer.
    targets = np.empty(N)
    for class_id, class_name in enumerate(classes):
        is_class = (class_values == class_name)
        targets[is_class] = class_id
    # We assume all our inputs are real numbers (or can be
    # represented as such), so we'll convert all these columns to a 2d numpy
    # array object
    inputs = dataframe[input_cols].values
        #classes=classes
    # get the shape of the data (N = number of rows, D = number of columns)
    N, D = inputs.shape
    # separate by each class
    inputs0 = inputs[targets==0]
    inputs1 = inputs[targets==1]
    # find maximum likelihood approximations to the two data-sets
    m0, S_0 = max_lik_mv_gaussian(inputs0)
    m1, S_1 = max_lik_mv_gaussian(inputs1)
    # convert the mean vectors to column vectors (type matrix)
    m0 = np.matrix(m0).reshape((D,1))
    m1 = np.matrix(m1).reshape((D,1))
    # calculate the total within-class covariance matrix (type matrix)
    S_W = np.matrix(S_0 + S_1)
    # calculate weights vector
    weights = np.array(np.linalg.inv(S_W)*(m1-m0))
    # normalise
    weights = weights/np.sum(weights)
    # we want to make sure that the projection is in the right direction
    # i.e. giving larger projected values to class1 so:
    projected_m0 = np.mean(project_data(inputs0, weights))
    projected_m1 = np.mean(project_data(inputs1, weights))
    if projected_m0 > projected_m1:
        weights = -weights
    #apply the weights to the data
    projected_inputs = project_data(inputs, weights)
    ax = plot_class_histograms(projected_inputs, targets)
    # label x axis
    ax.set_xlabel(r"$\mathbf{w}^T\mathbf{x}$")
    ax.set_title("Projected Data: %s" % "fisher")
    if not classes is None:
        ax.legend(classes)

def project_data(data, weights):
    """
    Projects data onto single dimension according to some weight vector

    parameters
    ----------
    data - a 2d data matrix (shape NxD array-like)
    weights -- a 1d weight vector (shape D array like)

    returns
    -------
    projected_data -- 1d vector (shape N np.array)
    """
    N, D = data.shape
    data = np.matrix(data)
    weights = np.matrix(weights).reshape((D,1))
    projected_data = np.array(data*weights).flatten()
    return projected_data

def max_lik_mv_gaussian(data):
    """
    Finds the maximum likelihood mean and covariance matrix for gaussian data
    samples (data)

    parameters
    ----------
    data - data array, 2d array of samples, each row is assumed to be an
      independent sample from a multi-variate gaussian

    returns
    -------
    mu - mean vector
    Sigma - 2d array corresponding to the covariance matrix  
    """
    # N = number of rows in data, dim = number of columns
    N, dim = data.shape
    # the mean sample is the mean of the rows of data
    mu = np.mean(data,0)
    Sigma = np.zeros((dim,dim))
    # the covariance matrix requires us to sum the dyadic product of
    # each sample minus the mean.
    for x in data:
        # subtract mean from data point, and reshape to column vector
        # note that numpy.matrix is being used so that the * operator
        # in the next line performs the outer-product v * v.T 
        x_minus_mu = np.matrix(x - mu).reshape((dim,1))
        # the outer-product v * v.T of a k-dimentional vector v gives
        # a (k x k)-matrix as output. This is added to the running total.
        Sigma += x_minus_mu * x_minus_mu.T
    # Sigma is unnormalised, so we divide by the number of datapoints
    Sigma /= N
    # we convert Sigma matrix back to an array to avoid confusion later
    return mu, np.asarray(Sigma)

def plot_class_histograms(
        inputs, class_assignments, bins=20, colors=None, ax=None):
    """
    Plots histograms of 1d input data, split according to class

    parameters
    ----------
    inputs - 1d vector of input values (array-like)
    class_assignments - 1d vector of class values as integers (array-like)
    colors (optional) - a vector of colors one per class
    ax (optional) - pass in an existing axes object (otherwise one will be
        created)
    """
    class_ids = np.unique(class_assignments)
    num_classes = len(class_ids)
    # calculate a good division of bins for the whole data-set
    _, bins = np.histogram(inputs, bins=bins)
    # create an axes object if needed
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    # plot histograms
    for i, class_id in enumerate(class_ids):
        class_inputs = inputs[class_assignments==class_id]
     #   ax.hist(class_inputs, bins=bins, color=colors[i], alpha=0.6)
        ax.hist(class_inputs, bins=bins, alpha=0.6)
    print("class_ids = %r" % (class_ids,))
    return ax



