from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import math

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
def fishers_LDA(x_train, y_train, x_test):
    inputs0 = x_train[y_train.Survived==0]
    inputs1 = x_train[y_train.Survived==1]
    m0, S0, N0 = max_lik(inputs0)
    m1, S1, N1 = max_lik(inputs1)
    p0 = N0/(N0+N1)
    p1 = N1/(N0+N1)
    #For plot
    Sw = S0 + S1
    n_vars = len(x_train.columns)
    w = np.dot(np.linalg.inv(Sw),(m0-m1).reshape(n_vars,1))
    N, D = x_train.shape
    w_norm = w/np.sum(w)
    # we want to make sure that the projection is in the right direction
    # i.e. giving larger projected values to class1 so:
    projected_m0 = np.mean(project_data(inputs0, w))
    projected_m1 = np.mean(project_data(inputs1, w))
    #print(projected_m0, projected_m1)
    if projected_m0 > projected_m1:
        w_norm = -w_norm
    #apply the weights to the training data
    projected_inputs_train = project_data(x_train, w)
    # In case the classes are not integers, this is a simple encoding from class to integer.
    N = x_train.shape[0]
    targets_train = np.empty(N)
    #we assume there are only 2 classes in the target variable
    # get the class values as a pandas Series object
    class_values = y_train['Survived']
    classes = class_values.unique()
    for class_id, class_name in enumerate(classes):
        is_class = (class_values == class_name)
        targets_train[is_class] = class_id
    ax_train = plot_class_histograms(projected_inputs_train, targets_train)
    # label x axis
    ax_train.set_xlabel(r"$\mathbf{w}^T\mathbf{x}$")
    ax_train.set_title("Projected Data: %s" % "fisher")
    ax_train.legend(classes)
    #Predict
    projected_inputs_test = project_data(x_test, w)
    predict_prob = 1/(1+(projected_inputs_test))
    y_pred = [0]*len(x_test)
    threshold = -0.2
    #return projected_inputs_test
    for i in range(len(x_test)):
      x = predict_prob[i]
      if x >= threshold:
        y_pred[i] = 1
      else:
        y_pred[i] = 0
    return y_pred

def max_lik(data):
    N = len(data)
    m = np.array(data.apply(sum)/N)
    S = np.zeros((15,15))
    data1 = np.array(data)
    for i in range(N):
        S = S + np.dot((data1[i,:]-m).reshape(15,1),(data1[i,:]-m).reshape(15,1).T)
    return m, S/N, N

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

# Creating a classifier for Logistic Regression
# This method uses stochastic gradient descent
class LogisticRegression():
    
    # Constructor used to initialize the objects in the class
    # Where itr = number of iterations for convergance
    # learnrate = learning rate of gradient descent
    # bias indicates if bias should be accounted for
    def __init__(self, learnrate=0.01, itr =100000, bias=True):

        
    # Equating the variables defined in this function to that of LR
        self.learnrate = learnrate
        self.itr = itr
        self.bias = bias
        self.weights = None 
       
    # Logistic sigmoid equation
    @staticmethod
    def __sigmoid(a):
        return 1 / (1 + np.exp(-a))

    # Intercept array
    @staticmethod
    def __intercepts(X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    

    # Weighting the training data  
    def weighting(self, X, y, test, threshold = 0.5):
        if self.bias:
            X = self.__intercepts(X)
        
        self.weights = np.zeros(X.shape[1])

        #Calculating across all iterations
        for i in range(self.itr):
            
            # Gradient Descent equation
            a = np.dot(X, self.weights)
            h = self.__sigmoid(a)
            graddes = np.dot(X.T, (h - y)) / y.size
            self.weights -= self.learnrate * graddes

        # Predicts the membership using the probabilities
        prediction = self.prob(test) >= threshold
        return np.multiply(prediction, 1) 
    
    # Determines the probability of a class membership
    def prob(self, X):
        if self.bias:
            X = self.__intercepts(X)
            a = np.dot(X, self.weights)
    
        return self.__sigmoid(a)

