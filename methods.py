from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Random Forest Classifier - class membership prediction
def randomForest(x_train, y_train, x_test, n_estimators):
    random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, max_samples=0.3)
    random_forest.fit(x_train, y_train)
    return random_forest.predict(x_test)

# Random Forest Classifier - class membership probability
def randomForest_prob(x_train, y_train, test, n_estimators):
        
    # classifier
    random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, max_samples=0.3)
    
    # training model
    random_forest.fit(x_train, y_train)
      
    # predict
    prediction = random_forest.predict_proba(test)
    
    return prediction

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
    my_classifier = KNeighborsClassifier(n_neighbors=K, weights='distance')
    
    # training model
    KNN = my_classifier.fit(x_train,y_train)
      
    # predict
    prediction = my_classifier.predict_proba(test)
    
    return prediction

# Fisher's LDA  
def fishers_LDA(x_train, y_train, x_test, plot_hist = False):
    """This function trains the Fishers LDA model on the training set and predicts 
        the probability of survival of the testing set"""
    # separate target by classes
    inputs0 = []
    inputs1 = []
    for i in range(len(y_train)):
        if y_train[i] == 0:
            inputs0.append(x_train[i])
        else:
            inputs1.append(x_train[i])
    # convert to numpy array
    inputs0 = np.array(inputs0)
    inputs1 = np.array(inputs1)
    #Calculate the mean vector, variance matrix and number of data points for each data set
    m0, S0, N0 = max_lik(inputs0)
    m1, S1, N1 = max_lik(inputs1)
    #Calculate the proportion of survived and died
    #this is the prior for class 0 and 1
    p0 = N0/(N0+N1)
    p1 = N1/(N0+N1)
    Sw = S0 + S1
    n_vars = len(x_train[0])
    #Calculate the weights
    w = np.dot(np.linalg.inv(Sw),(m0-m1).reshape(n_vars,1))
    N, D = x_train.shape
    #normalise the weights
    w_norm = w/np.sum(w)
    # we want to make sure that the projection is in the right direction
    # i.e. giving larger projected values to class1 so:
    projected_m0 = np.mean(project_data(inputs0, w))
    projected_m1 = np.mean(project_data(inputs1, w))
    #print(projected_m0, projected_m1)
    if projected_m0 > projected_m1:
        w_norm = -w_norm
    #apply the weights to the training data
    projected_inputs_train = project_data(x_train, w_norm)
    if plot_hist == True:
        projected_inputs0 = project_data(inputs0, w_norm) 
        projected_inputs1 = project_data(inputs1, w_norm) 
        plt.figure(figsize=(10,8))
        plt.hist(projected_inputs0, bins=20, alpha=0.5, label='Survived = 0')
        plt.hist(projected_inputs1, bins=20, alpha=0.5, label='Survived = 1') 
        plt.xlabel("Projected data")
        plt.ylabel("Density")
        plt.legend()  
        plt.savefig('plots\Fisher_separation.png')   
    # To calculate threshold for prediction we need the mean and variance for the 
    # separate classes in the training data
    projected_inputs0 = project_data(inputs0, w)
    projm0 = np.mean(projected_inputs0)
    projs0 = np.var(projected_inputs0)
    projected_inputs1 = project_data(inputs1, w)
    projm1 = np.mean(projected_inputs1)
    projs1 = np.var(projected_inputs1)
    #Prediction
    #Apply the weights to the test data to get to 1d
    projected_inputs_test = project_data(x_test, w)
    #create empty array to fill with prediction
    y_pred = [0]*len(x_test)
    #Predict class for each element in test data
    for i in range(len(x_test)):
      x = projected_inputs_test[i]
      #Calculate the probability of each point being in class0 or class 1
      prob_c0 = (math.log(p0)+math.log(1/math.sqrt(projs0))-(((x-projm0)**2)/(2*projs0)))
      prob_c1 = (math.log(p1)+math.log(1/math.sqrt(projs1))-(((x-projm1)**2)/(2*projs1)))
      if prob_c0 >= prob_c1:
        y_pred[i] = 0
      else:
        y_pred[i] = 1   
    return np.array(y_pred)


def max_lik(data):
    """Finds the maximum likelihood mean and covariance matrix for gaussian data
    samples (data)"""
    N, dim = data.shape
    mu = np.mean(data, 0)
    Sigma = np.zeros((dim, dim))
    # the covariance matrix requires us to sum the dyadic product of
    # each sample minus the mean.
    for x in data:
        # subtract mean from data point, and reshape to column vector
        # note that numpy.matrix is being used so that the * operator
        # in the next line performs the outer-product v * v.T
        x_minus_mu = np.matrix(x - mu).reshape((dim, 1))
        # the outer-product v * v.T of a k-dimentional vector v gives
        # a (k x k)-matrix as output. This is added to the running total.
        Sigma += x_minus_mu * x_minus_mu.T
    # Sigma is unnormalised, so we divide by the number of datapoints
    #Sigma /= N
    # we convert Sigma matrix back to an array to avoid confusion later
    return mu, np.asarray(Sigma), N

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

