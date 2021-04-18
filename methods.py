from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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

# added change



