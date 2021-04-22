import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

import preprocessing as prep
import evaluation as eval
import methods


def preprocessing(df):
    prep.sex2binary(df)
    #prep.fillagewithmean(df)
    prep.fill_na_age(df)
    prep.fillembarked3(df)
    df = prep.extractTitles(df)
    #remove outlier for Fare
    df = df[df.Fare != max(df.Fare)]
    return prep.convert2onehot(df, 'Sex', 'Embarked', 'Title')
    #return df.drop(['Sex', 'Embarked', 'Title'], axis=1)

def main():
    # set cmd panda view and import data
    pd.set_option('display.max_columns', None)
    alldata = prep.import2df(r'C:\Users\user\ML-group-project.git\ML-group-project\data\train.csv')

    # fill in missing data and convert categories to one hot
    alldata = preprocessing(alldata)

    # split 80:20 into training data
    x = alldata.drop('Survived', axis=1)
    y = alldata["Survived"]
    #TODO use different method to split data
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = prep.partition(x, y)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=10)


    # TODO everyone can run their models here

    # Running the KNN model
    knn_prediction = methods.KNN_predict(X_train, Y_train, X_test, 5)
    print(knn_prediction)
    print(eval.accuracy(knn_prediction,Y_test))
    eval.accuracy_v_sample(x,y,model="knn")

    # Run a random forest model with 100 n_estimators
    forest_prediction = methods.randomForest(X_train, Y_train, X_test, n_estimators=100)
    print(forest_prediction)
    print(eval.accuracy(forest_prediction,Y_test))
    eval.accuracy_v_sample(x,y,model="forest")


    #fisher's LDA
    #fisher_prediction = methods.fishers_LDA(X_train, Y_train, X_test, Y_test)
    plt.show()

    #Logistic Regression

# Fit the classifier on training data X_train, Y_train
    # import time
    # start_time = time.time()


    # methods.classifier = LogisticRegression()
    # predictions = methods.classifier.weighting(X_train, Y_train, X_test)
    


    # print("--- %s seconds ---" % (time.time() - start_time))



    # # TODO score (don't submit this we need to do our own evaluations) add evaluation techniques here
    # from sklearn.metrics import accuracy_score
    # print(accuracy_score(Y_test, forest_prediction))



if __name__ == "__main__":
    main()