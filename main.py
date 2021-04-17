import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import preprocessing as prep
import testing as test
import methods


def preprocessing(df):
    prep.sex2binary(df)
    #prep.fillagewithmean(df)
    prep.fill_age(df)
    prep.fillembarked3(df)
    df = prep.extractTitles(df)
    return prep.convert2onehot(df, 'Sex', 'Embarked', 'Title')


def main():
    # set cmd panda view and import data
    pd.set_option('display.max_columns', None)
    alldata = prep.import2df('data/train.csv')

    # fill in missing data and convert categories to one hot
    alldata = preprocessing(alldata)

    # split 80:20 into training data
    x = alldata.drop('Survived', axis=1)
    y = alldata["Survived"]
    #TODO use different method to split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=10)

    # TODO everyone can run their models here

    # run a random forest model with 100 n_estimators
    forest_prediction = methods.randomForest(x_train, y_train, x_test, n_estimators=100)

    # TODO score (don't submit this we need to do our own evaluations) add evaluation techniques here
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, forest_prediction))



if __name__ == "__main__":
    main()