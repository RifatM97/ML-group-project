import pandas as pd
import random 
import numpy as np

def import2df(file):
    return pd.read_csv(file, usecols=['Survived', 'Pclass', 'Name',
                                      'Sex', 'Age', 'SibSp', 'Parch',
                                      'Fare', 'Embarked'])


def sex2binary(df):
    """Change sex column to 1 or zero"""
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})


def convert2onehot(df, *args):
    """Convert any int column to a one hot vector"""
    for column in args:
        onehot = pd.get_dummies(df[column])
        df = df.drop(column, axis=1)
        df = df.join(onehot)
    return df


def prep_fill_na_age(df, df_med):
    """Replace missing age values with the median of the PClass and SibSP
        Note: this function was adapted from an idea posted online. Please see README file for reference [1]."""
    for x in range(len(df)):
        #For each class...
        if df["Pclass"][x]==1:
            #For each number of siblings
            if df["SibSp"][x]==0:
                #Obtain the median age
                return df_med.loc[1,0]["Age"]
            elif df["SibSp"][x]==1:
                return df_med.loc[1,1]["Age"]
            elif df["SibSp"][x]==2:
                return df_med.loc[1,2]["Age"]
            elif df["SibSp"][x]==3:
                return df_med.loc[1,3]["Age"]
        elif df["Pclass"][x]==2:
            if df["SibSp"][x]==0:
                return df_med.loc[2,0]["Age"]
            elif df["SibSp"][x]==1:
                return df_med.loc[2,1]["Age"]
            elif df["SibSp"][x]==2:
                return df_med.loc[2,2]["Age"]
            elif df["SibSp"][x]==3:
                return df_med.loc[2,3]["Age"]
        elif df["Pclass"][x]==3:
            if df["SibSp"][x]==0:
                return df_med.loc[3,0]["Age"]
            elif df["SibSp"][x]==1:
                return df_med.loc[3,1]["Age"]
            elif df["SibSp"][x]==2:
                return df_med.loc[3,2]["Age"]
            elif df["SibSp"][x]==3:
                return df_med.loc[3,3]["Age"]
            elif df["SibSp"][x]==4:
                return df_med.loc[3,4]["Age"]
            elif df["SibSp"][x]==5:
                return df_med.loc[3,5]["Age"]
            elif df["SibSp"][x]==8:
                return df_med.loc[3]["Age"].median() 

def fill_na_age(df):
    """Set up and call function to replace missing age values 
        with the median of the PClass and SibSP """
    #Group the data by class and number of siblings
    df_med=df.groupby(["Pclass","SibSp"]).median()
    #Fill missing age values with the median of the class and number of siblings
    df["Age"]=df["Age"].fillna(prep_fill_na_age(df, df_med))

def fillembarked3(df):
    """Filling missing values for Embarked column with the most common result"""
    df['Embarked'] = df['Embarked'].fillna(value='S')  


def extractTitles(df):
    """EXTRACT title column with Mr,Mrs,Miss,Master or rare
        Note: this function was adapted from an idea posted online. Please see README file for reference [2]"""
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    #Collapse less frequent titles into larger groups for easier comparison
    df['Title'] = df['Title'].replace(
        ['Capt', 'Col', 'Countess', 'Lady', 'Don', 'Dona', 'Dr', 'Major', 'Jonkheer', 'Rev', 'Sir'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df.drop(['Name'], axis=1)


def partition(x, y, train_portion=None):
    """ Partitions the data into train-test.
    Inputs  - x : the titanic dataset
            - y : the survived columns
    Outputs - The data split in 2 different parts
    """
    # Divide datasets
    np.random.seed(100)  # same seed for consistent workflow

    # Decide training portion
    if train_portion is None:
        train_portion = 0.8
        test_portion = 0.2
    else:
        train_portion = train_portion
        test_portion = 1-train_portion

    # Converting the df to numpy arrays
    y = y.to_numpy()
    x = x.to_numpy()

    ### randomise the data set
    idx = [i for i in range(len(x))]
    np.random.shuffle(idx)

    #Split the row numbers into train and test set
    train_idx, test_idx = np.split(idx, [int(train_portion * len(x))])

    #Subset the data according to the train and test split of the row numbers
    X_train = x[train_idx]
    Y_train = y[train_idx]

    X_test = x[test_idx]
    Y_test = y[test_idx]

    return X_train, Y_train, X_test, Y_test

