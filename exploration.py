import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import preprocessing as prep

# Data Loading 
train = pd.read_csv(r"data\train.csv")
print(train) # training data
test = pd.read_csv(r"data\train.csv")
print(test) # testing data

# Concatenating to full data
def inspection():
    df = [train,test]
    titanic = pd.concat(df)
    print(titanic.info())

    # Not using the testing set as it does not contain survival column
    # For supervised models we need to train models on survival column
    # Only training set will be explored

    # Data types
    for column in train.columns.values:
        print (column, "data type: ", type(train[column][1]))

    # Identifying missing values
    for column in train.columns.values:
        count_nan = train[column].isna().sum()
        print (column, "total missing: ", count_nan)

    # Cabin data is full of gaps, Age needs to be filled.

    # Inspecting numerical data
    print(train.describe())
    # Inspecting non-numerical data
    print(train.describe(include=['O']))

    # Inspecting name column and titles
    train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    print(pd.crosstab(train['Title'], train['Sex']))

# Visualizing data

def visual():
    """Visualizing data"""

    # plotting a scatter plot of all the features against each other
    sns.pairplot(train[["Age","SibSp","Parch","Fare","Pclass","Survived"]])
    plt.savefig('exploratory_plots\scatterplots.png')

    # plotting correlation heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(train.corr(), annot=True)
    plt.savefig('exploratory_plots\heatmap.png')

    # plotting a 
    list = ["Age","SibSp","Parch","Fare","Pclass","Sex"]
    for i in list:
        age_dist = sns.FacetGrid(train, col='Survived')
        age_dist.map(plt.hist, i, bins=20)
        plt.savefig('exploratory_plots\Bar_chart1{}.png'.format(i))

    # plotting a
    list = ["Age","SibSp","Parch","Fare","Pclass","Sex"]
    for i in list:
        plt.figure(figsize=(10,8))
        sns.countplot(x = i, data = train, hue = "Survived")
        plt.savefig('exploratory_plots\Bar_chart2{}.png'.format(i))
    

# printing the functions above
# All the exploratory plots are saved in the exploratory_plots folder

inspection()
visual()

