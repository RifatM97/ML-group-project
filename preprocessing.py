import pandas as pd


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


def fillagewithmean(df):
    """Replace missing age values with the average"""
    df['Age'] = df['Age'].fillna(value=df['Age'].mean())

#Need to include link to this function in reference
def prep_fill_na_age(df, df_med):
    """Replace missing age values with the median of the PClass and SibSP"""
    for x in range(len(df)):
        if df["Pclass"][x]==1:
            if df["SibSp"][x]==0:
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
    df_med=df.groupby(["Pclass","SibSp"]).median()
    df["Age"]=df["Age"].fillna(prep_fill_na_age(df, df_med))

def fillembarked3(df):
    ## filling with the most common
    df['Embarked'] = df['Embarked'].fillna(value='S')  


def extractTitles(df):
    """EXTRACT title column with Mr,Mrs,Miss,Master or rare"""
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Capt', 'Col', 'Countess', 'Lady', 'Col', 'Don', 'Dona', 'Dr', 'Major', 'Jonkheer', 'Rev', 'Sir'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df.drop(['Name'], axis=1)



def main():
    """This section is for testing the preprocessing, will be run if you run this file only"""
    pd.set_option('display.max_columns', None)

    train = import2df('data/train.csv')
    print(train.to_string())
    print(train.isna().sum())

    sex2binary(train)
    fillagewithmean(train)
    fillembarked3(train)
    train = extractTitles(train)
    print(train)
    train = convert2onehot(train, 'Sex', 'Embarked', 'Title')
    print(train)


if __name__ == "__main__":
    main()
