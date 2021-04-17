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


def fillembarked3(df):
    df['Embarked'] = df['Embarked'].fillna(value='S')  ## filling with the most common


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
