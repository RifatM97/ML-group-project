from sklearn.ensemble import RandomForestClassifier


def randomForest(x_train, y_train, x_test, n_estimators):
    random_forest = RandomForestClassifier(n_estimators=n_estimators)
    random_forest.fit(x_train, y_train)
    return random_forest.predict(x_test)



