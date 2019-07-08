class ClassificationMethods:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def logisticRegression(self):
        # Trying logistic regression
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state=0)
        classifier.fit(self.X_train, self.y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(self.X_test)
        return y_pred

