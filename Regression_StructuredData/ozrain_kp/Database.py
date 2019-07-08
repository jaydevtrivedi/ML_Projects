class DataCleaner:
    def __init__(self, dataset):
        self.dataset = dataset

    # Clean the dataset(dealing with Nans), if a column has more than 25% Nan we drop it
    # if it has less we replace it with the median
    def deal_with_nans(self):
        dataset = self.dataset
        columns = dataset.columns
        columns = columns.drop(['Date', 'Location'])
        for column_name in columns:
            print("column %s up for a test" % (column_name))
            if dataset[column_name].isnull().sum() > 0.25 * len(dataset):
                print("column %s qualifies for a drop" % (column_name))
                dataset.drop(column_name, axis=1, inplace=True)
            elif type(dataset[column_name].loc[0]) == str:
                print("column %s qualifies for a stringfill" % (column_name))
                dataset[column_name].fillna(method='ffill', inplace=True)
                dataset[column_name] = dataset[column_name].str.replace("'", "")
            else:
                print("column %s qualifies for a mean" % (column_name))
                dataset[column_name].fillna(dataset[column_name].mean(), inplace=True)
        return dataset

    # Scaling the data
    def scaledata(self, X_train, X_test):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
        return X_train, X_test

    def peform_encoding(self, clean_dataset):
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        X = clean_dataset.iloc[:, 1:-1].values
        y = clean_dataset.iloc[:, -1].values

        # Dealing with categorical data for locations
        label_encoder_X0 = LabelEncoder()
        label_encoder_X4 = LabelEncoder()
        label_encoder_X6 = LabelEncoder()
        label_encoder_X7 = LabelEncoder()
        label_encoder_X16 = LabelEncoder()
        X[:, 0] = label_encoder_X0.fit_transform(X[:, 0])
        X[:, 4] = label_encoder_X4.fit_transform(X[:, 4])
        X[:, 6] = label_encoder_X6.fit_transform(X[:, 6])
        X[:, 7] = label_encoder_X7.fit_transform(X[:, 7])
        X[:, 16] = label_encoder_X16.fit_transform(X[:, 16])
        oneHotEncoder = OneHotEncoder(categorical_features=[0, 4, 6, 7])
        X = oneHotEncoder.fit_transform(X)
        y = label_encoder_X16.fit_transform(y)
        return X, y
