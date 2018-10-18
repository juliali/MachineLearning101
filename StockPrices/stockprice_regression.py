# importing modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import numpy as np

def GetModel(model_type):
    if (model_type == 'LinearRegression'):
        model = LinearRegression()
        return model
    elif(model_type == 'SVR'):
        model = SVR(C=1e3, gamma=0.1)
        return model

    return None

def ModelPredict(model_type, train_data,  test_data,  target_field, feature_fields, error_tolerance = 0.1, print_details = False):
    X_train = np.column_stack(train_data[fn] for fn in feature_fields)
    y_train = train_data[target_field]

    test_data = test_data.sort_values(['date_number'], ascending=[1])

    X_test = np.column_stack(test_data[fn] for fn in feature_fields) #np.column_stack([test_data['date_number']])

    model = GetModel(model_type)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    actual = test_data[target_field]

    total = 0
    correct = 0
    for i, (testitem, preditem, actualitem) in enumerate(zip(X_test, pred, actual)):
        total += 1
        errorrate = ((actualitem - preditem) if (actualitem - preditem > 0) else  (preditem - actualitem)) / actualitem
        result = 'correct' if (errorrate <= error_tolerance) else 'wrong'
        correct += 1 if (result == 'correct') else 0
        if print_details:
            print(testitem, "{:.2f}".format(preditem), "{:.2f}".format(actualitem), "{:.2f}".format(errorrate * 100)+'%' , result)

    accuracy = correct / total * 100

    print("\nUsing Model: {0} . Error Tolerance: {1}".format(model_type, error_tolerance))
    print("\nFeatures: {0}".format(feature_fields))
    print ("\nTest sample total number: {0}. Accuracy : {1:.2f}%".format(total, accuracy))
    #print("\nFinished")

# reading textfile using pandas
data = pd.read_csv("f:\\data\\msft_stock_price.csv", sep=",")

# Describing Data
data.describe()

# Checking is there any null value
data.isnull().sum()

# creating new columns data frame week nymber
data['date_number'] = [i for i in range(1,len(data['Date'])+1)]
data.tail()

# Checking Correlation of features
data.corr()


# Creating Train And Target Data
train_data, test_data = train_test_split(data, test_size=0.2)


# train Data
dates = data['date_number'].values.reshape(len(data['date_number']),1)

# Target Data
Open_Price = data['Open Price']
High_Price = data['High Price']
Low_Price = data['Low Price']
Close_Price = data['Close Price']
Volume = data['Volume']

ModelPredict('LinearRegression', train_data, test_data, 'Close Price', ['date_number', 'Volume'])
ModelPredict('SVR', train_data, test_data, 'Open Price', ['date_number'], 0.05)
