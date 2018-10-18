# importing modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from hmmlearn.hmm import GaussianHMM

import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

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


def GetFeatures(data, target_filed):
    dates = np.array(data['date_number'], dtype=int)
    prices = np.array(data[target_filed])
    volumes = np.array(data['Volume'])[1:]

    # Take diff of close value. Note that this makes
    # ``len(diff) = len(close_t) - 1``, therefore, other quantities also
    # need to be shifted by 1.
    diff = np.diff(prices)
    dates = dates[1:]
    prices = prices[1:]

    # Pack diff and volume for training.
    X = np.column_stack([diff, volumes])



    #print (dp)
    return X, dates, prices

def TrainModel(X_train):
    print("fitting to HMM and decoding ...")

    # Make an HMM instance and execute fit
    model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000).fit(X_train)
    print("Transition matrix")
    print(model.transmat_)
    print()

    print("Means and vars of each hidden state")
    for i in range(model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", model.means_[i])
        print("var = ", np.diag(model.covars_[i]))
        print()

    return model

def Predict(model, X_test, dates_test, prices_test, show_details=False):
    #dates_test = list(dp.keys())
    #prices_test =list(dp.values())

    dp = dict()
    for i in range(0, len(dates_test)):
        dp[dates_test[i]] = prices_test[i]

    #print(dates_test)
    #print(prices_test)
    hidden_states = model.predict(X_test)
    #print(hidden_states)

    if (show_details):
        fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
        colours = cm.rainbow(np.linspace(0, 1, model.n_components))
        for i, (ax, colour) in enumerate(zip(axs, colours)):
            #print("i is {0}".format(i))
            # Use fancy indexing to plot data in each state.
            mask = hidden_states == i
            ax.plot_date(dates_test[mask], prices_test[mask], ".", c=colour)
            ax.set_title("{0}th hidden state".format(i))

            # Format the ticks.
            ax.xaxis.set_major_locator(YearLocator())
            ax.xaxis.set_minor_locator(MonthLocator())

            ax.grid(True)

        plt.show()

    expected_returns_and_volumes = np.dot(model.transmat_, model.means_)

    returns_and_volumes_columnwise = list(zip(*expected_returns_and_volumes))
    returns = returns_and_volumes_columnwise[0]

    predicted_prices = []
    lastN = len(X_test)
    mape = 0
    number = 0
    for idx in range(0, lastN-1):
        state = hidden_states[idx]

        current_date = dates_test[idx]
        current_price = prices_test[idx]

        predicted_date = current_date + 1
        predicted_price = current_price  + returns[state]
        predicted_prices.append((predicted_date, predicted_price))
        actual_price = dp[predicted_date]
        if show_details:
            print("current_date: {1}, actual_price_current_date: {2}; state: {0}, \npredicted_data: {3}; preidicted_price: {4}; actual_price_predicted_date: {5}\n".format(state,
                                                                                                                 current_date,
                                                                                                                 current_price, predicted_date, predicted_price, actual_price))

        mape += ((actual_price - predicted_price) if (actual_price >= predicted_price) else (predicted_price- actual_price))/actual_price
        number += 1

    mape = mape/number * 100

    print("MAPE: {0:.2f}%".format(mape))

    return

#train_data, test_data = train_test_split(data, test_size=0.2)
train_len = int(len(data) * 0.8)
train_data = data[:train_len]
test_data = data[train_len:]

X_train, d,p = GetFeatures(train_data, 'Close Price')
model = TrainModel(X_train)

X_test, d_test, p_test = GetFeatures(test_data, 'Close Price')
Predict(model, X_test, d_test,p_test, True)


