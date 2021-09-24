import pandas as pd
import numpy as np
import sklearn

from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures

data = pd.read_csv('../data/flight_delay.csv')

x = data.drop('Delay', axis=1)
y = data['Delay']

sample = data.loc[(data['Scheduled depature time'] > '2015-10-31 23:59:00') &
            (data['Scheduled depature time'] < '2015-12-01 00:00:00')]

# we now have to encode sample data too

le = LabelEncoder()
for i in sample.columns.values:
    fitter = sample[i]
    sample[i] = le.fit_transform(fitter)

z = np.abs(stats.zscore(sample))
threshold = 3

# Number of outliers from a sampled month, Nov, 2015
sample_outliers = len(np.where(z > threshold)[0])

# We split our data taking all data less than 2018 for train  and data >= 2018 for testing.
train_data = data.loc[data['Scheduled depature time'] < '2018-01-01 00:00:00']
test_data = data.loc[data['Scheduled depature time'] > '2017-12-31 23:59:59']

for i in data.columns.values:
    fitter = train_data[i].append(test_data[i])
    le.fit(fitter)
    train_data[i] = le.transform(train_data[i])
    test_data[i] = le.transform(test_data[i])

z_train = np.abs(stats.zscore(train_data))
train_data_n = train_data[(z_train < 3).all(axis=1)]

z_test = np.abs(stats.zscore(test_data))
test_data_n = test_data[(z_test < 3).all(axis=1)]

outliers_in_train_data = train_data.shape[0] - train_data_n.shape[0] 
outliers_in_test_data = test_data.shape[0] - test_data_n.shape[0]

test_X = test_data.drop('Delay', axis=1)
test_y = test_data['Delay']

train_X = train_data.drop('Delay', axis=1)
train_y = train_data['Delay']

min_max = MinMaxScaler()
train_X = min_max.fit_transform(train_X)
test_X = min_max.fit_transform(test_X)

# We now start training with a LinearRegressor
regressor = LinearRegression()
regressor.fit(train_X, train_y)

y_pred = regressor.predict(test_X)

lr_mse = metrics.mean_squared_error(test_y, y_pred)
lr_mae =  metrics.mean_absolute_error(test_y, y_pred)

# Next is to do polynomial regression  on the flight duration alone.
# Or more appropriately we do a linear regression but transform the input into a polynomial order.
# We found out that a degree of 1 performed better, so we'll do just that here

flight_duration_test = test_X[:,1] - test_X[:,3]
flight_duration_train = train_X[:,1] - train_X[:,3]

degree = 1

polynomial_features = PolynomialFeatures(degree=degree)
linear_regression = LinearRegression()


pipeline = Pipeline([("polynomial_features", polynomial_features),
                    ("linear_regression", linear_regression)])

x = flight_duration_train
x = x[:, np.newaxis]
y = train_y

pipeline.fit(x, y)

y_pred = pipeline.predict(flight_duration_test[:, np.newaxis])

pr_mse = metrics.mean_squared_error(test_y, y_pred)
pr_mae =  metrics.mean_absolute_error(test_y, y_pred)

# Now we will perform Lasso regularization on the Linear Regressor
# In other words, we will try a linear regression model which uses the lasso regression.

# Again we found out after testing that the alpha value which gives the minimum error is 3

alpha = 3.1
lasso = Lasso(alpha=alpha)
lasso.fit(train_X, train_y)

pred_y = lasso.predict(test_X)

lass_mse = metrics.mean_squared_error(test_y, y_pred)
lass_mae =  metrics.mean_absolute_error(test_y, y_pred)

for i, j, k in [('Linear Regression', lr_mae, lr_mse),
                ('Polynomial Regression', pr_mae, pr_mse), 
                ('Lasso Regression', lass_mae, lass_mse)]:
  print(f'Algorithm: {i}\nMAE: {j:.5f}\nMSE: {k:.5f}\n')

print(f'Outliers in Train Data {outliers_in_train_data}')
print(f'Outliers in Test Data {outliers_in_test_data}')