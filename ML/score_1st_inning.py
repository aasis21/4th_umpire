# Random Forest Regression
import sys
# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('data/1st_inning.csv')

X = dataset.iloc[:, [1,2,3,4,5,6]].values.astype(int)
y = dataset.iloc[:, 7].values
# -*- coding: utf-8 -*-
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0,1,5])
enc = onehotencoder.fit(X)
X = enc.transform(X).toarray()
#X = np.delete(X,0,1)
#X = np.delete(X,15,1)



from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

#from sklearn.metrics import accuracy_score
#accuracy= accuracy_score(y_test,y_pred)

from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.externals import joblib
joblib.dump(regressor, '1st_inn_pred.pkl')

import pickle
with open('1st_inn_hot.pkl', 'wb') as pickle_file:
    pickle.dump(enc, pickle_file)
