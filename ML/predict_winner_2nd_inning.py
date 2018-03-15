import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('data/2nd_inning.csv')
dataset['winner']= np.where(dataset['winner']==dataset['batting_team'],1,0)



X = dataset.iloc[:, [2,3,4,5,6,10,11]].values
y = dataset.iloc[:, 7].values

# -*- coding: utf-8 -*-
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0,1,6])
enc = onehotencoder.fit(X)
X = enc.transform(X).toarray()



from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

#from sklearn.metrics import accuracy_score
#accuracy= accuracy_score(y_test,y_pred)

from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5 ] = 0

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
per = (cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])

from sklearn.externals import joblib
joblib.dump(regressor, '2st_inn_win_pred.pkl') 

import pickle
with open('2st_inn_win_hot.pkl', 'wb') as pickle_file:
    pickle.dump(enc, pickle_file)
