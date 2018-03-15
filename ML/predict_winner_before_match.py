# Random Forest Regression

# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/match_with_city.csv')

dataset['winner']= np.where(dataset['winner']==dataset['team1'],1,0)
X = dataset.iloc[:, [2,5,6,19]].values
y = dataset.iloc[:, 11].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [0,1,2,3])
enc = onehotencoder.fit(X)
X = enc.transform(X).toarray()
# Encoding the Dependent Variabl

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

test = ["2017",1,4,2]
test = enc.transform([test]).toarray()
p = regressor.predict(test)
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5 ] = 0

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#per = (cm[0][0] + cm[1][1])/96


from sklearn.externals import joblib
joblib.dump(regressor, 'pre_pred.pkl')

import pickle
with open('pre_hot.pkl', 'wb') as pickle_file:
    pickle.dump(enc, pickle_file)





# Visualising the Random Forest Regression results (higher resolution)
