import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('data/2nd_inning.csv')
dataset["winner"] = np.where(dataset["winner"]==dataset["batting_team"],1,0)

dataset_bat_wins = dataset[(dataset["winner"]==1)]
dataset_bowl_wins = dataset[(dataset["winner"]==0)]


X_bat = dataset_bat_wins.iloc[:, [2,3,4,5,6,10,11]].values
y_bat = dataset_bat_wins.iloc[:,9 ].values

# -*- coding: utf-8 -*-
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0,1,6])
enc_bat = onehotencoder.fit(X_bat)
X_bat = enc_bat.transform(X_bat).toarray()



from sklearn.cross_validation import train_test_split
X_train_bat, X_test_bat, y_train_bat, y_test_bat = train_test_split(X_bat, y_bat, test_size = 0.25, random_state = 0)


from sklearn.ensemble import RandomForestRegressor
regressor_bat = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor_bat.fit(X_train_bat, y_train_bat)

# Predicting a new result
y_pred_bat = regressor_bat.predict(X_test_bat)

from sklearn.externals import joblib
joblib.dump(regressor_bat, '2st_inn_bat_win_wicket.pkl')

import pickle
with open('2st_inn_bat_win_hot.pkl', 'wb') as pickle_file:
    pickle.dump(enc_bat, pickle_file)


#from sklearn.metrics import accuracy_score
#accuracy= accuracy_score(y_test,y_pred)

from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test_bat, y_pred_bat)))


# 2nd model if batting team loses

X_bowl = dataset_bowl_wins.iloc[:, [2,3,4,5,6,10,11]].values
y_bowl = dataset_bowl_wins.iloc[:,8 ].values

# -*- coding: utf-8 -*-
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0,1,6])
enc_bowl = onehotencoder.fit(X_bowl)
X_bowl = enc_bowl.transform(X_bowl).toarray()


from sklearn.cross_validation import train_test_split
X_train_bowl, X_test_bowl, y_train_bowl, y_test_bowl = train_test_split(X_bowl, y_bowl, test_size = 0.25, random_state = 0)


from sklearn.ensemble import RandomForestRegressor
regressor_bowl = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor_bowl.fit(X_train_bowl, y_train_bowl)

# Predicting a new result
y_pred_bowl = regressor_bowl.predict(X_test_bowl)

#from sklearn.metrics import accuracy_score
#accuracy= accuracy_score(y_test,y_pred)

from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test_bowl, y_pred_bowl)))


from sklearn.externals import joblib
joblib.dump(regressor_bowl, '2st_inn_bowl_win_run.pkl')

import pickle
with open('2st_inn_bowl_win_hot.pkl', 'wb') as pickle_file:
    pickle.dump(enc_bowl, pickle_file)



# model to predict over predicting match end over..


X = dataset.iloc[:, [2,3,4,5,6,10,11]].values
y = dataset.iloc[:,13].values

# -*- coding: utf-8 -*-
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0,1,6])
enc = onehotencoder.fit(X)
X = enc.transform(X).toarray()


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

#from sklearn.metrics import accuracy_score
#accuracy= accuracy_score(y_test,y_pred)

from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

test = enc.transform([[1,3,23,34,2,124,1]]).toarray()
ans = regressor.predict(test)

from sklearn.externals import joblib
joblib.dump(regressor, '2st_inn_end.pkl')

import pickle
with open('2st_inn_end_hot.pkl', 'wb') as pickle_file:
    pickle.dump(enc, pickle_file)
