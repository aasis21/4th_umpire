import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib
import pickle

def predict_1st_inn(team_batting,team_bowling,run,ball,wicket,city):
    regressor = joblib.load('1st_inn_pred.pkl')
    enc = pickle.load( open( "1st_inn_hot.pkl", "rb" ))
    X_test = [[team_batting,team_bowling,run,ball,wicket,city]]
    X_test = enc.transform(X_test).toarray()
    print(len(X_test[0]))
    y_pred = regressor.predict(X_test)
    print("our_prediction:",y_pred)

predict_1st_inn(2,5,12,15,2,4)
