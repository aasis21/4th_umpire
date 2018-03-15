"""
Objective:
Originally the problem was listed on Kaggle with objective only one objective (#1 listed below). However, I extended the problem to the following multiple objectives: 

1. Predict winner of any given match (of any season) even before the match has started.
2. Predict the winner of any given match when the first inning is finished and before the start of the second inning
3. Predict who will win the 9th season final before the match has begun
4. Predict who will win the 9th season final before the match has begun
5. Predict the winner of final of 9th season based on ball-by-ball (i.e. at any given state of the match)

"""

#Load modules
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
#%matplotlib inline

pd.set_option('display.max_columns', 50)



#Read Dataset
data_path = "../data/"
match_df = pd.read_csv("data/matches.csv")
score_df = pd.read_csv("data/deliveries.csv")
match_df.head()



# Processing Data for Predicting winner before match and after first innings

columns_pre = ['match_id', 'season', 'city', 'date', 'team1', 'team2', 'toss_winner', 'toss_decision', 'winner']
columns_pre = ['id', 'season', 'city', 'team1', 'team2', 'toss_winner', 'toss_decision', 'umpire1', 'umpire2', 'dl_applied']
winner = ['winner']

label = match_df['winner']
feats = match_df[columns_pre[:]]

feats.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW'],inplace=True)

label.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW'],inplace=True)



def get_in_match_feats(score_df):
    score_df = pd.merge(score_df, match_df[['id','season', 'winner', 'result', 'dl_applied', 'team1', 'team2']], left_on='match_id', right_on='id')
    score_df.player_dismissed.fillna(0, inplace=True)
    score_df['player_dismissed'].ix[score_df['player_dismissed'] != 0] = 1
    train_df = score_df.groupby(['match_id', 'inning', 'over', 'team1', 'team2', 'batting_team', 'winner'])[['total_runs', 'player_dismissed']].agg(['sum']).reset_index()
    train_df.columns = train_df.columns.get_level_values(0)

    # Innings score and wickets
    train_df['innings_wickets'] = train_df.groupby(['match_id', 'inning'])['player_dismissed'].cumsum()
    train_df['innings_score'] = train_df.groupby(['match_id', 'inning'])['total_runs'].cumsum()
    train_df.groupby(['match_id', 'inning'])['innings_score'].sum()
    train_df = train_df[train_df.inning < 2]
    #train_df.head()
    
    return train_df


def transform_feat(feats):
    team_1 = pd.get_dummies(feats['team1'], prefix='t1')
    team_2 = pd.get_dummies(feats['team2'], prefix='t2')
    season = pd.get_dummies(feats['season'])
    city = pd.get_dummies(feats['city'])
    toss_winner = pd.get_dummies(feats['toss_winner'], prefix='toss_win')
    toss_decision = pd.get_dummies(feats['toss_decision'])
    ump1 = pd.get_dummies(feats['umpire1'])
    ump2 = pd.get_dummies(feats['umpire2'])

    X = pd.concat([team_1,team_2,season, city, toss_winner, toss_decision, ump1, ump2], axis=1)
    
    # getting target from first inning
    train_df = get_in_match_feats(score_df)
    inning_score_df = train_df.groupby(['match_id', 'inning'], sort=False)['innings_score'].max().to_frame()
    inning_score_df.reset_index(level=['match_id', 'inning'], inplace=True)
    out_df = pd.merge(inning_score_df, feats, left_on='match_id', right_on='id')
    
    X_mid = pd.concat([X,out_df['innings_score']], axis=1)
    
    del train_df
    return X, X_mid
    
t_label = pd.get_dummies(label)
t_feats, t_feats_mid = transform_feat(feats)


X = np.array(t_feats.values)
Y = np.array(match_df['winner'].fillna(-1).values)

X_mid = np.array(t_feats_mid.values)


le = preprocessing.LabelEncoder()
Y_c = le.fit_transform(Y)
#print list(le.classes_)





#Predicting the winner before match and post first innings
k = 20
dev_X = X[:-k]
dev_Y = Y_c[:-k]
val_X = X[-k:]
val_Y = Y_c[-k:]

dev_X_mid = X_mid[:-k]
val_X_mid = X_mid[-k:]

#model = runXGB(dev_X, dev_Y)
#xgtest = xgb.DMatrix(val_X)
#preds = model.predict(xgtest)
#print preds

X_train, X_test, y_train, y_test = train_test_split(X, Y_c, test_size=0.05, random_state=42)
X_train_mid, X_test_mid, y_train_mid, y_test_mid = train_test_split(X_mid, Y_c, test_size=0.05, random_state=42)

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

xgb_model_mid = xgb.XGBClassifier()
xgb_model_mid.fit(X_train_mid, y_train_mid)

preds = xgb_model.predict(X_test)
probs = xgb_model.predict_proba(X_test)

preds_mid = xgb_model_mid.predict(X_test_mid)
probs_mid = xgb_model_mid.predict_proba(X_test_mid)



t1_t2 = []
for x in X_test:
    non_zero = [i for i, v in enumerate(x[:26]) if v > 0]
#    non_zero[1] = non_zero[1] - 13
    t1_t2.append(non_zero)


updated_preds = []
#for p, teams in zip(probs, t1_t2):
 #   if p[teams[0]+1] > p[teams[1]+1]:
  
#      updated_preds.append(teams[0]+1)
 #   else:
  #      updated_preds.append(teams[1]+1)


print("========================================================================================")
print("Accuracy of prediction before match: ")
print(accuracy_score(y_test, preds))

print("Accuracy of prediction post first innings: ")
print(accuracy_score(y_test_mid, preds_mid))
print("========================================================================================")


#Predicting Winner of season 9 before the match and post first innnings

k = 1
dev_X = X[:-k]
dev_Y = Y_c[:-k]
val_X = X[-k:]
val_Y = Y_c[-k:]

dev_X_mid = X_mid[:-k]
val_X_mid = X_mid[-k:]

X_train, X_test, y_train, y_test = dev_X, val_X, dev_Y, val_Y
X_train_mid, X_test_mid, y_train_mid, y_test_mid = dev_X_mid, val_X_mid, dev_Y, val_Y

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

xgb_model_mid = xgb.XGBClassifier()
xgb_model_mid.fit(X_train_mid, y_train_mid)

preds = xgb_model.predict(X_test)
probs = xgb_model.predict_proba(X_test)

preds_mid = xgb_model_mid.predict(X_test_mid)
probs_mid = xgb_model_mid.predict_proba(X_test_mid)


print("========================================================================================")


print("Accuracy of prediction of Final of Season 9 before match: ")
print(accuracy_score(y_test, preds))

print("Accuracy of prediction of Final of Season 9 post first innings: ")
print(accuracy_score(y_test_mid, preds_mid))
print("========================================================================================")



# Predicting Winner of season 9 over by over

#match_df = match_df.ix[match_df.season==2016,:]
match_df = match_df.ix[match_df.dl_applied == 0,:]
match_df.head()


# Runs and wickets per over
score_df = pd.merge(score_df, match_df[['id','season', 'winner', 'result', 'dl_applied', 'team1', 'team2']], left_on='match_id', right_on='id')
score_df.player_dismissed.fillna(0, inplace=True)
score_df['player_dismissed'].ix[score_df['player_dismissed'] != 0] = 1
train_df = score_df.groupby(['match_id', 'inning', 'over', 'team1', 'team2', 'batting_team', 'winner'])[['total_runs', 'player_dismissed']].agg(['sum']).reset_index()
train_df.columns = train_df.columns.get_level_values(0)

# Innings score and wickets
train_df['innings_wickets'] = train_df.groupby(['match_id', 'inning'])['player_dismissed'].cumsum()
train_df['innings_score'] = train_df.groupby(['match_id', 'inning'])['total_runs'].cumsum()
train_df.head()

# Get the target column 
temp_df = train_df.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
temp_df = temp_df.ix[temp_df['inning']==1,:]
temp_df['inning'] = 2
temp_df.columns = ['match_id', 'inning', 'score_target']
train_df = train_df.merge(temp_df, how='left', on = ['match_id', 'inning'])
train_df['score_target'].fillna(-1, inplace=True)

# Get the remaining target
def get_remaining_target(row):
    if row['score_target'] == -1.:
        return -1
    else:
        return row['score_target'] - row['innings_score']

train_df['remaining_target'] = train_df.apply(lambda row: get_remaining_target(row),axis=1)

# Get the run rate
train_df['run_rate'] = train_df['innings_score'] / train_df['over']

# Get the remaining run rate
def get_required_rr(row):
    if row['remaining_target'] == -1:
        return -1.
    elif row['over'] == 20:
        return 99
    else:
        return row['remaining_target'] / (20-row['over'])
    
train_df['required_run_rate'] = train_df.apply(lambda row: get_required_rr(row), axis=1)

def get_rr_diff(row):
    if row['inning'] == 1:
        return -1
    else:
        return row['run_rate'] - row['required_run_rate']
    
train_df['runrate_diff'] = train_df.apply(lambda row: get_rr_diff(row), axis=1)
train_df['is_batting_team'] = (train_df['team1'] == train_df['batting_team']).astype('int')
train_df['target'] = (train_df['team1'] == train_df['winner']).astype('int')

train_df.head()





# Split Data and use final match data for validation sample
x_cols = ['inning', 'over', 'total_runs', 'player_dismissed', 'innings_wickets', 'innings_score', 'score_target', 'remaining_target', 'run_rate', 'required_run_rate', 'runrate_diff', 'is_batting_team']

# let us take all the matches but for the final as development sample and final as val sample #
val_df = train_df.ix[train_df.match_id == 577,:]
dev_df = train_df.ix[train_df.match_id != 577,:]

# create the input and target variables #
dev_X = np.array(dev_df[x_cols[:]])
dev_y = np.array(dev_df['target'])
val_X = np.array(val_df[x_cols[:]])[:-1,:]
val_y = np.array(val_df['target'])[:-1]
print(dev_X.shape, dev_y.shape)
print(val_X.shape, val_y.shape)

#print dev_X[0]
#print dev_y[0:5], dev_y[-5:]
#print list(dev_y)


# Using XGBoost for odeling
# http://dmlc.cs.washington.edu/xgboost.html
def runXGB(train_X, train_y, seed_val=0):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.05
    param['max_depth'] = 8
    param['silent'] = 1
    param['eval_metric'] = "auc"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = 100

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    model = xgb.train(plst, xgtrain, num_rounds)
    return model

# Build model and get prection for final match
model = runXGB(dev_X, dev_y)
xgtest = xgb.DMatrix(val_X)
preds = model.predict(xgtest)


# Important variables contributing to win
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i,feat))
    outfile.close()

create_feature_map(x_cols)
importance = model.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
imp_df = pd.DataFrame(importance, columns=['feature','fscore'])
imp_df['fscore'] = imp_df['fscore'] / imp_df['fscore'].sum()

# create a function for labeling #
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                '%f' % float(height),
                ha='center', va='bottom')
        
labels = np.array(imp_df.feature.values)
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,6))
rects = ax.bar(ind, np.array(imp_df.fscore.values), width=width, color='y')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Importance score")
ax.set_title("Variable importance")
autolabel(rects)
plt.show()



# Win probability at end of each over
out_df = pd.DataFrame({'Team1':val_df.team1.values})
out_df['is_batting_team'] = val_df.is_batting_team.values
out_df['innings_over'] = np.array(val_df.apply(lambda row: str(row['inning']) + "_" + str(row['over']), axis=1))
out_df['innings_score'] = val_df.innings_score.values
out_df['innings_wickets'] = val_df.innings_wickets.values
out_df['score_target'] = val_df.score_target.values
out_df['total_runs'] = val_df.total_runs.values
out_df['predictions'] = list(preds)+[1]

fig, ax1 = plt.subplots(figsize=(12,6))
ax2 = ax1.twinx()
labels = np.array(out_df['innings_over'])
ind = np.arange(len(labels))
width = 0.7
rects = ax1.bar(ind, np.array(out_df['innings_score']), width=width, color=['yellow']*20 + ['green']*20)
ax1.set_xticks(ind+((width)/2.))
ax1.set_xticklabels(labels, rotation='vertical')
ax1.set_ylabel("Innings score")
ax1.set_xlabel("Innings and over")
ax1.set_title("Win percentage prediction for Sunrisers Hyderabad - over by over")

ax2.plot(ind+0.35, np.array(out_df['predictions']), color='b', marker='o')
ax2.plot(ind+0.35, np.array([0.5]*40), color='red', marker='o')
ax2.set_ylabel("Win percentage", color='b')
ax2.set_ylim([0,1])
ax2.grid(b=False)
plt.show()


#Observations
# Scores in the corresponding over: Yellow bar - SRH ; Green - RCB
# Red line - Equal win probability; Blue line - Win probability of SRH at the end of each over.

# No of runs scored in the over instead of cumulative runs (like previous viz)
fig, ax1 = plt.subplots(figsize=(12,6))
ax2 = ax1.twinx()
labels = np.array(out_df['innings_over'])
ind = np.arange(len(labels))
width = 0.7
rects = ax1.bar(ind, np.array(out_df['total_runs']), width=width, color=['yellow']*20 + ['green']*20)
ax1.set_xticks(ind+((width)/2.))
ax1.set_xticklabels(labels, rotation='vertical')
ax1.set_ylabel("Runs in the given over")
ax1.set_xlabel("Innings and over")
ax1.set_title("Win percentage prediction for Sunrisers Hyderabad - over by over")

ax2.plot(ind+0.35, np.array(out_df['predictions']), color='b', marker='o')
ax2.plot(ind+0.35, np.array([0.5]*40), color='red', marker='o')
ax2.set_ylabel("Win percentage", color='b')
ax2.set_ylim([0,1])
ax2.grid(b=False)
plt.show()


# OBSERVATIONS
#  SRH scored 16 and 24 runs in the last 2 overs whch gave them an edge over RCB in the final. They have constant
#  low run rate in the first 8 overs, so the probability of SRH winning the match was above 0.5. After 8th over (when
#  RCB scored 21 runs, the winning probability reduced). Wickets fell in 13th and 15th over and SRH conceded only 
#  4 runs in 16th over, which shifted the game towards SRH and increased the probability of winning the match.# -*- coding: utf-8 -*-

