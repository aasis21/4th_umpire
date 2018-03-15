import pandas as pd
import numpy as np


delivery_data = pd.read_csv('data/deliveries.csv')
match_data = pd.read_csv('data/matches.csv')
team_data = pd.read_csv('data/teams.csv')
city_data = pd.read_csv('data/city_id.csv')


def get_team(id):
    return teams_data[(team_data["team_id"] == id)]

def get_match(id):
    return match_data[(match_data["id"] == id)]

    

def get_winner(match):
     match_info = get_match(match).values
     winner =  match_info[0,10]
     by_run =  match_info[0,11]
     by_wicket = match_info[0,12]
     print(winner,by_run,by_wicket)
     return (winner,by_run,by_wicket)
 
def get_inning_match(inning):
    data = delivery_data[(delivery_data["inning"]==inning)]
    inn = delivery_data[(delivery_data["inning"]==1)]
    return_data = []
    for match in range(1,636):
        match_info = get_match(match)
        winner =  get_winner(match)
        print("iter:",match)
        
        for_run = inn[(inn["match_id"]==match)]
        for_run["run"] = for_run.total_runs.cumsum()
        run = for_run['run'].max()
        
        print("max_run : " ,run)
        
     
        match = data[(data["match_id"]==match)]
        match["run"] = match.total_runs.cumsum()
        match["balls"] = 6 * (match["over"] - 1 ) + match["ball"]
        match["player_dismissed"] = np.where(match["player_dismissed"].isnull(),0,1)
        match["wicket"] = match.player_dismissed.cumsum()
    
        
       
      
      
        match["winner"] = winner[0]
        match["win_by_runs"] =  winner[1]
        match["win_by_wickets"] =  winner[2]
        match["1st_inning_run"] = run
        
        
        city = match_info["city"].astype(str)
        city_id = np.nan 
        for index, row in city_data.iterrows():
            #print(row['id'], row['name'])
            if row['name'] in str(city):
                city_id = row['id']
                break
        match["city"] = city_id
        return_data.append(match)
    
    my_data = pd.concat(return_data)
    my_data = pd.DataFrame(data)
    return my_data

    
#data = get_1st_inning_match()
#data.to_csv("new.csv")

def get_1st_inning_total_run():
    data = delivery_data[(delivery_data["inning"]==2)]
    return_data = []
    for match in range(636):
        print(match)
        
        match = data[(data["match_id"]==match)]
        match["run"] = match.total_runs.cumsum()
        run = match['run'].max()
        match["total"] = run
        return_data.append(match)
    data = pd.concat(return_data)
    data = data.iloc[:, [22]].values
    data = pd.DataFrame(data)
    return data

def get_1st_inning_total_ball():
    data = pd.read_csv('data/1st_inning.csv')
    return_data = []
    for match in range(636):
        print(match)
        run = match['ball'].max()
        match["total"] = run
        return_data.append(match)
    data = pd.concat(return_data)
    data = data.iloc[:, [22]].values
    data = pd.DataFrame(data)
    return data


max_run = get_1st_inning_total_run()


def append_final_run():
    dataset = pd.read_csv('data/deliveries_1st_inning.csv')
    run = pd.read_csv('data/run_1st_inning.csv')
    y = run.iloc[:, 4].values
    run = pd.DataFrame(y)
    data = dataset.join(run)
    data = data.drop(data.columns[[0]],axis=1)
    return data 

        

    


def get_2nd_inning_data():
    data =  get_inning_match(2)
    return data
    
plk = get_2nd_inning_data()


kl = pl.join(max_run)
plk.to_csv("2nd_inng.csv")

max_run.to_csv("maxrun.csv")

y = kl.iloc[:,[0,2,3,21,22,23,24,25,26,27,28,29]].values
klp = pd.DataFrame(y)
klp.to_csv("main_2nd.csv")



k_dataset = pd.read_csv("main_2nd.csv")
k_run = pd.read_csv("maxrun.csv")
  
k_data = k_dataset.join(k_run)
k_data.to_csv("main_2nd.csv")


def get_2st_inning_total_ball():
    data = pd.read_csv('main_2nd.csv')
    return_data = []
    for match in range(636):
        
        print(match)
        match = data[(data["match_id"]==match)]
        run = match["balls"].max()
        match["total_ball"] = run
        return_data.append(match)
    data = pd.concat(return_data)
    data = pd.DataFrame(data)
    return data


toatal_ball = get_2st_inning_total_ball()
total_final = toatal_ball[toatal_ball.columns[[2,4,5,23,24,25,26,27,28,29,30,32,33]]]

total_final.to_csv("total_final.csv")



    
#winner(0)

def add_city_code():
    all_data = match_data
    return_data = []
    ctr = 0
    for match in range(1,636):
        print(match)
        match = get_match(match)
       
        city = match["city"].astype(str)
        city_id = np.nan 
        for index, row in city_data.iterrows():
            #print(row['id'], row['name'])
            if row['name'] in str(city):
                city_id = row['id']
                break
        match["city_id"] = city_id
        return_data.append(match)
    my_data = pd.concat(return_data)
    my_data = pd.DataFrame(data)
    return my_data

new_data = add_city_code()
    
        
    
        
    