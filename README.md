# 4th Umpire

This Django and Machine Learning powered Web app predicts and analyse IPL matches. It currently has three working model i.e 
prediction of a match winner before toss, prediction of expected score of 1st inning at any point of time during the match 
and prediction of winner, match concluding over during the 2nd inning of the match.


## Webpage
Live project is available [here](https://4thUmpire.pythonanywhere.com).

## Local Setup
Create and activate a virtualenv:

```bash
virtualenv cfd_apriori
cd cfd_apriori
source bin/activate
```
Clone the repository on your local environment <br>

```bash
git clone https://github.com/aasis21/4thUmpire.git `
```

Navigate to the folder <br>
```bash 
cd 4thUmpire/web
```

Install the required dependencies <br>
```bash
pip3 install -r requirements.txt 
```

Run the localhost-server <br>
```bash 
python3 manage.py runserver
```

The web-app will be available at `127.0.0.1:8000` on your browser. 

## About
This web-app is created for Microsoft Code Fun Do competition.

