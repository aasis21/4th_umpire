from django import forms
from .models import Match

team_choice = (
        (1,	'Sunrisers Hyderabad'),
        (2,	'Royal Challengers Bangalore'),
        (3,	'Chennai Super Kings'),
        (4,	'Kings XI Punjab'),
        (5,	'Rajasthan Royals'),
        (6,	'Delhi Daredevils'),
        (7,	'Mumbai Indians'),
        (8,	'Kolkata Knight Riders'),
)

city_choice = (
    (1	, 'Hyderabad'),
    (2	, 'Pune'),
    (3	, 'Rajkot'),
    (4	, 'Indore'),
    (5	, 'Bangalore'),
    (6	, 'Mumbai'),
    (7	, 'Delhi'),
    (8	, 'Chennai'),
    (9	, 'Sharjah'),
    (10	, 'Ranchi'),
    (11	, 'Patna'),
    (12	, 'Kolkata'),
    (13	, 'Chandigarh'),
    (14	, 'Kanpur'),
    (15	, 'Jaipur'),
    (24	, 'Ahmedabad'),
    (25	, 'Cuttack'),
    (26	, 'Nagpur'),
    (27	, 'Dharamsala'),
    (28	, 'Kochi'),
    (29	, 'Visakhapatnam'),
    (30	, 'Raipur'),
)

class PreMatch(forms.Form):
    team1 = forms.ChoiceField(choices=team_choice, widget=forms.Select())
    team2 = forms.ChoiceField(choices=team_choice, widget=forms.Select())
    venue = forms.ChoiceField(choices=city_choice, widget=forms.Select())


class InningsFirst(forms.Form):
    team1 = forms.ChoiceField(choices=team_choice, widget=forms.Select(),label="Batting Team")
    team2 = forms.ChoiceField(choices=team_choice, widget=forms.Select(),label="Bowling Team")
    venue = forms.ChoiceField(choices=city_choice, widget=forms.Select())
    runs = forms.IntegerField()
    overs_played = forms.IntegerField()
    wickets_fallen = forms.IntegerField()

class InningsSecond(forms.Form):
    team1 = forms.ChoiceField(choices=team_choice, widget=forms.Select(),label="Current Batting Team")
    team2 = forms.ChoiceField(choices=team_choice, widget=forms.Select(),label="Current Bowling Team")
    venue = forms.ChoiceField(choices=city_choice, widget=forms.Select())
    runs = forms.IntegerField()
    wickets_fallen = forms.IntegerField()
    overs_played = forms.IntegerField()
    target_set = forms.IntegerField()
