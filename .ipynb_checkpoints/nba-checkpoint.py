'''
api: https://github.com/swar/nba_api
how to use: https://medium.com/@ben.g.ballard/how-to-analyze-nba-data-using-python-and-the-nba-api-429b0e65454d
I had to change the python version I am using since nba_api was pip installed to Python 3.11.5('base') ~\anaconda3\python.exe
'''

from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import teams
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import teams
from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import playbyplay
from nba_api.stats.endpoints import BoxScoreAdvancedV2
from nba_api.stats.endpoints import PlayerGameLogs
from nba_api.stats.endpoints import BoxScoreAdvancedV3
from nba_api.stats.endpoints import BoxScoreTraditionalV2
from nba_api.stats.endpoints import BoxScoreTraditionalV3
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import PredictionErrorDisplay
import matplotlib.pyplot as plt



# Nikola Jokić
#career = playercareerstats.PlayerCareerStats(player_id='203999') 

# pandas data frames (optional: pip install pandas)
#jokic = career.get_data_frames()[0]
#print(jokic)

def get_team_dict(team_name: str) -> dict:
    '''
    To search for an individual team or player by its name (or other attribute), dictionary comprehensions are your friend.
    '''
    nba_teams = teams.get_teams()
    return [team for team in nba_teams if team['abbreviation'] == team_name][0]
def get_player_dict(player_name: str) -> dict:
    '''
    '''
    nba_players = players.get_players()
    return [player for player in nba_players if player["full_name"].lower() == player_name.lower()][0]
def get_team_games_all(team_name: str):
    '''
    '''
    id = get_team_dict(team_name)['id'] #you'll have to fix this
    # Query for games where the nuggets were playing
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=id) # this is just an object
    # The first DataFrame of those returned is what we want.
    return gamefinder.get_data_frames()[0]

def get_team_games_years(team_name: str, years: list[int], start = None):
    '''
    '''
    id = get_team_dict(team_name)['id'] #you'll have to fix this
    games = leaguegamefinder.LeagueGameFinder(team_id_nullable=id).get_data_frames()[0] 
    # both parts of the season, start and end
    if start is None:
        years = [str(y) for y in years]
        games = games[games.SEASON_ID.str[-4:].isin(years)]
    elif start:
        years = ['1' + str(y) for y in years]
        games = games[games.SEASON_ID.str[:5].isin(years)]
    else:
        years = ['2' + str(y) for y in years]
        games = games[games.SEASON_ID.str[:5].isin(years)]
    return games


#def get_team_game_single(game_id):
#    '''
#    '''
    

def get_opp_team_game(df, opp_team_name: str):
    '''
    '''
    return df[df.MATCHUP.str.contains(op_team.upper())]

def get_single_game(df, game_id: int):
    return playbyplay.PlayByPlay(game_id).get_data_frames()[0]

def get_points_in_single_game(player_name: str, game_id: int, team: str):
    '''
    '''
    name_len, p = len(player_name), re.compile('\(\d+')
    df = playbyplay.PlayByPlay(game_id).get_data_frames()[0]
    # 1 is for field goals and 3 is for free throws
    points = df.loc[df['EVENTMSGTYPE'].isin([1,3])]
    if np.any(points.VISITORDESCRIPTION.str.contains(player_name)):
        final_string = points[points.VISITORDESCRIPTION.str[:name_len] == name]['VISITORDESCRIPTION'].iloc[-1]
        result = p.search(final_string).group(0)
        return result
    else:
        return 0

def get_box_adv_v2(game_id: str):
    '''
    '''
    return BoxScoreAdvancedV2(game_id).get_data_frames()[0]

def get_box_adv_v3(game_id: str):
    '''
    '''
    return BoxScoreAdvancedV3(game_id).get_data_frames()[0]

def get_box_trad_v2(game_id: str):
    '''
    '''
    return BoxScoreTraditionalV2(game_id).get_data_frames()[0]


def get_box_trad_v3(game_id: str):
    '''
    '''
    return BoxScoreTraditionalV3(game_id).get_data_frames()[0]


def get_game_logs(player_id, season = '2022-23'):
    '''
    '''
    return PlayerGameLogs(player_id_nullable = player_id, season_nullable = season).get_data_frames()[0]

def get_player_box(games, player_name: str, version = 'v3'):
    '''
    '''
    player_box = []
    for game_id in games['GAME_ID']:
        box_score = get_box_trad_v3(game_id)
        player_game_data = box_score[box_score.firstName == player_name.split()[0]]
        player_box.append(player_game_data)
    player_box = pd.concat(player_box)
    return player_box

def plot_pmf(b, var):
    '''
    Freeze the distribution and display the frozen pmg
    '''
    mu = b[var].mean()
    x = np.arange(poisson.ppf(0.01, mu),
                  poisson.ppf(0.99, mu))
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf');
    ax.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5);

def plot_stat_over_games(b, vars: list[str]):
    '''
    '''
    for key in vars:
        plt.plot(box[vars].values[::-1])
    plt.legend(vars)


# This resource describes some distributions and when they are used. https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119197096.app03
def plot_distributions(b, vars):
    
    mins_loc = list(b.columns).index('minutes') + 1

    # field goals made has a strange distribution becuase you get either 2 or
    
    #keys = ['freeThrowsMade']
    for key in vars:
        data = b.iloc[:,mins_loc:][key]
        n, bins, _ = plt.hist(data, bins = data.max(), alpha = .2); #.hist()
    #print(bins)
    plt.legend(vars)
# then the first bin is [1, 2) (including 1, but excluding 2) and the second [2, 3). The last bin, however, is [3, 4], which includes 4.

def get_poisson(b, var, val, less_than = True):
    
    mu = b[var].mean()
    if less_than :
         
        data = poisson.stats(mu, loc=0, moments='mv') 
        return data + (poisson.cdf(val, mu),)
    else:
        return data + (1 - poisson.cdf(val, mu),)
        
def get_game_logs(player_id, season = '2022-23'):
    '''
    '''
    return PlayerGameLogs(player_id_nullable = player_id, season_nullable = season).get_data_frames()[0]

def get_data(box,var, n_games):
    '''
    '''
    if 'minutes' in box.columns:
        min_index = list(box.columns).index('minutes')
    elif 'MIN' in box.columns:
        min_index = list(box.columns).index('MIN')
    box = box.iloc[:,min_index:]

    y = box[var]
    box = box.drop(columns = var)
    games_to_concat  = []
    l = len(box)
    for i in range(1, n_games  + 1):
        bottom_remove = n_games - i
        games_to_concat.append(box.iloc[i:l - bottom_remove].add_suffix('_%d'%i).reset_index(drop = True))
    data = pd.concat(games_to_concat, axis = 1)
    data['next_game_%s'%var] = y[:-n_games]
    return data

def return_floats(l):
    return [float(i) for i in l]



def print_performance_stats(model,X_train, X_test, y_train, y_test):
    '''
    '''
    # if the var is float don't change it to int otherwise you lose data
    y_pred =  model.predict(X_test)
    train_predict =  model.predict(X_train)
    y_train = return_floats(y_train)
    y_test = return_floats(y_test)
    
    print('TRAINING')
    print('-'* 30)
    # accuracy: {accuracy_score(y_train, train_predict)}\n
    print(f'r2: {r2_score(y_train,train_predict)}
    \nMSE: {mean_squared_error(y_train,train_predict)}
    \nRMSE: {mean_squared_error(y_train,train_predict, squared = False)}
    \nMAE {mean_absolute_error(y_train,train_predict)} ')
    print()
    print('TESTING')
    print('-'* 30)
    # accuracy: {accuracy_score(y_test, y_pred)}\n
    print(f'r2: {r2_score(y_test,y_pred)}\nMSE: {mean_squared_error(y_test,y_pred)}\nRMSE: {mean_squared_error(y_test,y_pred, squared = False)}\nMAE {mean_absolute_error(y_test,y_pred)} ')

def plot_prediction(model, X_train, X_test, y_train, y_test, model_name, var):
    test_pred = model.predict(X_test)#[int(pred) for pred in ]
    train_pred = model.predict(X_train)#[int(pred) for pred in ]
    fig, axs = plt.subplots(nrows = 2, ncols=2, figsize=(16, 8))
    #print(y_train.size, train_pred.size)
    #print(y_test.size, test_pred.size)
    PredictionErrorDisplay.from_predictions(
        y_train,
        y_pred=train_pred,
        kind="actual_vs_predicted",
        subsample=100,
        ax=axs[0,0],
        random_state=0,
    );
    axs[0,0].set_title("TRAINING: Actual vs. Predicted values")
    
    PredictionErrorDisplay.from_predictions(
        y_train,
        y_pred=train_pred,
        kind="residual_vs_predicted",
        subsample=100,
        ax=axs[0,1],
        random_state=0,
    );
    axs[0,1].set_title("TRAINING: Residuals vs. Predicted Values")
    
    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred=test_pred,
        kind="actual_vs_predicted",
        subsample=100,
        ax=axs[1,0],
        random_state=0,
    );
    axs[1,0].set_title("TESTING: Actual vs. Predicted values")
    
    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred=test_pred,
        kind="residual_vs_predicted",
        subsample=100,
        ax=axs[1,1],
        random_state=0,
    );
    axs[1,1].set_title("TESTING: Residuals vs. Predicted Values")
    
    fig.suptitle(F"{model_name} ON \'{var}\': Plotting cross-validated predictions")
    plt.tight_layout()
    plt.show()

