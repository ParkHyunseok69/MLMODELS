import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from nba_api.stats.endpoints import PlayerGameLog

player_id = 1627750  
season = "2023-24"  

game_log = PlayerGameLog(player_id=player_id, season=season)
df = game_log.get_data_frames()[0]  # Get the stats as a DataFrame
df.drop(columns=['SEASON_ID', 'Player_ID', 'Game_ID', 'GAME_DATE', 'MATCHUP', 'WL',
       'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
       'VIDEO_AVAILABLE'], inplace=True)

# Shift PTS column to make the next game's points the target (y)
df['Next_Game_PTS'] = df['PTS'].shift(-1)

# Drop last row since it has no next game to predict
df = df.dropna()

X = df[['FGM', 'FG_PCT', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'MIN', 'PLUS_MINUS']]
y = df['Next_Game_PTS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(pd.DataFrame([X.iloc[-1].values], columns=X.columns))

print(f'Predicted_PTS: {round(prediction[0])}')



