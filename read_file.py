import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("game_data.csv")

df.drop(columns=['url','black','white','uuid','tcn','fen','initial_setup','rules','time_control','end_time'],inplace=True)
#removing time control because time_class is the same
print(df.head())
df['time_class'].replace(to_replace=['blitz','rapid','bullet'],value=[1,2,3],inplace=True)
df['player_color'].replace(to_replace=['black','white'],value=[0,1],inplace=True)
df['opp_color'].replace(to_replace=['black','white'],value=[0,1],inplace=True)
df['winner'].replace(to_replace=['black','white','none'],value=[0,1,2],inplace=True)
df['did_player_win'] = (df['winner']==df['player_color'])
df['rated'].replace(to_replace=[False,True],value=[0,1],inplace=True)
df['did_player_win'].replace(to_replace=[False,True],value=[0,1],inplace=True)
df['difference'] = df['player_rating']-df['opp_rating']


print(df.head())

x = df[['rated','time_class','player_color','opp_color','difference']]
y = df['did_player_win']

x = preprocessing.StandardScaler().fit(x).transform(x)

X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=4, test_size=0.2)

LR_model = LogisticRegression().fit(X_train,y_train)
predicted = LR_model.predict(X_test)
predicted_probability = LR_model.predict_proba(X_test)


print(predicted)
print(predicted_probability)

print(LR_model.score(X_test,y_test))

print(np.array(y_test))
print(predicted)