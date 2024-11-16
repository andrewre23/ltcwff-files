import pandas as pd
import statsmodels.formula.api as smf
from os import path

DATA_DIR = './data'

## 7.1

# from example
df = pd.read_csv(path.join(DATA_DIR, 'play_data_sample.csv'))
df = df.loc[(df['play_type'] == 'run') | (df['play_type'] == 'pass')]
df['offensive_td'] = ((df['touchdown'] == 1) & (df['yards_gained'] > 0))
df['offensive_td'] = df['offensive_td'].astype(int)
df['yardline_100_sq'] = df['yardline_100'] ** 2
df[['offensive_td', 'yardline_100', 'yardline_100_sq']].head()
model = smf.ols(formula='offensive_td ~ yardline_100 + yardline_100_sq', data=df)
results = model.fit()

def prob_of_td(yds):
    b0, b1, b2 = results.params
    return (b0 + b1*yds + b2*(yds**2))

df['offensive_td_hat'] = results.predict(df)
df[['offensive_td', 'offensive_td_hat']].head()

# 7.1.a
df['offensive_td_hat_alt'] = df['yardline_100'].apply(prob_of_td)
df[['offensive_td', 'offensive_td_hat', 'offensive_td_hat_alt']].head() # match

# 7.1.b
model = smf.ols(formula='offensive_td ~ yardline_100 + yardline_100_sq + ydstogo', data=df)
results = model.fit()

# 7.1.c
model = smf.ols(formula='offensive_td ~ yardline_100 + yardline_100_sq + C(down)', data=df)
results = model.fit()

# 7.1.d
for down in df['down'].unique():
    df[f'is{int(down)}'] = df.down == down
model = smf.ols(formula='offensive_td ~ yardline_100 + yardline_100_sq + is2 + is3 + is4', data=df)
results = model.fit()


## 7.2
import random
from pandas import DataFrame, Series
import statsmodels.formula.api as smf

# from example
coin = ['H', 'T']
df = DataFrame(index=range(100))
df['guess'] = [random.choice(coin) for _ in range(100)]
df['result'] = [random.choice(coin) for _ in range(100)]
df['right'] = (df['guess'] == df['result']).astype(int)
model = smf.ols(formula='right ~ C(guess)', data=df)
results = model.fit()

# 7.2.a
def run_sim_get_p_value(n:int = 100):
    """Flip coin 'n' times, run regression, return p-value of guess."""
    coin = ['H', 'T']
    df = DataFrame(index=range(n))
    df['guess'] = [random.choice(coin) for _ in range(n)]
    df['result'] = [random.choice(coin) for _ in range(n)]
    df['right'] = (df['guess'] == df['result']).astype(int)
    model = smf.ols(formula='right ~ C(guess)', data=df)
    results = model.fit()
    return results.pvalues['C(guess)[T.T]']

# 7.2.b
results = Series(run_sim_get_p_value(1000))
print(results.mean())

# 7.2.c
p=0.05
def runs_till_threshold(i, p=0.05):
    pvalue = run_sim_get_p_value()
    if pvalue < p:
        return i
    else:
        return runs_till_threshold(i + 1, p)

results = Series([runs_till_threshold(1) for _ in range(100)])
results.mean()

# 7.2.d
import math
results.mean()                          # 18.58
mean = (1 - p) / p                      # 18.99
results.median()                        # 13.5
median = -(math.log(2) / math.log(1-p)) # 13.51


## 7.3
import pandas as pd
import math
import statsmodels.formula.api as smf
from os import path

# from example
df = pd.read_csv(path.join(DATA_DIR, 'play_data_sample.csv'))
df = df.loc[(df['play_type'] == 'run') | (df['play_type'] == 'pass')]
df['offensive_td'] = ((df['touchdown'] == 1) & (df['yards_gained'] > 0))

# 7.3.a
model = smf.ols(formula=
        """
        wpa ~ offensive_td + interception + yards_gained + fumble
        """, data=df)
results = model.fit()
# interceptions are worse

# 7.3.b
model = smf.ols(formula=
        """
        wpa ~ offensive_td + interception + yards_gained + fumble_lost
        """, data=df)
results = model.fit()
# closer, but interceptions still worse

# 7.4
# if b2 is positive, then we expect that rookies outproduce their ADP
# if b2 is negative, then we expect that rookies under-perform their ADP
# if b2 ~0, then the rookie effect is being accurately captures in ADP

# 7.5
import pandas as pd
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from os import path

df = pd.read_csv(path.join(DATA_DIR, 'player_game_2017_sample.csv'))

xvars = ['carries', 'rush_yards', 'rush_fumbles', 'rush_tds', 'targets',
         'receptions', 'rec_yards', 'raw_yac', 'rec_fumbles', 'rec_tds',
         'ac_tds', 'rec_raw_airyards', 'caught_airyards', 'attempts',
         'completions', 'pass_yards', 'pass_raw_airyards', 'comp_airyards',
         'timeshit', 'interceptions', 'pass_tds', 'air_tds']
yvar = 'pos'

# 7.5.a
from typing import List
def get_cv_score_for_agg(df:DataFrame, group_index:List[str]):
    df_grp = df.groupby(group_index)[xvars].mean()
    df_grp['pos'] = df.groupby(group_index)[yvar].first()
    model = RandomForestClassifier(n_estimators=100)
    scores = cross_val_score(model, df_grp[xvars], df_grp[yvar], cv=10)
    return scores.mean()

for index in (
        ['player_id'],
        ['player_id', 'gameid'],
):
    print(f"index:{index} : {get_cv_score_for_agg(df, index)}")
# player is better than player/week (week-to-week volatility)

# 7.5.b
df_med = df.groupby('player_id')[xvars].median()
df_max = df.groupby('player_id')[xvars].max()
df_min = df.groupby('player_id')[xvars].min()
df_mean = df.groupby('player_id')[xvars].mean()

df_med.columns = [f'{x}_med' for x in df_med.columns]
df_max.columns = [f'{x}_max' for x in df_max.columns]
df_min.columns = [f'{x}_min' for x in df_min.columns]
df_mean.columns = [f'{x}_mean' for x in df_mean.columns]

df_mult = pd.concat([df_med, df_max, df_min, df_mean], axis=1)
xvars_mult = list(df_mult.columns)

df_mult['pos'] = df.groupby('player_id')[yvar].first()

model_b = RandomForestClassifier(n_estimators=100)
scores_b = cross_val_score(model_b, df_mult[xvars_mult], df_mult[yvar], cv=10)
print(scores_b.mean())
