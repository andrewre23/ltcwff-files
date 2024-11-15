import requests
from pandas import DataFrame
import pandas as pd

adp_url = 'https://api.myfantasyleague.com/2019/export?TYPE=adp&JSON=1'
player_info_url = 'https://api.myfantasyleague.com/2019/export?TYPE=players&JSON=1'

adp_resp = requests.get(adp_url)
player_info_resp = requests.get(player_info_url)

df_adp = DataFrame(adp_resp.json()['adp']['player']).set_index('rank')
df_player_info = DataFrame(player_info_resp.json()['players']['player']).set_index('id')

df_top_200 = (df_adp
              .merge(df_player_info, how='left', left_on='id', right_index=True)
              .query("position not in ['LB', 'S', 'DE', 'DT', 'CB']")
              .head(200))

