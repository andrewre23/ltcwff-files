import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import path

# change this to the directory where the csv files that come with the book are
# stored
# on Windows it might be something like 'C:/mydir'

DATA_DIR = './data'

###############
# distributions
###############

pg = pd.read_csv(path.join(DATA_DIR, 'player_game_2017_sample.csv'))
pbp = pd.read_csv(path.join(DATA_DIR, 'play_data_sample.csv'))


# 6.1.a
g = sns.displot(pbp, x='yards_gained', kind='kde', fill=True)
plt.show()

# 6.1.b
g = sns.displot(pbp.query("down <= 3"), x='yards_gained', hue='down', kind='kde', fill=True)
plt.show()

# 6.1.c
g = sns.displot(pbp.query("down <= 3"), x='yards_gained', col='down', kind='kde', fill=True)
plt.show()

# 6.1.d
g = sns.displot(pbp, x='yards_gained', col='down', kind='kde', row='posteam', fill=True)
plt.show()

# 6.1.e
g = sns.displot(pbp, x='yards_gained', col='down', kind='kde', row='posteam', hue='posteam', fill=True)
plt.show()


# 6.2.a
g = sns.relplot(x='carries', y='rush_yards', hue='pos', data=pg)
g.figure.subplots_adjust(top=0.9)
g.figure.suptitle('Carries vs. Rushing Yards by Position')
#g.set(xlim=(-10, 40), ylim=(0, 0.014))
g.set_xlabels('carries')
g.set_ylabels('rush_yards')
plt.show()

# 6.2.b
pg.set_index(['gameid', 'player_id'], inplace=True)
pg.groupby('pos')['rush_yards'].sum() / pg.groupby('pos')['carries'].sum()

