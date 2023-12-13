# Rift Mastery: Logistic Analysis of LoL Dragon Souls
**Name(s)**: Chris Luo

### Framing the Problem
***Dragon soul*** is a powerful team-wide bonus that is granted to a team after they haev succeessfully slain four Elemental Drakes - four powerful monster that requires the entire team to conquer. In the game of League of Legends, even in the pro scene, whoever acquire the Dragon Soul almost equals to declaring the victory. Thus, which team will acquire the Dragon Soul has become a crucial part of prediction for all League of Legends game forcasters. In this project we will deal with the following dataset: "2023_LoL_esports_match_data_from_OraclesElixir". Below are the first few lines of the dataset.\
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gameid</th>
      <th>datacompleteness</th>
      <th>url</th>
      <th>league</th>
      <th>year</th>
      <th>split</th>
      <th>playoffs</th>
      <th>date</th>
      <th>game</th>
      <th>patch</th>
      <th>...</th>
      <th>opp_csat15</th>
      <th>golddiffat15</th>
      <th>xpdiffat15</th>
      <th>csdiffat15</th>
      <th>killsat15</th>
      <th>assistsat15</th>
      <th>deathsat15</th>
      <th>opp_killsat15</th>
      <th>opp_assistsat15</th>
      <th>opp_deathsat15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ESPORTSTMNT06_2753012</td>
      <td>complete</td>
      <td>NaN</td>
      <td>LFL2</td>
      <td>2023</td>
      <td>Spring</td>
      <td>0</td>
      <td>2023-01-10 17:07:16</td>
      <td>1</td>
      <td>13.01</td>
      <td>...</td>
      <td>131.0</td>
      <td>322.0</td>
      <td>263.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ESPORTSTMNT06_2753012</td>
      <td>complete</td>
      <td>NaN</td>
      <td>LFL2</td>
      <td>2023</td>
      <td>Spring</td>
      <td>0</td>
      <td>2023-01-10 17:07:16</td>
      <td>1</td>
      <td>13.01</td>
      <td>...</td>
      <td>117.0</td>
      <td>-357.0</td>
      <td>-1323.0</td>
      <td>-43.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ESPORTSTMNT06_2753012</td>
      <td>complete</td>
      <td>NaN</td>
      <td>LFL2</td>
      <td>2023</td>
      <td>Spring</td>
      <td>0</td>
      <td>2023-01-10 17:07:16</td>
      <td>1</td>
      <td>13.01</td>
      <td>...</td>
      <td>162.0</td>
      <td>-479.0</td>
      <td>-324.0</td>
      <td>-26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ESPORTSTMNT06_2753012</td>
      <td>complete</td>
      <td>NaN</td>
      <td>LFL2</td>
      <td>2023</td>
      <td>Spring</td>
      <td>0</td>
      <td>2023-01-10 17:07:16</td>
      <td>1</td>
      <td>13.01</td>
      <td>...</td>
      <td>122.0</td>
      <td>200.0</td>
      <td>292.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ESPORTSTMNT06_2753012</td>
      <td>complete</td>
      <td>NaN</td>
      <td>LFL2</td>
      <td>2023</td>
      <td>Spring</td>
      <td>0</td>
      <td>2023-01-10 17:07:16</td>
      <td>1</td>
      <td>13.01</td>
      <td>...</td>
      <td>3.0</td>
      <td>-216.0</td>
      <td>-579.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
<p>129348 rows × 123 columns</p>
</div>

To achieve this, we need to build a regression model based on a various of features that can help us predict which side is likely to retireve the dragon soul. And since for forcasters, how likely the prediction is correct is the most essential part, accuracy will be used as the metric. At the same time, the other three common used metrics, precision, recall, and F1-score will all be used to assed if the model is bias or over/underfitted.\
First, let's clean the data and find the features for the regression. The dataframe below has being clean as the same process in my previous project. Here are the remaining columns:
Index(['gameid', 'datacompleteness', 'league', 'year', 'split', 'playoffs',
           'date', 'game', 'patch', 'participantid', 'side', 'position',
           'playername', 'playerid', 'teamname', 'teamid', 'champion', 'ban1',
           'ban2', 'ban3', 'ban4', 'ban5', 'gamelength', 'result', 'kills',
           'deaths', 'assists', 'teamkills', 'teamdeaths', 'doublekills',
           'triplekills', 'quadrakills', 'pentakills', 'firstblood',
           'firstbloodkill', 'firstbloodassist', 'firstbloodvictim', 'team kpm',
           'ckpm', 'firstdragon', 'firstbaron', 'barons', 'opp_barons',
           'inhibitors', 'opp_inhibitors', 'damagetochampions', 'dpm',
           'damageshare', 'damagetakenperminute', 'damagemitigatedperminute',
           'wardsplaced', 'wpm', 'wardskilled', 'wcpm', 'controlwardsbought',
           'visionscore', 'vspm', 'totalgold', 'earnedgold', 'earned gpm',
           'earnedgoldshare', 'goldspent', 'total cs', 'minionkills',
           'monsterkills', 'cspm', 'goldat10', 'xpat10', 'csat10', 'opp_goldat10',
           'opp_xpat10', 'opp_csat10', 'golddiffat10', 'xpdiffat10', 'csdiffat10',
           'killsat10', 'assistsat10', 'deathsat10', 'opp_killsat10',
           'opp_assistsat10', 'opp_deathsat10', 'goldat15', 'xpat15', 'csat15',
           'opp_goldat15', 'opp_xpat15', 'opp_csat15', 'golddiffat15',
           'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15',
           'opp_killsat15', 'opp_assistsat15', 'opp_deathsat15', 'kdaat15',
           'dragons'],
          dtype='object')



\In addition, As only a well prepared and developed team, we should use features that can reflect a team's overall ability.Also, as it is a mid-game prediction, we should not use the final statistics such as 'kills' and 'assists' which are collected post game. Instead, we should use statistics from the first half of the game as features for this model.\

Teams which obtain the first dragon are more likely to snowball their way to victory as they have more control over the map.

Combined, we should use features below:\
'goldat15', 'xpat15', 'csat15','opp_goldat15', 'opp_xpat15', 'opp_csat15', 'golddiffat15','xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15','opp_killsat15', 'opp_assistsat15', 'opp_deathsat15', 'firstdragon',and 'firstbaron'. And here is the first few lines of the dataframe:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gameid</th>
      <th>teamname</th>
      <th>dragons</th>
      <th>goldat15</th>
      <th>xpat15</th>
      <th>csat15</th>
      <th>opp_goldat15</th>
      <th>opp_xpat15</th>
      <th>opp_csat15</th>
      <th>golddiffat15</th>
      <th>xpdiffat15</th>
      <th>csdiffat15</th>
      <th>killsat15</th>
      <th>assistsat15</th>
      <th>deathsat15</th>
      <th>opp_killsat15</th>
      <th>opp_assistsat15</th>
      <th>opp_deathsat15</th>
      <th>firstdragon</th>
      <th>firstbaron</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>ESPORTSTMNT06_2753012</td>
      <td>Klanik Esport</td>
      <td>1</td>
      <td>22384.0</td>
      <td>29220.0</td>
      <td>498.0</td>
      <td>22914.0</td>
      <td>30891.0</td>
      <td>535.0</td>
      <td>-530.0</td>
      <td>-1671.0</td>
      <td>-37.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ESPORTSTMNT06_2753012</td>
      <td>MS Company</td>
      <td>0</td>
      <td>22914.0</td>
      <td>30891.0</td>
      <td>535.0</td>
      <td>22384.0</td>
      <td>29220.0</td>
      <td>498.0</td>
      <td>530.0</td>
      <td>1671.0</td>
      <td>37.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>ESPORTSTMNT06_2754023</td>
      <td>beGenius ESC</td>
      <td>0</td>
      <td>24771.0</td>
      <td>30084.0</td>
      <td>498.0</td>
      <td>24098.0</td>
      <td>29554.0</td>
      <td>532.0</td>
      <td>673.0</td>
      <td>530.0</td>
      <td>-34.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>ESPORTSTMNT06_2754023</td>
      <td>ViV Esport</td>
      <td>1</td>
      <td>24098.0</td>
      <td>29554.0</td>
      <td>532.0</td>
      <td>24771.0</td>
      <td>30084.0</td>
      <td>498.0</td>
      <td>-673.0</td>
      <td>-530.0</td>
      <td>34.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>34</th>
      <td>ESPORTSTMNT06_2755035</td>
      <td>Team du Sud</td>
      <td>1</td>
      <td>22945.0</td>
      <td>27423.0</td>
      <td>510.0</td>
      <td>24846.0</td>
      <td>28186.0</td>
      <td>452.0</td>
      <td>-1901.0</td>
      <td>-763.0</td>
      <td>58.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
<p>21558 rows × 20 columns</p>
</div>

### Baseline Model

For the first model, I will build a logistic regression model using _'xpat15', 'firstdragon', and 'golddiffat15'_ as they are strong indicator of team strength. \
'xpat15': 