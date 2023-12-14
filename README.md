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
First, let's clean the data and find the features for the regression. The dataframe below has being clean as the same process in my previous project. Here are the remaining columns:\

<u>['gameid', 'datacompleteness', 'league', 'year', 'split', 'playoffs',
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
           'dragons']



As only a well prepared and developed team, we should use features that can reflect a team's overall ability.Also, as it is a mid-game prediction, we should not use the final statistics such as 'kills' and 'assists' which are collected post game. Instead, we should use statistics from the first half of the game as features for this model. Teams which obtain the first dragon are more likely to snowball their way to victory as they have more control over the map.
Combined, we should use features below:\
_'goldat15', 'xpat15', 'csat15','opp_goldat15', 'opp_xpat15', 'opp_csat15', 'golddiffat15','xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15','opp_killsat15', 'opp_assistsat15', 'opp_deathsat15', 'firstdragon',and 'firstbaron'._ And here is the first few lines of the dataframe:
<div>
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
'**xpat15**': a quantitative variable depicting difference in expeience at 15 minuets. This is a very convincing metric in measuring the team's difference in developing their champions. The higher the number is, the greater the difference if development there is between the two teams at 15 minuets.\
'**firstdragon**': this is a boolean variable that stays whether the teams has the firstdragon or not. The team with the first dragon has a higher likelihood of obtaining 3 more dragons and thus the dragon soul.\
'**golddiffat15**': a quantitative variable depicting the difference in gold at 15 minuets. The team that slain more monsters and enemies will have more gold. And with more assets, champions can equip better gear. Thus, this is a good indicator of team strength\
#### Train_test_split & pipeline
Here I have split the datatrame into dataframe for test and dataframe for trainning with a test_size of **0.3**. This means that 30% of the data is allocated for testing purposes while the remaining 70% is used for training the model. The split is crucial for evaluating the model's performance on unseen data, ensuring that it not only learns from the training set but also generalizes well to new, unseen data. In this setup, I first fix any missing data in the 'xpat15' and 'golddiffat15' columns. Then, I use these columns along with 'firstdragon' to train a logistic regression model. This model learns to predict outcomes based on how much experience and gold teams have at 15 minutes into the game, and whether they captured the first dragon. The whole process is automated in a pipeline to make it easier and more efficient.\
Here is the result: ${logit(dragon soul)}=-6e-05\text{firstbaron} + 0.0002{firstdragon} - 0.0{golddiffat15} -5.7238347596014364e-08$\
Now, we will use this model and apply this on the testing part of the dataframe. Following are the four performance metrics for this model:
| Metric    | Value                  |
|-----------|------------------------|
| Accuracy  | 0.8178726035868893     |
| Precision | 0.0                    |
| Recall    | 0.0                    |
| F1 Score  | 0.0                    |
Though the accuracy is high, the other metrics indicate that it is not effectively identifying the positive class. It seems like this model has severely overfitted. Resulting in an incapability of applying the model to any other data. We will fix this in the final model.

### Final Model
For the final model, I want to include _firstbaron,csdiffat15,killsat15.assistsat15,deathsat15_.
'**firstbaron**': a boolean variable that records the first baron. Baron is a hardest boss to slain in the game and requires the whole team's effort. Thus, this is a good indicator of team's strength.
'**csdiffat15**': a quantitative variable that records the difference in creep score. A high cs difference at 15 minutes would imply that the lane has priority over the opponent, and therefore has the chance of occupying the landscapes for dragons and stopping opponents from getting the dragon soul.
 As for **killsat15**, **assistsat15**, **deathsat15**, I would like to combine them into 1 column called **kdaat15** following this formula: $kda = \frac{\text{kill}+\text{assit}}{\text{daeth}}$. This metric is capable of displaying how good a player is doing in a game. The higher the KDA, the more it implies that it is difficult for the opponents to kill this player, meaning the opponents will be likely to concede the dragon and aim for killing the player instead.
#### Feature Engineering
In this part of our analysis, I've implemented a series of feature engineering steps, tailored to optimize our model's performance. Let me walk you through these steps:
1. **Preprocessing with `ColumnTransformer`:**
   - I use this to apply specific preprocessing techniques to different subsets of our features, enhancing their relevance and impact on the model.

2. **Handling Numerical Features (`'xpdiffat15', 'csdiffat15', 'golddiffat15', 'kdaat15')**:
   - **Imputation with Median**: Here, I address any missing values by filling them with the median value. This approach is robust against outliers.
   - **Polynomial Features**: I then generate polynomial features to capture complex interactions and non-linear relationships, which are crucial for a nuanced understanding of our data.
   - **Standardization**: I standardize these features to have a mean of 0 and a standard deviation of 1, ensuring that our logistic regression model treats all features equally.

3. **Encoding Boolean Features (`'firstbaron', 'firstdragon')**:
   - **Ordinal Encoding**: I convert these categorical features into a numerical format, making them interpretable for our model.

4. **Passing Through Remaining Features**:
   - With `remainder='passthrough'` in our `ColumnTransformer`, I ensure that any other features not explicitly mentioned are left unchanged, preserving their original structure.

5. **Configuring the Classifier (Logistic Regression)**:
   - Finally, I use these preprocessed features to train our logistic regression classifier. I've set a high `max_iter` value to ensure convergence and a small `tol` value for precision.

To sum up, my pipeline involves meticulous imputation, scaling, generation of polynomial features, and encoding of boolean variables. This prepares our data comprehensively for the logistic regression model, aiming to capture as much complexity and variation in the data as possible for accurate predictions.
#### Hyperparameter
In this section of our analysis, I have focused on optimizing our logistic regression model by fine-tuning its hyperparameters:

1. **Defining a Range of Hyperparameters**:
   - I've identified key hyperparameters that influence the performance of our logistic regression model. These include:
     - `classifier__C`: The inverse of regularization strength. I've chosen values like 0.01, 0.1, 1, 10, and 100 to explore a wide range from strong to weak regularization.
     - `classifier__solver`: The algorithm used for optimization. Options like 'newton-cg', 'lbfgs', 'sag', and 'saga' offer different approaches to reaching convergence.
     - `classifier__penalty`: The norm used in the penalization. I've fixed this to 'l2', which is suitable for many logistic regression problems.
     - `classifier__max_iter`: The maximum number of iterations taken for the solvers to converge. I've varied this from 100 to 2000 to explore different convergence behaviors.

2. **Grid Search with Cross-Validation**:
   - To systematically evaluate the combinations of these hyperparameters, I've used `GridSearchCV`. This approach exhaustively tries every combination of hyperparameters, ensuring that we do not miss the optimal setting.
   - I've set the cross-validation (`cv`) parameter to 7, which means the training set is split into 7 smaller sets; the model is trained on 6 and validated on the 7th, and this is repeated for all combinations.

3. **Fitting the Model**:
   - I've then fit this grid search setup on our modified training data (`X_train_new` and `y_train_new`).
   - This process not only identifies the best hyperparameters but also trains the model on these optimal settings.

4. **Selecting the Best Model**:
   - After the grid search, the `best_estimator_` attribute of `GridSearchCV` gives us the model that performed the best during cross-validation.
   - This model, `best_model`, is now our optimal logistic regression model, fine-tuned to our specific dataset and ready for further evaluation or deployment.

And thus, here is the formula for my final regression mode:
$\text{logit}(\text{dragon soul}) = -3.6060 + (0.3421)*\text{xpdiffat15} + (0.5512)*\text{csdiffat15} + (0.4734)*\text{golddiffat15} + (0.3611)*\text{kdaat15} + (0.3522)*\text{xpdiffat15 csdiffat15} + (0.4767)*\text{xpdiffat15 golddiffat15} + (0.5054)*\text{xpdiffat15 kdaat15} + (0.5910)*\text{csdiffat15 golddiffat15} + (0.8880)*\text{csdiffat15 kdaat15} + (0.6805)*\text{golddiffat15 kdaat15} + (0.3227)*\text{firstbaron} + (0.0000)*\text{firstdragon}$
Simialr to the baseline model. Now we should fit this into the testing part of the dataframe and below is the result:
| Metric    | Value                 |
|-----------|-----------------------|
| Accuracy  | 0.8214285714285714    |
| Precision | 0.48214285714285715   |
| Recall    | 0.023417172593235037  |
| F1 Score  | 0.04466501240694789   |
As seem, all the metrics has seem increase. Though Accuracy for this model has only increased slightly. Precision, Recall, and F1 score has all increase dramatically through more feature engineering and increasing useful features. This indicates that the model is now better at correctly identifying positive cases (as shown by the improved Precision and Recall) and balancing these two aspects (as reflected in the higher F1 Score). The enhancements in feature engineering and the addition of relevant features have contributed significantly to the model's ability to make more accurate and reliable predictions, especially in distinguishing between different classes effectively.

### Fairness Analysis
To see if the model is biased, we need to create a permutation test. Though all are esports competitions, but playoff games are typically more intense and representative of the League of Legends pro-scene. Thus, I will conduct a hypothesis testing below to see if this model is biased or not:
##### Group X and Group Y
* Group X: Playoff games (playoff column value = 1)\
* Group Y: Non-playoff games (playoff column value = 0)
##### Evaluation Metric
* Precision: Measures the proportion of correctly identified positive instances among all instances predicted as positive by the model.
##### Null and Alternative Hypotheses
* Null Hypothesis (H0): The model is fair regarding playoff status. The precision for predicting playoff games (Group X) is the same as for predicting non-playoff games (Group Y), and any differences in precision are due to random chance.
* Alternative Hypothesis (H1): The model is unfair regarding playoff status. The precision for predicting playoff games (Group X) significantly differs from the precision for predicting non-playoff games (Group Y).
##### Choice of Test Statistic
* Difference in Precision: The test statistic is the difference in precision between playoff games and non-playoff games, calculated as Precision(Group X) - Precision(Group Y).
##### Significance Level
* A standard significance level (α) of 0.05 is chosen. A p-value less than 0.05 will lead to rejecting the null hypothesis, suggesting significant differences in model performance between playoff and non-playoff games.

Below is a code snippet of the permutation test:
playoff_games = dff[dff['playoffs'] == 1]
playoff_games = playoff_games[['dragons','firstbaron', 'firstdragon', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'kdaat15']]
non_playoff_games = dff[dff['playoffs'] == 0]
non_playoff_games = non_playoff_games[['dragons','firstbaron', 'firstdragon', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'kdaat15']]
precision_x = precision_score(playoff_games['dragons'], best_model.predict(playoff_games.drop('dragons', axis=1)))
precision_y = precision_score(non_playoff_games['dragons'], best_model.predict(non_playoff_games.drop('dragons', axis=1)))
observed_diff = precision_x - precision_y
n_permutations = 1000
count = 0
for _ in range(n_permutations):
    test = dff.copy()
    test['playoffs'] = np.random.permutation(test['playoffs'])
    playoff_shuffle = test[test['playoffs'] == 1][['dragons','firstbaron', 'firstdragon', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'kdaat15']]
    non_playoff_shuffle = test[test['playoffs'] == 0][['dragons','firstbaron', 'firstdragon', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'kdaat15']]
    precision_playoff_shuffle = precision_score(playoff_shuffle['dragons'], best_model.predict(playoff_shuffle.drop('dragons', axis=1)))
    precision_non_playoff_shuffle = precision_score(non_playoff_shuffle['dragons'], best_model.predict(non_playoff_shuffle.drop('dragons', axis=1)))
    diff_shuffled = precision_playoff_shuffle - precision_non_playoff_shuffle#test statistic
    if diff_shuffled >= observed_diff:
        count += 1
p_value = count / n_permutations
<iframe src="permutation_test_histogram.html" width=800 height=600 frameBorder=0></iframe>

Thus, with a p-value that is significantly higher than typical alpha level of 0.05. We failed to reject the null hypothesis. Thus, there is sufficient prove that the prediction model is unbiased for both playoff and non-playoff games.