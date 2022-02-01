import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random_strategy
import deterministic_strategy
import decision_trees_strategy
import decision_trees_sklearn_strategy
import warnings
warnings.filterwarnings('ignore')
from ROC_AUC import print_calculate
from visualize import visualize_predictive_features
from accuracy import get_accuracy_deterministic, get_accuracy_random

# read and prepare training data
df = pd.read_csv('Training_football.csv')
training_data = df[['HomeTeam', 'AwayTeam', 'outcome_by_HOME', 'RANKINGHOME', 'RANKING AWAY', 'LAST_GAME_RHOMETEAM',
                    'LAST_GAME_RAWAYTEAM']].copy()
training_data.columns = ['HT', 'AT', 'Res', 'RH', 'RA', 'LGH', 'LGA']
training_data.Res = training_data.Res.replace('L', -1)
training_data.Res = training_data.Res.replace('W', 1)
training_data.Res = training_data.Res.replace('D', 0)

# visualize training data
sns.pairplot(training_data)
plt.show()
sns.heatmap(training_data[['Res', 'RH', 'RA', 'LGH', 'LGA']].corr())
plt.show()

# read and prepare test data
df_test = pd.read_csv('Test_football.csv')
test_data = df_test[['HomeTeam', 'AwayTeam', 'ACTUAL_RESULT', 'RANKING_HOME', 'RANKING_AWAY', 'LAST_GAME_RHOMETEAM',
                     'LAST_GAME_RAWAYTEAM']].copy()
test_data.columns = ['HT', 'AT', 'Res', 'RH', 'RA', 'LGH', 'LGA']
test_data.Res = test_data.Res.replace('L', -1)
test_data.Res = test_data.Res.replace('W', 1)
test_data.Res = test_data.Res.replace('D', 0)
size = len(test_data['Res'])
actual = test_data['Res'].to_numpy()

# for ROC curves 2 versions with win + draw : lose and win : draw + lose
actual_win = np.empty(size, int)
for i in range(size):
    if actual[i] == 0:
        actual_win[i] = 1
    else:
        actual_win[i] = actual[i]

actual_lose = np.empty(size, int)
for i in range(size):
    if actual[i] == 0:
        actual_lose[i] = -1
    else:
        actual_lose[i] = actual[i]

# random strategy
random_result = random_strategy.get_result(size)

random_result_win = np.empty((size, 2), float)
for i in range(size):
    random_result_win[i] = np.array([random_result[i][0] + random_result[i][1], random_result[i][2]])

random_result_lose = np.empty((size, 2), float)
for i in range(size):
    random_result_lose[i] = np.array([random_result[i][0], random_result[i][1] + random_result[i][2]])

print_calculate(random_result_win, actual_win, "Random strategy (Win 1/Draw : Win 2)")
print_calculate(random_result_lose, actual_lose, "Random strategy (Win 1 : Draw/Win 2)")
print("Random strategy accuracy: ", get_accuracy_random(actual, random_result))

# deterministic strategy
deterministic_result = deterministic_strategy.get_result(size)
print_calculate(deterministic_result, actual_win, "Deterministic strategy (Win 1/Draw : Win 2)", False)
print_calculate(deterministic_result, actual_lose, "Deterministic strategy (Win 1 : Draw/Win 2)", False)
print("Deterministic strategy accuracy: ", get_accuracy_deterministic(actual))

# decision tree strategy
decision_trees_correct = decision_trees_strategy.get_result(training_data[['RH', 'RA', 'LGH', 'LGA', 'Res']].to_numpy(),
                                                            test_data[['RH', 'RA', 'LGH', 'LGA', 'Res']].to_numpy())
print("Decision trees strategy: ", decision_trees_correct / size)

# decision tree strategy training accuracy
decision_trees_correct = decision_trees_strategy.get_result(training_data[['RH', 'RA', 'LGH', 'LGA', 'Res']].to_numpy(),
                                                            training_data[['RH', 'RA', 'LGH', 'LGA', 'Res']].to_numpy())
print("Decision trees strategy training: ", decision_trees_correct / len(training_data['Res']))

training_data_win = np.empty(len(training_data['Res']), int)
for i in range(len(training_data['Res'])):
    if training_data['Res'][i] == 0:
        training_data_win[i] = 1
    else:
        training_data_win[i] = training_data['Res'][i]

training_data_lose = np.empty(len(training_data['Res']), int)
for i in range(len(training_data['Res'])):
    if training_data['Res'][i] == 0:
        training_data_lose[i] = 1
    else:
        training_data_lose[i] = training_data['Res'][i]

# decision tree sklearn strategy and ROC curves
decision_trees_sklearn_strategy.get_result(training_data[['RH', 'RA', 'LGH', 'LGA']].to_numpy(),
                                           training_data_win,
                                           test_data[['RH', 'RA', 'LGH', 'LGA']].to_numpy(),
                                           actual_win, "Decision trees strategy (Win 1/Draw : Win 2)")

decision_trees_sklearn_strategy.get_result(training_data[['RH', 'RA', 'LGH', 'LGA']].to_numpy(),
                                           training_data_lose,
                                           test_data[['RH', 'RA', 'LGH', 'LGA']].to_numpy(),
                                           actual_lose, "Decision trees strategy (Win 1 : Draw/Win 2)")

# visualize predictive features (optional)
win_data = training_data[training_data['Res'] == 1]
lose_data = training_data[training_data['Res'] == -1]
draw_data = training_data[training_data['Res'] == 0]
visualize_predictive_features(win_data, lose_data, draw_data)
visualize_predictive_features(win_data, lose_data, draw_data, False)
