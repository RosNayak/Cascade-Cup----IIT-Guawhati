import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

data = pd.read_csv('../input/cascade-cup/train_age_dataset.csv')
data.drop('Unnamed: 0', axis=1, inplace=True)

test = pd.read_csv('../input/cascade-cup/test_age_dataset.csv')
test.drop('Unnamed: 0', axis=1, inplace=True)

def cat(col):
    if col[0] == 0:
        return 0
    else:
        return 1
    
data['num_of_hashtags_per_action_b'] = data[['num_of_hashtags_per_action']].apply(cat, axis=1)
test['num_of_hashtags_per_action_b'] = test[['num_of_hashtags_per_action']].apply(cat, axis=1)

data['emoji_count_per_action_b'] = data[['emoji_count_per_action']].apply(cat, axis=1)
test['emoji_count_per_action_b'] = test[['emoji_count_per_action']].apply(cat, axis=1)

def logT(col):
    return np.log(col[0] + 1)

data['content_views'] = data[['content_views']].apply(logT, axis=1)
test['content_views'] = test[['content_views']].apply(logT, axis=1)

data['num_of_comments'] = data[['num_of_comments']].apply(logT, axis=1)
test['num_of_comments'] = test[['num_of_comments']].apply(logT, axis=1)

tData = data.drop('age_group', axis=1)
labels = pd.DataFrame(data.age_group, columns=['age_group'])

train, val, tlabels, vlabels = train_test_split(tData, labels, test_size=0.15, stratify=labels.age_group)

temp = train.drop(['avgCompletion', 'avgTimeSpent', 'avgDuration', 'slot1_trails_watched_per_day', 'slot2_trails_watched_per_day', 'slot3_trails_watched_per_day', 'slot4_trails_watched_per_day', 'avgt2', 'weekdays_trails_watched_per_day'], axis=1)
tVal = val.drop(['avgCompletion', 'avgTimeSpent', 'avgDuration', 'slot1_trails_watched_per_day', 'slot2_trails_watched_per_day', 'slot3_trails_watched_per_day', 'slot4_trails_watched_per_day', 'avgt2', 'weekdays_trails_watched_per_day'], axis=1)
tTemp = test.drop(['avgCompletion', 'avgTimeSpent', 'avgDuration', 'slot1_trails_watched_per_day', 'slot2_trails_watched_per_day', 'slot3_trails_watched_per_day', 'slot4_trails_watched_per_day', 'avgt2', 'weekdays_trails_watched_per_day'], axis=1)

xgb = XGBClassifier(eta=0.05, max_depth=12, subsample=0.9, n_estimators=215)
xgb.fit(temp, tlabels)
p = xgb.predict(tVal)
print(f1_score(vlabels, p, average='micro'))

p = xgb.predict(tTemp)
preds = pd.DataFrame(list(p), columns=['prediction'])
preds.to_csv('file.csv', index=False)