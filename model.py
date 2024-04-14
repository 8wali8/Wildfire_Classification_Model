import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load

from sklearn import tree, preprocessing
import sklearn.ensemble as ske
import sklearn.neural_network as skn
from sklearn.model_selection import train_test_split

#dataframe is in data_preprocessing

from subprocess import check_output
print(check_output(["ls", "../Wildfire_Cause_Prediction/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

cnx = sqlite3.connect('../Wildfire_Cause_Prediction/FPA_FOD_20170508.sqlite')

df = pd.read_sql_query("SELECT FIRE_YEAR,STAT_CAUSE_DESCR,LATITUDE,LONGITUDE,STATE,DISCOVERY_DATE,FIRE_SIZE FROM 'Fires'", cnx)

df['DATE'] = pd.to_datetime(df['DISCOVERY_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D')

df['MONTH'] = pd.DatetimeIndex(df['DATE']).month
df['DAY_OF_WEEK'] = df['DATE'].dt.day_name()
df_orig = df.copy() #I will use this copy later

le = preprocessing.LabelEncoder()
df['STAT_CAUSE_DESCR'] = le.fit_transform(df['STAT_CAUSE_DESCR'])
df['STATE'] = le.fit_transform(df['STATE'])
df['DAY_OF_WEEK'] = le.fit_transform(df['DAY_OF_WEEK'])

df = df.drop('DATE',axis=1)
df = df.dropna()

X = df.drop(['STAT_CAUSE_DESCR'], axis=1).values
y = df['STAT_CAUSE_DESCR'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0) #30% for testing, 70% for training

def set_label(cat):
    cause = 0
    natural = ['Lightning']
    accidental = ['Structure','Fireworks','Powerline','Railroad','Smoking','Children','Campfire','Equipment Use','Debris Burning']
    malicious = ['Arson']
    other = ['Missing/Undefined','Miscellaneous']
    if cat in natural:
        cause = 1
    elif cat in accidental:
        cause = 2
    elif cat in malicious:
        cause = 3
    else:
        cause = 4
    return cause


df['LABEL'] = df_orig['STAT_CAUSE_DESCR'].apply(lambda x: set_label(x)) # I created a copy of the original df earlier in the kernel
df = df.drop('STAT_CAUSE_DESCR',axis=1)

n_estimators = [10, 100, 400, 700, 1000]
criterion = ["gini", "entropy", "log_loss"]
max_features = ["sqrt", "log2"]


X = df.drop(['LABEL'], axis=1).values
y = df['LABEL'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)


clf_rf = ske.RandomForestClassifier(n_estimators=100)
clf_rf = clf_rf.fit(X_train, y_train)
print(clf_rf.score(X_test,y_test))

dump(clf_rf, 'model.joblib') 