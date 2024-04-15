import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load

from sklearn import tree, preprocessing
import sklearn.ensemble as ske
import sklearn.neural_network as skn
from sklearn.model_selection import train_test_split

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

clf_rf = load('model.joblib')

#pd.DataFrame(X_test).to_csv('X_test.csv')

pred = clf_rf.predict_proba(X_test)
X_test_df = pd.DataFrame(X_test)
predictions = pd.DataFrame(pred, columns = ["natural", "accidental", "malicious", "other"])
print(X_test_df)
long_list = X_test_df[1].tolist()
lat_list = X_test_df[2].tolist()
predictions2 = predictions.assign(longitude = long_list)
predictions3 = predictions2.assign(latitude = lat_list)

print(predictions3)
predictions3.to_csv('X_test_predictions.csv', index = False)
"""for x in n_estimators:
    for y in criterion:
        for z in max_features:
            clf_rf = ske.RandomForestClassifier(n_estimators=x, criterion = y, max_features = z)
            clf_rf = clf_rf.fit(X_train, y_train)
            print(clf_rf.score(X_test,y_test))
            print("n_estimators = " + str(x) + " | criterion = " + str(y) + " | max_features = " + str(z) + " |")
"""
from sklearn.metrics import confusion_matrix
y_pred = clf_rf.fit(X_train, y_train).predict(X_test)
cm = confusion_matrix(y_true=y_test,y_pred=y_pred)
print(cm)

cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig,ax = plt.subplots(figsize=(10,10))
ax.matshow(cmn,cmap=plt.cm.Oranges,alpha=0.7)
for i in range(cmn.shape[0]):
    for j in range(cmn.shape[1]):
        ax.text(x=j,y=i,s=cmn[i,j],va='center',ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()