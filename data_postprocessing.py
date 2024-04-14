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
"""