import numpy as py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

df = pd.read_csv('diabetes.csv')

df.info()

pd.plotting.scatter_matrix(df,figsize=(15,15))


plt.figure(figsize=(15, 10))
sn.heatmap(df.corr(), annot=True)

X = df.drop('Outcome',axis=1).values
y = df['Outcome'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train,y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,roc_curve, f1_score,roc_auc_score

y_pred = knn.predict(X_test)

mx=confusion_matrix(y_test,y_pred)

accuracy_score(y_test,y_pred)

precision_score(y_test,y_pred)

recall_score(y_test,y_pred)

f1_score(y_test,y_pred)

acc = []
from sklearn import metrics
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, y_pred))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,20),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')














