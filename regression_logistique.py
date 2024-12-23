# Régression Logistique
# c'est le cas d'une entreprise qui détient un data set qui contient des informations sur des clients 
# qui ont acheté ou non des voitures après avoir reçu une publicité sur les réseaux sociaux. 
# nous devons construire un modèle qui nous aide à prédire les acheteurs potentiels pour leur envoyer 
# des publicités ciblées. 
# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importer le dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
dataset.columns
dataset["Purchased"].unique()

table = pd.crosstab(dataset['Purchased'],dataset['Gender'])
table

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


# Gérer les données manquantes
dataset.info()

# Gérer les variables catégoriques


# Diviser le dataset entre le Training set et le Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Construction du modèle
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Faire de nouvelles prédictions
y_pred = classifier.predict(X_test)

# Matrice de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#évaluation du modèle
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,roc_curve, f1_score,roc_auc_score
accuracy_score(y_test,y_pred)

precision_score(y_test,y_pred)

recall_score(y_test,y_pred)

f1_score(y_test,y_pred)

fpr,tpr,thresholds =  roc_curve(y_test,y_pred)
plt.plot(fpr,tpr,"b")
plt.plot([0,1],[0,1],"r-")

roc_auc_score(y_test,y_pred)

#la validation croisée
from sklearn.model_selection import cross_val_score
modelCV = LogisticRegression()

cross_val_score(modelCV,X_train,y_train,cv=5)


# Visualiser les résultats
from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X_train[:,0].max()
X_train[:,0].min()
X_train[:,1].max()
X_train[:,1].min()


X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title(u'Résultats du Training set')
plt.xlabel(u'Age')
plt.ylabel(u'Salaire Estimé')
plt.legend()
plt.show()

