import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Chargement du dataset Iris
iris = load_iris()
X = iris.data  
y = iris.target  

# Division des données en ensembles d'entraînement et de test
#stratify=y garantit que si votre ensemble de données contient 30% de la classe A, 50% de la classe B et 20% de la classe C, alors ces proportions seront respectées dans les ensembles d’entraînement et de test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalisation des données (le KNN est sensible aux échelles)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Création du modèle KNN (ici, k=3)
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

# Prédictions
y_pred = knn.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :\n", conf_matrix)

# Rapport de classification (precision, recall, F1-score pour chaque classe)
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("Rapport de classification :\n", report)
