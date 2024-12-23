# Importation des bibliothèques nécessaires
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Charger le dataset Iris
iris = load_iris()
X = iris.data  
y = iris.target  
target_names = iris.target_names  

# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création et entraînement du modèle Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=target_names))

# Visualisation de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap="YlGnBu")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.title("Matrice de confusion - Naive Bayes")
plt.show()

# Visualisation des données sur deux caractéristiques
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = iris.target
plt.figure(figsize=(8, 6))
for species, color in zip(range(3), ['red', 'blue', 'green']):
    subset = df[df['species'] == species]
    plt.scatter(subset[iris.feature_names[0]], subset[iris.feature_names[1]], label=target_names[species], color=color)

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Graphique de dispersion des classes Iris")
plt.legend()
plt.show()
