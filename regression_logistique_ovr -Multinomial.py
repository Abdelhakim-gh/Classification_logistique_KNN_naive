# Importer les bibliothèques nécessaires
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score 

# Charger le dataset Iris
iris = load_iris()
X = iris.data  
y = iris.target  

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer un modèle de régression logistique multinomiale
model_multinomial = LogisticRegression(multi_class='multinomial')  

# Entraîner le modèle sur les données d'entraînement
model_multinomial.fit(X_train, y_train)

# Prédire les classes sur l'ensemble de test
y_pred = model_multinomial.predict(X_test)

# Calculer et afficher la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Précision du modèle multinomial : {accuracy:.2f}')
