# Importer les bibliothèques nécessaires
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score ,classification_report

# Charger le dataset Iris
iris = load_iris()
X = iris.data  
y = iris.target  

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer un modèle de régression logistique avec One-vs-Rest
model_ovr = LogisticRegression(multi_class='ovr')  

model_ovr.fit(X_train, y_train)

y_pred = model_ovr.predict(X_test)

print(classification_report(y_test, y_pred, target_names=iris.target_names))
     
accuracy = accuracy_score(y_test, y_pred)
print(f'Précision du modèle One-vs-Rest : {accuracy:.2f}')

