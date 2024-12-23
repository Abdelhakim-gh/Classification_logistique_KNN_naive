import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Charger les données
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

# Préparer les données
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Séparer les textes et les étiquettes
texts = [" ".join(words) for words, label in documents]
labels = [label for words, label in documents]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorisation des données textuelles
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Entraîner le modèle Naive Bayes
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Prédiction
y_pred = model.predict(X_test_vectors)

# Évaluation
print("Classification Report:\n", classification_report(y_test, y_pred))

# Tester sur un commentaire personnalisé
new_comment = ["This movie was fantastic! and Great acting and amazing storyline."]
new_comment_vector = vectorizer.transform(new_comment)
prediction = model.predict(new_comment_vector)
print("\nLe commentaire est prédit comme :", prediction[0])
