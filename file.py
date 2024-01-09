import pandas as pd


data = pd.read_csv('news.csv')
print(data.head())

print(data.isnull().sum())
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix


data = data.fillna('')


X = data['text']
y = data['label']


encoder = LabelEncoder()
y = encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)


tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
from sklearn.linear_model import PassiveAggressiveClassifier


pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)


y_pred = pac.predict(tfidf_test)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
