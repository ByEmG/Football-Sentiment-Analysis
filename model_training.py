import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load data clearly
df = pd.read_csv("final_cleaned_sentiment_analysis.csv")
df['cleaned_text'] = df['cleaned_text'].fillna('')

# 2. TF-IDF Feature Extraction
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
X = tfidf.fit_transform(df['cleaned_text']).toarray()

# 3. Label Encoding
encoder = LabelEncoder()
y = encoder.fit_transform(df['Sentiment'])

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Logistic Regression
logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train, y_train)

# Evaluate Logistic Regression
y_pred_logreg = logreg.predict(X_test)
print("📌 Logistic Regression Classification Report:\n")
print(classification_report(y_test, y_pred_logreg, target_names=encoder.classes_))

# 6. Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Evaluate Random Forest
y_pred_rf = rf.predict(X_test)
print("\n📌 Random Forest Classification Report:\n")
print(classification_report(y_test, y_pred_rf, target_names=encoder.classes_))