import re
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "Resume.csv"

# Load and clean data
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['Resume_str', 'Category'])


def clean_data(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\+\#\&\-/\. ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def trim_resume(text: str, max_length: int = 10000) -> str:
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[:5000] + ' ' + text[-5000:]


df['clean_resume'] = df['Resume_str'].apply(clean_data)
df['clean_resume'] = df['clean_resume'].apply(trim_resume)

df = df[['clean_resume', 'Category']]
X = df['clean_resume']
y = df['Category']

pipeline = Pipeline([
    (
        'tfidf',
        TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=12000,
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            token_pattern=r'(?u)\b[\w\+\#\&\-/\.]+\b',
        ),
    ),
    (
        'clf',
        LogisticRegression(
            solver='lbfgs',
            max_iter=2000,
            class_weight='balanced',
            C=2.0,
            random_state=42,
        ),
    ),
])

print('Dataset shape:', df.shape)
print('Number of classes:', y.nunique())
print('Class counts:')
print(y.value_counts().sort_values(ascending=False).head(20))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=1)
print(f'Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f'Test set accuracy: {accuracy:.4f}')
print('\nClassification report:')
print(classification_report(y_test, preds, zero_division=0))

vectorizer = pipeline.named_steps['tfidf']
model = pipeline.named_steps['clf']

with open(PROJECT_ROOT / 'vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open(PROJECT_ROOT / 'model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('\nSaved model.pkl and vectorizer.pkl to', PROJECT_ROOT)
