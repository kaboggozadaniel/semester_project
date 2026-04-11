import pickle
import numpy as np

# load once
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def predict_resume(text: str):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    probs = model.predict_proba(X)
    confidence = round(np.max(probs) * 100, 2)

    return prediction, confidence
