import re

def clean_data(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def trim_resume(text):
    if len(text) < 1000:
        return text
    return text[:500] + " " + text[-500:]

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\+\#\&\-/\. ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # boosting
    tech_keywords = ['python', 'java', 'sql', 'react', 'api']
    for word in tech_keywords:
        if word in text:
            text += ' ' + (word + ' ') * 3

    return text.strip()