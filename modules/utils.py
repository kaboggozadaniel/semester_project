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