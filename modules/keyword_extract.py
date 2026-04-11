def extract_keywords(text: str):
    keywords = [
        'python', 'java', 'sql', 'react',
        'machine learning', 'data', 'api',
        'developer', 'engineer'
    ]

    found = []
    text_lower = text.lower()

    for word in keywords:
        if word in text_lower:
            found.append(word)

    return list(set(found))