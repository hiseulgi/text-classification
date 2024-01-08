import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

import re
from string import punctuation

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stopwords_list = stopwords.words("english")


async def clean_text(text: str) -> str:
    """Clean text from punctuation, stopwords, numbers and lemmatize it."""
    text = text.lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub("\s+", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    # tokens
    tokens = word_tokenize(text)

    data = [i for i in tokens if i not in punctuation]
    data = [i for i in data if i not in stopwords_list]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    final_text = []
    for i in data:
        word = lemmatizer.lemmatize(i)
        final_text.append(word)

    return " ".join(final_text)
