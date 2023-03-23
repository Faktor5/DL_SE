import pandas as pd
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

def clean_text(text):
    # Converting to Lowercase
    text = text.lower()
    
    # lemmitization
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc]
    text = ' '.join(tokens)
    
    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text