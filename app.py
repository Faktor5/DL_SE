import os
import re
import pandas as pd
import libraries.wikip as wikip
import libraries.corpus_loader as cl
import libraries.data_cleaner as dc
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv


load_dotenv()

env = {
    "Language"  : os.getenv("Language"),
    "Version"   : os.getenv("version"),
    "Article_File"  : os.getenv("Article_File"),
    "max_df"    : os.getenv("max_df")
}

articles = None
articleWordVectors = None

model = TfidfVectorizer(max_df = float(env["max_df"]))

def prepare_corpus():
    global articles
    loaded_articles = cl.load_corpus(env)
    list_articles = loaded_articles[0]
    list_articles = list(map(lambda x: dc.clean_text(x), list_articles))
    titles_articles = loaded_articles[1]
    articles = pd.DataFrame({
        "title": titles_articles,
        "article": list_articles
    })

def train_model():
    global articles
    global model
    global articleWordVectors
    model.fit(articles["article"].tolist())
    transformed = model.transform(articles["article"].tolist())
    articleWordVectors = pd.DataFrame(transformed.todense(), columns=model.get_feature_names_out())

def main():
    prepare_corpus()
    train_model()
    print(list(filter(lambda x: x.startswith("k"), articleWordVectors.columns)))
    name = articles["title"].iloc[1]
    vector = articleWordVectors.iloc[1]
    print(name)
    print(vector)

if __name__ == "__main__":
    main()
