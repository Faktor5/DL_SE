import os
import re
import time
import pandas as pd
import numpy as np
import libraries.wikip as wikip
import libraries.corpus_loader as cl
import libraries.data_cleaner as dc
import libraries.search_engine as se
from libraries.corpus_filter import CorpusFilter
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request
from dotenv import load_dotenv

#region environment variables
load_dotenv()
env = {
    "Language"  : os.getenv("Language"),
    "Version"   : os.getenv("version"),
    "Article_File"  : os.getenv("Article_File"),
    "max_df"    : os.getenv("max_df"),
    "max_articles"  : int(os.getenv("max_articles")),
    "site_name": os.getenv("site_name"),
    "host": os.getenv("host"),
    "port": int(os.getenv("port")),
    "encoding": os.getenv("encoding"),
}
#endregion

#region Global Variables

# [word1, word2, word3, ...] -> List
words = None
# {title, article, url, raw}
# Index 1: {title, article, url, raw}
# Index 2: {title, article, url, raw}
articles = None
# {word1word2, word3}
# Article 1: {word1: 0.1, word2: 0.2, word3: 0.3, ...}
article_word_matrix = None

# the model which uses the tfidf vectorizer to index the words
model = TfidfVectorizer(max_df = float(env["max_df"]))

# the corpus searcher,
# which is used to search the corpus, articles and words
# in a simple and easy way
check = None

# the flask app
app = Flask(env["site_name"])

#endregion

def prepare_corpus():
    global articles
    # use the corpus loader to load the articles
    loaded_articles = cl.load_corpus(env)
    
    # results -> wikipedia text, name in csv, url in csv
    list_articles = loaded_articles[0]
    titles_articles = loaded_articles[1]
    urls_articles = loaded_articles[2]
    
    # clean the articles with the data cleaner
    cl_articles = list(map(lambda x: dc.clean_text(x), list_articles))

    # save the articles in a dataframe articles
    articles = pd.DataFrame({
        "title": titles_articles,
        "article": cl_articles,
        "url": urls_articles,
        "raw": list_articles
    })

def train_model():
    global check
    global model
    global articles,article_word_matrix, words
    
    # train the model with the articles (texts)
    # from the dataframe articles as a list
    model.fit(articles["article"].tolist())
    
    # save the words in a list
    words = model.get_feature_names_out()
    # transform the articles into a list of vectors
    # (the words are the entirety of all used texts)
    transformed = model.transform(articles["article"].tolist())
    
    # save the word vector per article in a dataframe
    article_word_matrix = pd.DataFrame(
        transformed.todense(),
        columns=words)

    # a tool to filter and walk through the created and organised data
    # just for the developer to understand his own data
    check = CorpusFilter(articles, words, article_word_matrix)

def start_server():
    global app
    app.run(
        host=env["host"],
        port=env["port"],
        debug=True)

def main():
    start_time = time.time()
    prepare_corpus()
    train_model()
    training_time = time.time() - start_time
    print("Model is ready to use! (main.py)")
    print("Corpus is ready to be searched! (main.py)")
    print("------------------------------------")
    print("Time to prepare corpus: " + str(round(training_time)) + " seconds")
    print("------------------------------------")
    start_server()

#region flask routes

@app.route('/words', methods=['GET'])
def get_words_with_prefix():
    global check
    prefix = request.args.get('prefix')
    if prefix is None:
        prefix = "a"
    return {
        "words": check.get_words_by_prefix(prefix)
    }

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', title=env["site_name"])

@app.route('/search')
def search():
    query = request.args.get('query')
    print(query)
    results = se.round_percent(se.top_filter(se.search(query, model, article_word_matrix, check)))
    return render_template('search.html', title=env["site_name"], query=query, results=results)

@app.route('/article')
def article():
    name = request.args.get('name')
    text = check.raw_by_name(name)
    url = check.article_url_by_name(name)
    # cleaned = check.article_text_by_name(name)
    # print(cleaned)
    if text is None:
        return render_template('404.html', title=env["site_name"], message="der Artikel wurde nicht gefunden")
    return render_template('article.html', title=env["site_name"], name=name, text=text, url=url)

#endregion

if __name__ == "__main__":
    main()
