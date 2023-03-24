import os
import re
import pandas as pd
import libraries.wikip as wikip
import libraries.corpus_loader as cl
import libraries.data_cleaner as dc
from libraries.corpus_filter import CorpusFilter
from sklearn.feature_extraction.text import TfidfVectorizer
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
}
#endregion

#region Global Variables

# [word1, word2, word3, ...] -> List
words = None
# {title, article, url}
# Index 1: {title, article, url}
# Index 2: {title, article, url}
articles = None
# {word1word2, word3}
# Article 1: {word1: 0.1, word2: 0.2, word3: 0.3, ...}
article_word_matrix = None

# the model which uses the tfidf vectorizer to index the words
model = TfidfVectorizer(max_df = float(env["max_df"]))

# the corpus searcher,
# which is used to search the corpus, articles and words
# in a simple and easy way
ArticleFilter = None

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
    list_articles = list(map(lambda x: dc.clean_text(x), list_articles))
    
    # save the articles in a dataframe articles
    articles = pd.DataFrame({
        "title": titles_articles,
        "article": list_articles,
        "url": urls_articles
    })

def train_model():
    global articles
    global model
    global article_word_matrix, words
    
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

def start_server():
    "bruh"

def main():
    time = 0
    prepare_corpus()
    train_model()
    check = CorpusFilter(articles, words, article_word_matrix)
    time2 = time.time()
    # print(check.get_words_by_prefix("sch"))
    # result_test = check.article_info(1)
    # print(result_test['title'])
    # print(result_test['url'])
    # print( {k: result_test['vector'][k] for k in list(result_test['vector'])[:10]} )

    print("Model is ready to use! (main.py)")
    print("Corpus is ready to be searched! (main.py)")
    print("------------------------------------")
    print("Time to prepare corpus: " + str(time2 - time) + " seconds")
    print("------------------------------------")

if __name__ == "__main__":
    main()
