import os
import re
import pandas as pd
import libraries.wikip as wikip
import libraries.corpus_loader as cl
import libraries.data_cleaner as dc
from dotenv import load_dotenv


load_dotenv()

env = {
    "Language"  : os.getenv("Language"),
    "Version"   : os.getenv("version"),
    "Article_File"  : os.getenv("Article_File"),
}

corpus = []
model = None

def prepare_corpus():
    global corpus
    corpus = cl.load_corpus(env)
    corpus = list(map(lambda x: dc.clean_text(x), corpus))

def main():
    prepare_corpus()
    print(corpus)

if __name__ == "__main__":
    main()
