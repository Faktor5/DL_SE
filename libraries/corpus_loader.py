import libraries.wikip as wikip
import pandas as pd

def load_corpus(env):
    article_files = pd.read_csv(env["Article_File"])
    urls = article_files["url"]
    return [wikip.get_wikipedia_text(url) for url in urls]