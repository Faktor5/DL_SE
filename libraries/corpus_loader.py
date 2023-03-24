import libraries.wikip as wikip
import pandas as pd

def load_corpus(env):
    article_files = pd.read_csv(env["Article_File"]).head(env["max_articles"])
    urls = article_files["url"]
    return [str(wikip.get_wikipedia_text(url).encode(env['encoding'])) for url in urls], article_files["name"], article_files["url"]