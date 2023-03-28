import libraries.wikip as wikip
import pandas as pd
import sqlite3 as sql

def load_corpus(env):
    if env["useLocal"]:
        conn = sql.connect(env["localPath"])
        df = pd.read_sql_query("SELECT * FROM " + env["localDB"], conn)
        return df["article"], df["name"], df["url"]
    else:
        article_files = pd.read_csv(env["Article_File"]).head(env["max_articles"])
        article_files["article"] = [str(wikip.get_wikipedia_text(url)) for url in article_files["url"]]
        if env["saveLocal"]:
            conn = sql.connect(env["localPath"])
            article_files.to_sql(env["localDB"], conn, if_exists="replace", index=False)
        return article_files["article"], article_files["name"], article_files["url"]