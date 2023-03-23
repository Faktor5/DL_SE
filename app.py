import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

env = {
    "Language"  : os.getenv("Language"),
    "Version"   : os.getenv("version"),
    "Article_File"  : os.getenv("Article_File"),
}

article_files = pd.read_csv(env["Article_File"])