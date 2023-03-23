import os
from dotenv import load_dotenv

load_dotenv()

env = {
    "Language"  : os.getenv("Language"),
    "Version"   : os.getenv("version")
}

print(env)