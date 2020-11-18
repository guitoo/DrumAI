from dotenv import load_dotenv
load_dotenv()

import os
SAMPLE_DIR = os.getenv("SAMPLE_DIR")


def get_full_path(path):
    full_path = os.path.join(SAMPLE_DIR, path)
    print(full_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError
    return full_path