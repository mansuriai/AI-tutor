from sentence_transformers import SentenceTransformer
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

models = [
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "all-MiniLM-L6-v2"
]

for model_name in models:
    try:
        print(f"Downloading {model_name}...")
        SentenceTransformer(model_name, cache_folder="./models")
        print(f"Successfully cached {model_name}")
    except Exception as e:
        print(f"Failed to download {model_name}: {str(e)}")