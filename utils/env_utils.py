import os

from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
MODEL = os.getenv('MODEL')
BASE_URL = os.getenv('BASE_URL')
MILVUS_URL = os.getenv('MILVUS_URL')

COLLECTION_NAME = 't_collection01'
