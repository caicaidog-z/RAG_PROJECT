import os

from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://xiaoai.plus/v1')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')

MILVUS_URI = os.getenv('MILVUS_URI', 'http://1.95.116.112:19530')

COLLECTION_NAME = os.getenv('COLLECTION_NAME', 't_collection01')
