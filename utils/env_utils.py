import os

from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
MODEL = os.getenv('MODEL')
BASE_URL = os.getenv('BASE_URL')
MILVUS_URL = os.getenv('MILVUS_URL')

CONFLUENCE_BASE_URL = os.getenv('CONFLUENCE_BASE_URL', 'https://wiki2.rd.chanjet.com')
CONFLUENCE_TOKEN = os.getenv('CONFLUENCE_TOKEN')

# Tavily Web Search API（web_search_node 路由必填）
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

COLLECTION_NAME = 't_collection01'
