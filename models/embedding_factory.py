from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import OPENAI_API_KEY, OPENAI_BASE_URL

openai_embedding = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL
)

model_name = "BAAI/bge-small-zh-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
bge_embedding = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
