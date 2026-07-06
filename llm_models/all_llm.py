from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

from utils.env_utils import OPENAI_API_KEY, DEEPSEEK_API_KEY,BASE_URL,MODEL,TAVILY_API_KEY

llm = ChatOpenAI(  # openai的
    temperature=0,
    model=MODEL,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL)


web_search_tool = TavilySearchResults(max_results=2, api_key=TAVILY_API_KEY)

# llm = ChatOpenAI(
#     temperature=0.5,
#     model='deepseek-chat',
#     api_key=DEEPSEEK_API_KEY,
#     base_url="https://api.deepseek.com")