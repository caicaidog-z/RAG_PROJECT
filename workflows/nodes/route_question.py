from functools import lru_cache
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from models.llm_factory import get_llm
from workflows.nodes.json_output import parse_model_from_json_text


class RouteQuery(BaseModel):
    """将用户查询路由到最相关的数据源"""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="根据用户问题选择将其路由到向量知识库或网络搜索",
    )


_SYSTEM_PROMPT = """你是一个擅长将用户问题路由到向量知识库或网络搜索的专家。
向量知识库包含与半导体材料，芯片制造，光刻技术相关的文档。
对于这些主题的问题请使用向量知识库，其他情况使用网络搜索。
你只能输出 JSON，格式必须是:
{{"datasource":"vectorstore"}} 或 {{"datasource":"web_search"}}"""


@lru_cache(maxsize=1)
def get_question_router_chain():
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    return route_prompt | get_llm()


def route_query(question: str) -> RouteQuery:
    response = get_question_router_chain().invoke({"question": question})
    return parse_model_from_json_text(response.content, RouteQuery)
