from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from models.llm_factory import get_llm
from workflows.nodes.json_output import parse_model_from_json_text


class GradeDocuments(BaseModel):
    """对检索到的文档进行相关性评分的二元判断"""

    binary_score: str = Field(description="文档是否与问题相关，取值为'yes'或'no'")


_SYSTEM_PROMPT = """你是一个评估检索文档与用户问题相关性的评分器。\n
如果文档包含与用户问题相关的关键词或语义含义，则评为相关。\n
不需要非常严格的测试，目的是过滤掉错误的检索结果。\n
给出'yes'或'no'的二元评分来表示文档是否与问题相关。
你只能输出 JSON，格式必须是: {{"binary_score":"yes"}} 或 {{"binary_score":"no"}}"""


@lru_cache(maxsize=1)
def get_retrieval_grader_chain():
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    return grade_prompt | get_llm()


def grade_document_relevance(question: str, document: str) -> GradeDocuments:
    response = get_retrieval_grader_chain().invoke({"question": question, "document": document})
    return parse_model_from_json_text(response.content, GradeDocuments)
