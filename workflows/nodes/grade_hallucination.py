from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from models.llm_factory import get_llm
from workflows.nodes.json_output import parse_model_from_json_text


class GradeHallucinations(BaseModel):
    """对生成回答中是否存在幻觉进行二元评分"""

    binary_score: str = Field(description="回答是否基于事实，取值为'yes'或'no'")


_SYSTEM_PROMPT = """您是一个评估生成内容是否基于检索事实的评分器。\n
给出'yes'或'no'的二元评分。'yes'表示回答是基于/支持于给定事实集的。
你只能输出 JSON，格式必须是: {{"binary_score":"yes"}} 或 {{"binary_score":"no"}}"""


@lru_cache(maxsize=1)
def get_hallucination_grader_chain():
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", "事实集: \n\n {documents} \n\n 生成内容: {generation}"),
        ]
    )
    return hallucination_prompt | get_llm()


def grade_hallucination_risk(documents, generation: str) -> GradeHallucinations:
    response = get_hallucination_grader_chain().invoke(
        {"documents": documents, "generation": generation}
    )
    return parse_model_from_json_text(response.content, GradeHallucinations)
