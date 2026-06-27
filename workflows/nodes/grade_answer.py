from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from models.llm_factory import get_llm
from workflows.nodes.json_output import parse_model_from_json_text


class GradeAnswer(BaseModel):
    """评估回答是否解决用户问题的二元评分模型"""

    binary_score: str = Field(description="回答是否解决了问题，取值为'yes'或'no'")


_SYSTEM_PROMPT = """您是一个评估回答是否解决用户问题的评分器。\n
给出'yes'或'no'的二元评分。'yes'表示:回答确实解决了该问题。
你只能输出 JSON，格式必须是: {{"binary_score":"yes"}} 或 {{"binary_score":"no"}}"""


@lru_cache(maxsize=1)
def get_answer_grader_chain():
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", "用户问题: \n\n {question} \n\n 生成回答: {generation}"),
        ]
    )
    return answer_prompt | get_llm()


def grade_answer_quality(question: str, generation: str) -> GradeAnswer:
    response = get_answer_grader_chain().invoke({"question": question, "generation": generation})
    return parse_model_from_json_text(response.content, GradeAnswer)
