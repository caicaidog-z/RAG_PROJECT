from functools import lru_cache

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from config.settings import get_settings
from utils.logging import log
from workflows.nodes.generate_answer import generate
from workflows.nodes.grade_answer import grade_answer_quality
from workflows.nodes.grade_documents import grade_documents
from workflows.nodes.grade_hallucination import grade_hallucination_risk
from workflows.nodes.retrieve import retrieve
from workflows.nodes.rewrite_query import transform_query
from workflows.nodes.route_question import route_query
from workflows.nodes.web_search import web_search
from workflows.state import GraphState


def grade_generation_v_documents_and_question(state):
    """评估生成结果是否基于文档并正确回答问题。"""
    log.info("---检查生成内容是否存在幻觉---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = grade_hallucination_risk(documents, generation)
    grade = score.binary_score

    if grade == "yes":
        log.info("---判定：生成内容基于参考文档---")
        log.info("---评估：生成回答与问题的匹配度---")
        score = grade_answer_quality(question, generation)
        grade = score.binary_score
        if grade == "yes":
            log.info("---判定：生成内容准确回答问题---")
            return "useful"
        log.info("---判定：生成内容未能准确回答问题---")
        return "not useful"

    log.info("---判定：生成内容未基于参考文档，将重新尝试---")
    return "not supported"


def decide_to_generate(state):
    """决定是生成回答还是重新优化问题。"""
    log.info("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    transform_count = state.get("transform_count", 0)
    settings = get_settings()

    if not filtered_documents:
        if transform_count >= settings.max_transform_count:
            log.info("---决策：文档持续不相关，转为 web 查询问题---")
            return "web_search"
        log.info("---决策：所有文档都与问题无关，将转换查询问题---")
        return "transform_query"

    log.info("---决策：生成最终回答---")
    return "generate"


def route_question(state):
    """路由问题到网络搜索或RAG流程。"""
    log.info("---ROUTE QUESTION---")
    question = state["question"]
    source = route_query(question)

    if source.datasource == "web_search":
        log.info("---路由到web搜索---")
        return "web_search"

    log.info("---路由到RAG系统---")
    return "vectorstore"


def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate)
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    return workflow.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()
