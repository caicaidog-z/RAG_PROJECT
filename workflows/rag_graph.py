from langgraph.constants import START, END
from langgraph.graph import StateGraph

from utils.logging import log
from workflows.nodes.generate_answer import generate
from workflows.nodes.grade_answer import answer_grader_chain
from workflows.nodes.grade_documents import grade_documents
from workflows.nodes.grade_hallucination import hallucination_grader_chain
from workflows.nodes.retrieve import retrieve
from workflows.nodes.rewrite_query import transform_query
from workflows.nodes.route_question import question_router_chain
from workflows.nodes.web_search import web_search
from workflows.state import GraphState


def grade_generation_v_documents_and_question(state):
    """
    评估生成结果是否基于文档并正确回答问题
    Args:
        state (dict): 当前图状态，包含问题、文档和生成结果
    Returns:
        str: 下一节点的名称（useful/not useful/not supported）
    """
    log.info("---检查生成内容是否存在幻觉---")  # 阶段标识
    question = state["question"]  # 获取用户问题
    documents = state["documents"]  # 获取参考文档
    generation = state["generation"]  # 获取生成结果

    # 检查生成是否基于文档
    score = hallucination_grader_chain.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    if grade == "yes":  # 如果生成基于文档
        log.info("---判定：生成内容基于参考文档---")
        # 检查是否准确回答问题
        log.info("---评估：生成回答与问题的匹配度---")
        score = answer_grader_chain.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":  # 如果正确回答问题
            log.info("---判定：生成内容准确回答问题---")
            return "useful"  # 返回有用结果
        else:  # 如果没有回答问题
            log.info("---判定：生成内容未能准确回答问题---")
            return "not useful"  # 返回无用结果
    else:  # 如果生成不基于文档
        log.info("---判定：生成内容未基于参考文档，将重新尝试---")
        return "not supported"  # 返回不支持结果


def decide_to_generate(state):
    """
    决定是生成回答还是重新优化问题

    Args:
        state (dict): 当前图状态，包含问题和过滤后的文档

    Returns:
        str: 下一节点的名称（transform_query或generate）
    """
    log.info("---ASSESS GRADED DOCUMENTS---")  # 阶段标识

    filtered_documents = state["documents"]  # 获取已过滤文档
    transform_count = state.get("transform_count", 0)

    if not filtered_documents:  # 如果没有相关文档
        if transform_count >= 2:
            log.info("---决策：所有文档都与问题无关,并且已经循环了2次，转为web查询问题---")
            return "web_search"  # 返回问题优化节点
        log.info("---决策：所有文档都与问题无关，将转换查询问题---")
        return "transform_query"  # 返回问题优化节点
    else:  # 如果有相关文档
        log.info("---决策：生成最终回答---")
        return "generate"  # 返回回答生成节点


def route_question(state):
    """
    路由问题到网络搜索或RAG流程
    Args:
        state (dict): 当前图状态，包含用户问题

    Returns:
        str: 下一节点的名称（web_search或vectorstore）
    """
    log.info("---ROUTE QUESTION---")  # 阶段标识
    question = state["question"]  # 获取用户问题
    source = question_router_chain.invoke({"question": question})  # 调用问题路由器

    # 根据路由结果决定下一个节点
    if source.datasource == "web_search":
        log.info("---路由到web搜索---")
        return "web_search"
    elif source.datasource == "vectorstore":
        log.info("---路由到RAG系统---")
        return "vectorstore"


# 初始化工作流图
workflow = StateGraph(GraphState)

# 定义各状态节点
workflow.add_node("web_search", web_search)  # 网络搜索节点
workflow.add_node("retrieve", retrieve)  # 文档检索节点
workflow.add_node("grade_documents", grade_documents)  # 文档相关性评分节点
workflow.add_node("generate", generate)  # 回答生成节点
workflow.add_node("transform_query", transform_query)  # 查询优化节点

# 起始路由判断
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve"
    }
)

# 添加固定边
workflow.add_edge("web_search", "generate")  # 网络搜索后直接生成回答
workflow.add_edge("retrieve", "grade_documents")  # 检索后评估文档相关性

# 文档评估后的条件分支
workflow.add_conditional_edges(
    'grade_documents',
    decide_to_generate
)

# 生成结果评估后的条件分支
workflow.add_conditional_edges(
    "generate",  # 生成节点
    grade_generation_v_documents_and_question,  # 生成质量评估函数
    {
        "not supported": "generate",  # 生成不符合要求时重试
        "useful": END,  # 生成符合要求时结束
        "not useful": "transform_query",  # 生成无用结果时优化查询
    },
)

workflow.add_edge("transform_query", "retrieve")  # 查询优化后重新检索

def build_graph():
    return workflow.compile()


graph = build_graph()
