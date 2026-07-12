from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_models.all_llm import llm


# 查询的动态路由： 根据用户的提问，决策采用哪种检索策略（网络检索，RAG）


# 数据模型
class RouteQuery(BaseModel):
    """将用户查询路由到最相关的数据源"""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="根据用户问题选择将其路由到向量知识库或网络搜索",
    )


# 带函数调用的LLM
# method="function_calling": 走标准 tools 接口（DeepSeek 等不兼容 OpenAI json_schema beta 的网关用这个）
structured_llm_router = llm.with_structured_output(RouteQuery, method="function_calling")

# 提示词模板
system = """你是一个擅长将用户问题路由到向量知识库或网络搜索的专家。
向量知识库包含与畅捷通新生产部门相关的文档，涵盖生产制造业务、车间管理、BOM（物料清单）、工艺路线、物料需求计划（MRP）、生产订单与排产、库存与仓储、采购与供应商、质量管理、以及畅捷通生产制造软件（如 T+ 生产管理模块、车间管理、条码/扫码报工等）的使用与配置等内容。
对于这些主题的问题请使用向量知识库，其他情况使用网络搜索。"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 系统提示词
        ("human", "{question}"),  # 用户问题占位符
    ]
)

# 创建问题路由器链
question_router_chain = route_prompt | structured_llm_router


# 测试路由器
# print(  # 测试业务问题（应路由到向量知识库）
#     question_router_chain.invoke(
#         {"question": "T+里怎么配置BOM和工艺路线?"}
#     )
# )
print(  # 测试非业务问题（应路由到网络搜索）
    question_router_chain.invoke({"question": "今天，长沙的天气怎么样?"})
)