from langchain_core.tools import create_retriever_tool
from documents.milvus_db import MilvusVectorSave

mv = MilvusVectorSave()
mv.create_connection()
retriever = mv.vector_store_saved.as_retriever(
    search_type='similarity',  # 仅返回相似度超过阈值的文档
    search_kwargs={
        "k": 4,
        "score_threshold": 0.1,
        "ranker_type": "rrf",
        "ranker_params": {"k": 100},
        'filter': "category in ['content', 'Table']"
    }
)


retriever_tool = create_retriever_tool(
    retriever,
    'rag_retriever',
    '搜索并返回关于“畅捷通新生产部门”的信息，内容涵盖：生产制造业务、车间管理、BOM（物料清单）、工艺路线、物料需求计划（MRP）、生产订单与排产、库存与仓储、采购与供应商、质量管理，以及畅捷通生产制造软件（如 T+ 生产管理模块、车间管理、条码/扫码报工等）的使用与配置'
)