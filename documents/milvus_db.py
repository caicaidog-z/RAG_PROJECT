from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import IndexType, MilvusClient, Function
from pymilvus.client.types import MetricType, DataType, FunctionType

from documents.markdown_parser import MarkdownParser
from llm_models.embeddings_model import bge_embedding
from utils.env_utils import MILVUS_URL, COLLECTION_NAME
from utils.log_utils import log


class MilvusVectorSave:
    """把新的document数据插入到数据库中"""

    def __init__(self) -> object:
        """自定义collection的索引"""
        self.vector_store_saved: Milvus = None

    def create_collection(self):
        client = MilvusClient(uri=MILVUS_URL)
        schema = client.create_schema()
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=6000, enable_analyzer=True,
                         analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
        schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='category_depth', datatype=DataType.INT64)
        schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=512)

        bm25_function = Function(
            name="text_bm25_emb",  # Function name
            input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse"],
            # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,  # Set to `BM25`
        )
        schema.add_function(bm25_function)
        index_params = client.prepare_index_params()

        index_params.add_index(
            field_name="sparse",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                # Algorithm for building and querying the index. Valid values: DAAT_MAXSCORE, DAAT_WAND, TAAT_NAIVE.
                "bm25_k1": 1.2,
                "bm25_b": 0.75
            },
        )
        index_params.add_index(
            field_name="dense",
            index_name="dense_inverted_index",
            index_type=IndexType.HNSW,  # Inverted index type for sparse vectors
            metric_type=MetricType.IP,
            params={"M": 16, "efConstruction": 64}  # M :邻接节点数, efConstruction: 搜索范围
        )

        if COLLECTION_NAME in client.list_collections():
            # 先释放， 再删除索引，再删除collection
            client.release_collection(collection_name=COLLECTION_NAME)
            client.drop_index(collection_name=COLLECTION_NAME, index_name='sparse_inverted_index')
            client.drop_index(collection_name=COLLECTION_NAME, index_name='dense_inverted_index')
            client.drop_collection(collection_name=COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

    def create_connection(self):
        """创建一个Connection： milvus + langchain。pip install  langchain-milvus"""
        self.vector_store_saved = Milvus(
            embedding_function=bge_embedding,
            collection_name=COLLECTION_NAME,
            builtin_function=BM25BuiltInFunction(),
            vector_field=['dense', 'sparse'],
            consistency_level="Strong",
            auto_id=True,
            connection_args={"uri": MILVUS_URL}
        )

    def add_documents(self, datas: List[Document]):
        """把新的document保存到Milvus中"""
        # 最后一道保险：确保 text 不超过 Milvus VARCHAR(6000) 上限
        max_len = 5500
        for idx, d in enumerate(datas):
            log.info(f"[入库前] 第{idx}条: page_content长度={len(d.page_content)}, category={d.metadata.get('category')}")
            # 调试：打印前几条的完整 metadata 和内容片段
            if idx < 6:
                log.info(f"[入库前] 第{idx}条 metadata={d.metadata}")
                log.info(f"[入库前] 第{idx}条 内容前300字符={d.page_content[:300]!r}")
            # 同时检查 metadata 里是否有别的长文本字段
            for k, v in d.metadata.items():
                if isinstance(v, str) and len(v) > 1000:
                    log.warning(f"[入库前] 第{idx}条 metadata字段 {k} 长度={len(v)}")
            if len(d.page_content) > max_len:
                log.warning(f"[入库前] 第{idx}条 超长({len(d.page_content)})，截断到 {max_len}")
                d.page_content = d.page_content[:max_len] + '\n...(内容过长，已截断)'
        self.vector_store_saved.add_documents(datas)



if __name__ == '__main__':
    # 解析文件内容
    file_path = r'/Users/zhaozhihua/Downloads/RAG企业知识库项目/RAG_PROJECT/md/替代件详细需求-替代方案（新模板）.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    # 写入Milvus数据库
    mv = MilvusVectorSave()
    mv.create_collection()
    mv.create_connection()
    mv.add_documents(docs)

    client = mv.vector_store_saved.client
    # 得到表结构
    desc_collection = client.describe_collection(
        collection_name=COLLECTION_NAME
    )
    print('表结构是: ', desc_collection)

    # 得到当前表的，所有的index
    res = client.list_indexes(
        collection_name=COLLECTION_NAME
    )
    print('表中的所有索引：', res)

    if res:
        for i in res:
            # 得到索引的描述
            desc_index = client.describe_index(
                collection_name=COLLECTION_NAME,
                index_name=i
            )
            print(desc_index)

    result = client.query(
        collection_name=COLLECTION_NAME,
        filter="category == 'Title'",  # 查询 category == 'Title' 的所有数据
        output_fields=['text', 'category', 'filename']  # 指定返回的字段
    )

    print('测试 过滤查询的结果是: ', result)
