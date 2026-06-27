from typing import List

from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus
from pymilvus import Function, IndexType, MilvusClient
from pymilvus.client.types import DataType, FunctionType, MetricType

from config.settings import AppSettings, get_settings
from models.embedding_factory import get_bge_embedding


class MilvusStore:
    """Milvus collection management and document persistence."""

    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or get_settings()
        self.vector_store_saved: Milvus | None = None

    def create_collection(self, recreate: bool = True):
        client = MilvusClient(uri=self.settings.milvus_uri)
        schema = client.create_schema()
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=6000,
            enable_analyzer=True,
            analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]},
        )
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="filetype", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="category_depth", datatype=DataType.INT64)
        schema.add_field(field_name="page_number", datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name="element_type", datatype=DataType.VARCHAR, max_length=255, nullable=True)
        schema.add_field(field_name="asset_path", datatype=DataType.VARCHAR, max_length=2000, nullable=True)
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=512)

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="sparse",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
            },
        )
        index_params.add_index(
            field_name="dense",
            index_name="dense_inverted_index",
            index_type=IndexType.HNSW,
            metric_type=MetricType.IP,
            params={"M": 16, "efConstruction": 64},
        )

        if recreate and self.settings.collection_name in client.list_collections():
            client.release_collection(collection_name=self.settings.collection_name)
            client.drop_index(
                collection_name=self.settings.collection_name,
                index_name="sparse_inverted_index",
            )
            client.drop_index(
                collection_name=self.settings.collection_name,
                index_name="dense_inverted_index",
            )
            client.drop_collection(collection_name=self.settings.collection_name)

        if self.settings.collection_name not in client.list_collections():
            client.create_collection(
                collection_name=self.settings.collection_name,
                schema=schema,
                index_params=index_params,
            )

    def create_connection(self):
        self.vector_store_saved = Milvus(
            embedding_function=get_bge_embedding(),
            collection_name=self.settings.collection_name,
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            consistency_level="Strong",
            auto_id=True,
            connection_args={"uri": self.settings.milvus_uri},
        )

    def ensure_connection(self) -> Milvus:
        if self.vector_store_saved is None:
            self.create_connection()
        return self.vector_store_saved

    def add_documents(self, datas: List[Document]):
        self.ensure_connection().add_documents(datas)


MilvusVectorSave = MilvusStore
