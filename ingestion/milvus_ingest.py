import multiprocessing
import os
from multiprocessing import Queue

from config.settings import AppSettings, get_settings
from ingestion.markdown_parser import MarkdownParser
from ingestion.pdf_parser import PDFParser
from retrieval.milvus_store import MilvusStore
from utils.logging import log


SUPPORTED_DOCUMENT_EXTENSIONS = {".md", ".pdf"}


# 采用分布式，多进程的方式把海量数据写入Milvus数据库

def file_parser_process(dir_path: str, output_queue: Queue, batch_size: int = 20):
    """进程1：解析目录下所有支持的文档并分批放入队列"""
    log.info(f"解析进程开始扫描目录: {dir_path}")

    document_files = sorted(
        [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if os.path.splitext(f)[1].lower() in SUPPORTED_DOCUMENT_EXTENSIONS
        ]
    )

    if not document_files:
        log.warning("警告：未找到任何支持的文档文件（.md/.pdf）")
        output_queue.put(None)  # 发送终止信号
        return

    markdown_parser = MarkdownParser()
    pdf_parser = PDFParser()
    doc_batch = []
    for file_path in document_files:
        try:
            extension = os.path.splitext(file_path)[1].lower()
            if extension == ".md":
                docs = markdown_parser.parse_markdown_to_documents(file_path)
            else:
                docs = pdf_parser.parse_pdf_to_documents(file_path)
            if docs:
                doc_batch.extend(docs)

            # 达到批次大小时发送 到队列中
            if len(doc_batch) >= batch_size:
                output_queue.put(doc_batch.copy())
                doc_batch.clear()  # 清空当前缓冲区的所有批次数据
        except Exception as e:
            log.error(f"解析失败 {file_path}: {str(e)}")
            log.exception(e)

    # 发送剩余文档
    if doc_batch:
        output_queue.put(doc_batch)

    # 发送终止信号
    output_queue.put(None)
    log.info(f"解析完成，共处理{len(document_files)}个文件")


def milvus_writer_process(input_queue: Queue):
    """进程2：从队列读取并写入Milvus"""
    log.info("Milvus写入进程启动...")
    mv = MilvusStore()
    mv.create_connection()
    total_count = 0
    while True:
        try:
            datas = input_queue.get()  # 阻塞的函数
            if datas is None:  # 收到了终止的信号
                break

            if isinstance(datas, list):
                mv.add_documents(datas)
                total_count += len(datas)
                log.info(f"累计已写入: {total_count} 个文档")
        except Exception as e:
            log.error(f"写入数据是吧 ！")
            log.exception(e)

    mv.ensure_connection().col.flush()
    log.info(f"写入进程结束，总计写入 {total_count} 个文档")


def ingest_directory(
    document_dir: str,
    queue_maxsize: int = 20,
    settings: AppSettings | None = None,
):
    """解析指定目录下的 Markdown/PDF 文件并写入 Milvus。"""
    if not os.path.isdir(document_dir):
        raise FileNotFoundError(f"文档目录不存在: {document_dir}")

    settings = settings or get_settings()
    mv = MilvusStore(settings=settings)
    mv.create_collection()

    # 创建进程间通信队列
    docs_queue = Queue(maxsize=queue_maxsize)

    # 启动子进程
    parser_proc = multiprocessing.Process(
        target=file_parser_process,
        args=(document_dir, docs_queue),
    )
    writer_proc = multiprocessing.Process(
        target=milvus_writer_process,
        args=(docs_queue,),
    )

    parser_proc.start()
    writer_proc.start()

    # 等待进程结束
    parser_proc.join()
    writer_proc.join()

    print("系统提示：所有任务完成")
