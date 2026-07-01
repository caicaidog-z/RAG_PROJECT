from typing import List

from bs4 import BeautifulSoup
from langchain_experimental.text_splitter import SemanticChunker

from llm_models.embeddings_model import openai_embedding
from utils.log_utils import log
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document


class MarkdownParser:
    """
    专门负责markdown文件的解析和切片，支持标题、段落、表格、列表等元素
    """
    # 不允许被 SemanticChunker 切分的类别（保持结构完整）
    UNSPLITTABLE_CATEGORIES = {'Table', 'List'}
    # Milvus schema 中定义的字段（metadata 中只保留这些字段）
    MILVUS_SCHEMA_FIELDS = {'category', 'source', 'filename', 'filetype', 'title', 'category_depth'}

    def __init__(self):
        self.text_splitter = SemanticChunker(
            openai_embedding, breakpoint_threshold_type="percentile"
        )

    def text_chunker(self, datas: List[Document]) -> List[Document]:
        new_docs = []
        for d in datas:
            category = d.metadata.get('category', None)

            # 表格和列表不允许被 SemanticChunker 切分，保持结构完整
            if category in self.UNSPLITTABLE_CATEGORIES:
                if category == 'Table' and len(d.page_content) > 5000:
                    new_docs.extend(self._split_oversized_table(d))
                else:
                    new_docs.append(d)
                continue

            # 普通文本：内容超出了阈值，则按照语义再切割
            if len(d.page_content) > 5000:
                new_docs.extend(self.text_splitter.split_documents([d]))
                continue
            new_docs.append(d)
        return new_docs

    def parse_markdown_to_documents(self, md_file: str, encoding='utf-8') -> List[Document]:
        documents = self.parse_markdown(md_file)
        log.info(f'文件解析后的docs长度: {len(documents)}')

        merged_documents = self.merge_title_content(documents)
        log.info(f'文件合并后的长度: {len(merged_documents)}')

        # 清理 metadata，只保留 Milvus schema 中的字段
        merged_documents = [self._clean_metadata(doc) for doc in merged_documents]

        chunk_documents = self.text_chunker(merged_documents)
        log.info(f'语义切割后的长度: {len(chunk_documents)}')
        return chunk_documents

    def parse_markdown(self, md_file: str) -> List[Document]:
        loader = UnstructuredMarkdownLoader(
            file_path=md_file,
            mode='elements',
            strategy='fast'
        )
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)

        return docs

    def _group_list_items(self, datas: List[Document]) -> List[Document]:
        """将连续的 ListItem 元素合并为一个 Document，保留列表层级结构"""
        grouped = []
        i = 0
        while i < len(datas):
            doc = datas[i]
            category = doc.metadata.get('category', None)

            if category == 'ListItem':
                # 收集连续的、具有相同 parent_id 的 ListItem
                list_items = [doc]
                parent_id = doc.metadata.get('parent_id', None)
                j = i + 1
                while j < len(datas):
                    next_doc = datas[j]
                    next_category = next_doc.metadata.get('category', None)
                    next_parent_id = next_doc.metadata.get('parent_id', None)
                    if next_category == 'ListItem' and next_parent_id == parent_id:
                        list_items.append(next_doc)
                        j += 1
                    else:
                        break

                # 格式化为 Markdown 列表
                lines = []
                for item in list_items:
                    depth = item.metadata.get('category_depth', 0) or 0
                    indent = '  ' * depth
                    lines.append(f"{indent}- {item.page_content}")

                # 创建合并后的 Document
                merged_doc = Document(
                    page_content='\n'.join(lines),
                    metadata={
                        **list_items[0].metadata,
                        'category': 'List',
                    }
                )
                grouped.append(merged_doc)
                i = j
            else:
                grouped.append(doc)
                i += 1

        return grouped

    def merge_title_content(self, datas: List[Document]) -> List[Document]:
        # 先将连续的 ListItem 合并为 List
        datas = self._group_list_items(datas)

        merged_data = []
        parent_dict = {}  # key: element_id, value: Title Document（包含累积的内容）

        for document in datas:
            metadata = document.metadata
            if 'languages' in metadata:
                metadata.pop('languages')

            parent_id = metadata.get('parent_id', None)
            category = metadata.get('category', None)
            element_id = metadata.get('element_id', None)

            # --- Title ---
            if category == 'Title':
                document.metadata['title'] = document.page_content
                if parent_id in parent_dict:
                    document.page_content = (
                        parent_dict[parent_id].page_content + ' -> ' + document.page_content
                    )
                parent_dict[element_id] = document

            # --- NarrativeText ---
            elif category == 'NarrativeText':
                if parent_id and parent_id in parent_dict:
                    # 合并到父标题的内容中
                    parent_dict[parent_id].page_content = (
                        parent_dict[parent_id].page_content + ' ' + document.page_content
                    )
                    parent_dict[parent_id].metadata['category'] = 'content'
                else:
                    # 无父标题的独立段落
                    merged_data.append(document)

            # --- Table ---
            elif category == 'Table':
                # 将 HTML 表格转换为 Markdown 格式
                html_table = metadata.get('text_as_html', None)
                if html_table:
                    document.page_content = self._html_table_to_markdown(html_table)
                # 继承父标题的 title
                if parent_id and parent_id in parent_dict:
                    document.metadata['title'] = parent_dict[parent_id].metadata.get('title', '')
                # 表格作为独立 Document 保留，不合并到父标题
                merged_data.append(document)

            # --- List（已由 _group_list_items 合并） ---
            elif category == 'List':
                # 继承父标题的 title
                if parent_id and parent_id in parent_dict:
                    document.metadata['title'] = parent_dict[parent_id].metadata.get('title', '')
                # 列表作为独立 Document 保留，不合并到父标题
                merged_data.append(document)

            # --- UncategorizedText ---
            elif category == 'UncategorizedText':
                if parent_id and parent_id in parent_dict:
                    parent_dict[parent_id].page_content = (
                        parent_dict[parent_id].page_content + ' ' + document.page_content
                    )
                    parent_dict[parent_id].metadata['category'] = 'content'
                else:
                    merged_data.append(document)

            # --- 其他类别（Image 等） ---
            else:
                if parent_id and parent_id in parent_dict:
                    parent_dict[parent_id].page_content = (
                        parent_dict[parent_id].page_content + ' ' + document.page_content
                    )
                    parent_dict[parent_id].metadata['category'] = 'content'
                else:
                    merged_data.append(document)

        # 将所有父标题 Document（包含累积的内容）加入结果
        merged_data.extend(parent_dict.values())

        return merged_data

    def _html_table_to_markdown(self, html_table: str) -> str:
        """将 HTML 表格字符串转换为 Markdown 表格格式"""
        soup = BeautifulSoup(html_table, 'html.parser')
        table = soup.find('table')
        if not table:
            return html_table  # 降级：返回原始文本

        rows = table.find_all('tr')
        if not rows:
            return html_table

        md_lines = []
        header_processed = False

        for i, row in enumerate(rows):
            cols = row.find_all(['th', 'td'])
            if not cols:
                continue
            row_data = [col.get_text(strip=True).replace('\n', ' ') for col in cols]
            md_lines.append('| ' + ' | '.join(row_data) + ' |')

            # 在第一行之后添加 Markdown 表格分隔行
            if i == 0 and not header_processed:
                md_lines.append('| ' + ' | '.join(['---'] * len(cols)) + ' |')
                header_processed = True

        return '\n'.join(md_lines) if md_lines else html_table

    def _split_oversized_table(self, doc: Document, max_length: int = 5500) -> List[Document]:
        """按行拆分超长 Markdown 表格，每个分片保留表头

        Args:
            doc: 包含 Markdown 表格的 Document
            max_length: 每个分片的最大字符数，默认 5500（留出 Milvus VARCHAR(6000) 的安全余量）
        """
        content = doc.page_content
        lines = content.split('\n')

        # 识别表头：第一行数据 + 分隔行
        header_lines = []
        data_start = 0
        for i, line in enumerate(lines):
            header_lines.append(line)
            # Markdown 表格分隔行：只包含 |、-、: 和空格
            stripped = line.strip()
            if stripped.startswith('|') and set(stripped.replace('|', '').replace(' ', '')) <= {'-', ':'}:
                data_start = i + 1
                break

        if not header_lines or data_start >= len(lines):
            # 无法正确拆分，截断作为最后手段
            truncated = Document(
                page_content=content[:max_length] + '\n...(表格内容过长，已截断)',
                metadata={**doc.metadata}
            )
            return [truncated]

        header_str = '\n'.join(header_lines)
        header_len = len(header_str) + 1  # +1 for newline

        chunks = []
        current_rows = []
        current_len = header_len

        for row in lines[data_start:]:
            row_len = len(row) + 1
            if current_len + row_len > max_length and current_rows:
                # 输出当前分片
                chunk_content = header_str + '\n' + '\n'.join(current_rows)
                chunks.append(Document(
                    page_content=chunk_content,
                    metadata={**doc.metadata}
                ))
                current_rows = []
                current_len = header_len
            current_rows.append(row)
            current_len += row_len

        # 输出剩余行
        if current_rows:
            chunk_content = header_str + '\n' + '\n'.join(current_rows)
            chunks.append(Document(
                page_content=chunk_content,
                metadata={**doc.metadata}
            ))

        return chunks if chunks else [doc]

    def _clean_metadata(self, doc: Document) -> Document:
        """清理 metadata，只保留 Milvus schema 中定义的字段，防止插入报错"""
        cleaned_metadata = {}
        for key, value in doc.metadata.items():
            if key in self.MILVUS_SCHEMA_FIELDS:
                cleaned_metadata[key] = value

        # category_depth 是 INT64 字段，必须有值
        if 'category_depth' not in cleaned_metadata or cleaned_metadata['category_depth'] is None:
            cleaned_metadata['category_depth'] = 0

        doc.metadata = cleaned_metadata
        return doc


if __name__ == '__main__':
    file_path = r'D:\git\RAG_PROJECT\md\替代件详细需求-替代方案（新模板）.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)
    for item in docs:
        print(f"元数据: {item.metadata}")
        print(f"标题: {item.metadata.get('title', None)}")
        print(f"doc的内容: {item.page_content}\n")
        print("------" * 10)
