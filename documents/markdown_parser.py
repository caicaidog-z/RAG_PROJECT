import os
import re
from typing import List

from utils.log_utils import log
from langchain_core.documents import Document


class MarkdownParser:
    """
    Markdown 文件解析与切片：直接按固定大小 + 重合区间切分纯文本，表格整块保留不被切断。
    不走 UnstructuredMarkdownLoader 的结构化元素解析。
    """
    # 固定大小分片参数（按字符数计算，针对中文）
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    # 超长表格触发按行拆分的阈值（UTF-8 字节数，与 Milvus text VARCHAR(6000) 上限对应，留安全余量）
    TABLE_SPLIT_THRESHOLD = 5500
    # Milvus schema 中定义的字段（metadata 中只保留这些字段）
    MILVUS_SCHEMA_FIELDS = {'category', 'source', 'filename', 'filetype', 'title', 'category_depth'}
    # text 字段在 Milvus schema 中为 VARCHAR(6000)，按 UTF-8 字节数计，留安全余量后的截断阈值
    TEXT_MAX_LENGTH = 5500

    @staticmethod
    def _byte_len(s: str) -> int:
        """UTF-8 字节长度。Milvus VARCHAR(6000) 按 UTF-8 字节算上限，中文每字符 3 字节，
        故所有超长判断必须用字节数而非 len() 字符数。"""
        return len(s.encode('utf-8'))

    def parse_markdown_to_documents(self, md_file: str, encoding='utf-8') -> List[Document]:
        """读取 md 全文 → 从头按固定大小流式切分 → 切点落在表格上时把整个表格并到当前片"""
        filename = os.path.basename(md_file)
        filetype = 'text/markdown'

        with open(md_file, 'r', encoding=encoding) as f:
            text = f.read()

        # 先把全文切成"行级"单元，并标记每个表格的起止行（用于切点落在表格内时整体保留）
        lines = text.split('\n')
        table_ranges = self._find_table_ranges(lines)  # list of (start_idx, end_idx) 闭区间
        log.info(f'文件解析: 总行数={len(lines)}, 表格块数={len(table_ranges)}')

        # 流式按固定大小切分（带 overlap），切点落在表格内则推到表格末行之后
        chunks = self._split_by_size_with_table_protection(lines, table_ranges)

        base_meta = {
            'source': md_file,
            'filename': filename,
            'filetype': filetype,
            'category': 'content',
            'category_depth': 0,
        }

        # 组装 Document，统计每片里是否含表格用于 category 标记与 title
        chunk_docs: List[Document] = []
        for line_range in chunks:
            start, end = line_range
            content_lines = lines[start:end + 1]
            page_content = '\n'.join(content_lines)
            if not page_content.strip():
                continue
            contains_table = any(not (e < s or s > e) for s, e in table_ranges if s >= start and e <= end)
            meta = {**base_meta,
                    'category': 'Table' if contains_table else 'content',
                    'title': self._nearest_title(page_content)}
            doc = Document(page_content=page_content, metadata=meta)
            # 含表格且 UTF-8 字节数超阈值：把表格按行拆（表头复用），文本段单独成块
            if contains_table and self._byte_len(page_content) > self.TABLE_SPLIT_THRESHOLD:
                chunk_docs.extend(self._split_mixed_with_table(doc, table_ranges, lines, start))
            else:
                chunk_docs.append(doc)

        # 清理 metadata + 保底截断
        chunk_docs = [self._clean_metadata(d) for d in chunk_docs]
        log.info(f'固定大小切割后的长度: {len(chunk_docs)}')
        return chunk_docs

    def _find_table_ranges(self, lines: List[str]) -> List[tuple]:
        """识别所有 markdown 表格的行范围（闭区间 [start, end]）

        表格定义：连续以 | 开头的行，且第二行为分隔行。
        """
        ranges = []
        i = 0
        n = len(lines)
        while i < n:
            if (i + 1 < n
                    and self._is_table_row(lines[i])
                    and self._is_table_separator(lines[i + 1])):
                start = i
                j = i + 2
                while j < n and self._is_table_row(lines[j]):
                    j += 1
                ranges.append((start, j - 1))
                i = j
            else:
                i += 1
        return ranges

    def _split_by_size_with_table_protection(self, lines: List[str], table_ranges: List[tuple]) -> List[tuple]:
        """从头按 chunk_size 流式切分（带 chunk_overlap），切点落在表格内则推到表格末行之后

        返回 [(start_line_idx, end_line_idx), ...] 闭区间列表。
        切分以"行"为最小单元，按行字符长度累积；overlap 通过下一片起点回退若干行实现
        （回退字符数不超过 chunk_overlap，且不回退进表格内部）。
        """
        n = len(lines)
        if n == 0:
            return []

        def line_len(idx):
            return len(lines[idx]) + 1  # +1 for '\n'

        # 行号 -> 所属表格范围（若该行在某个表格内）
        line_to_table = {}
        for (s, e) in table_ranges:
            for idx in range(s, e + 1):
                line_to_table[idx] = (s, e)

        chunks: List[tuple] = []
        start = 0  # 当前片起始行
        while start < n:
            # 从 start 累积行，直到字符数 >= chunk_size 或到末尾
            end = start
            cur_len = line_len(start)
            while cur_len < self.CHUNK_SIZE and end < n - 1:
                end += 1
                cur_len += line_len(end)

            # 表格保护：若 end 落在某个表格内部，把 end 推到该表格末行之后
            # （整个表格归当前片，不切断）
            if end in line_to_table:
                _, tbl_end = line_to_table[end]
                if tbl_end >= n - 1:
                    # 表格延伸到文件末尾，直接收尾
                    end = n - 1
                else:
                    end = tbl_end  # 表格最后一行（表格整体在当前片）

            chunks.append((start, end))

            if end >= n - 1:
                break

            # 下一片起点：从 end 之后往前回退，实现 overlap
            # 回退条件：回退字符累计不超过 chunk_overlap，且不回退进表格内部
            next_start = end + 1
            overlap_len = 0
            back = end  # 从当前片最后一行开始尝试回退
            while back > start:
                if back in line_to_table:
                    break  # 不回退进表格内部
                ll = line_len(back)
                if overlap_len + ll > self.CHUNK_OVERLAP:
                    break
                overlap_len += ll
                back -= 1
            new_start = back + 1 if overlap_len > 0 else next_start
            # 防止回退导致原地不动（保证至少前进1行，避免死循环）
            if new_start <= start:
                new_start = start + 1
            start = new_start

        return chunks

    @staticmethod
    def _is_table_row(line: str) -> bool:
        """是否为 markdown 表格的数据/表头行（以 | 开头且至少有一个 | 分隔）"""
        stripped = line.strip()
        return stripped.startswith('|') and stripped.endswith('|') and stripped.count('|') >= 2

    @staticmethod
    def _is_table_separator(line: str) -> bool:
        """是否为 markdown 表格的分隔行（只含 |、-、:、空格，且至少含一个 -）"""
        stripped = line.strip()
        if not stripped.startswith('|'):
            return False
        inner = stripped.replace('|', '').replace(' ', '').replace(':', '')
        return len(inner) > 0 and set(inner) <= {'-'}

    @staticmethod
    def _nearest_title(text: str) -> str:
        """从文本片段中取最近的一个 markdown 标题（# 开头）作为 title，取不到则返回空串"""
        for line in reversed(text.strip().split('\n')):
            m = re.match(r'^#{1,6}\s+(.+?)\s*$', line.strip())
            if m:
                return m.group(1)
        return ''

    def _split_oversized_table(self, doc: Document, max_length: int = 5500) -> List[Document]:
        """按行拆分超长 Markdown 表格，每个分片保留表头。max_length 按 UTF-8 字节数计。"""
        content = doc.page_content
        lines = content.split('\n')

        # 识别表头：第一行数据 + 分隔行
        header_lines = []
        data_start = 0
        for i, line in enumerate(lines):
            header_lines.append(line)
            if self._is_table_separator(line):
                data_start = i + 1
                break

        if not header_lines or data_start >= len(lines):
            # 无法正确拆分，按字节截断作为最后手段
            truncated = self._truncate_by_bytes(content, max_length, '(表格内容过长，已截断)')
            return [Document(page_content=truncated, metadata={**doc.metadata})]

        header_str = '\n'.join(header_lines)
        header_len = self._byte_len(header_str) + 1  # +1 for newline

        chunks = []
        current_rows = []
        current_len = header_len

        for row in lines[data_start:]:
            row_len = self._byte_len(row) + 1
            if current_len + row_len > max_length and current_rows:
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

    def _truncate_by_bytes(self, content: str, max_bytes: int, suffix_note: str) -> str:
        """按 UTF-8 字节长度截断，保证不截断半个多字节字符"""
        encoded = content.encode('utf-8')
        if len(encoded) <= max_bytes:
            return content
        note = f'\n...({suffix_note})'
        note_bytes = len(note.encode('utf-8'))
        cut = max_bytes - note_bytes
        if cut <= 0:
            cut = max_bytes
            note = ''
        # 逐步回退避免截断多字节字符中段
        while cut > 0 and (encoded[cut] & 0xC0) == 0x80:
            cut -= 1
        return encoded[:cut].decode('utf-8', errors='ignore') + note

    def _split_mixed_with_table(self, doc: Document, table_ranges: List[tuple],
                                 all_lines: List[str], chunk_start: int) -> List[Document]:
        """拆分"文本+表格+文本"的混合超长片

        策略：把片内的表格段单独抽出（category=Table，超长走 _split_oversized_table），
        表格前后的文本段各自按 chunk_size + overlap 切分（category=content）。
        这样既保证表格完整不被字节上限卡死，又让文本回到正常固定大小切分。
        """
        content = doc.page_content
        base_meta = {k: v for k, v in doc.metadata.items()}

        # 找该片内包含的表格范围（行号是相对全文的）
        chunk_end = chunk_start + content.count('\n')
        in_chunk_tables = [(s, e) for s, e in table_ranges if s >= chunk_start and e <= chunk_end]
        if not in_chunk_tables:
            # 兜底：没识别到表格就直接按字节截断
            return [Document(page_content=self._truncate_by_bytes(content, self.TEXT_MAX_LENGTH, '内容过长，已截断'),
                             metadata={**base_meta})]

        result: List[Document] = []
        # 片内相对行号
        rel_lines = content.split('\n')

        # 把片内内容按"表格段 / 非表格段"切开
        segments = []  # 每段: ('table', [行]) 或 ('text', [行])
        i = 0
        n = len(rel_lines)
        for (ts, te) in in_chunk_tables:
            rs, re_ = ts - chunk_start, te - chunk_start
            # 表格前的文本
            if i < rs:
                segments.append(('text', rel_lines[i:rs]))
            segments.append(('table', rel_lines[rs:re_ + 1]))
            i = re_ + 1
        if i < n:
            segments.append(('text', rel_lines[i:n]))

        for seg_type, seg_lines in segments:
            seg_text = '\n'.join(seg_lines)
            if not seg_text.strip():
                continue
            if seg_type == 'table':
                tdoc = Document(page_content=seg_text, metadata={**base_meta, 'category': 'Table',
                                                                  'title': self._nearest_title(seg_text)})
                if self._byte_len(seg_text) > self.TEXT_MAX_LENGTH:
                    result.extend(self._split_oversized_table(tdoc))
                else:
                    result.append(tdoc)
            else:
                xdoc = Document(page_content=seg_text, metadata={**base_meta, 'category': 'content',
                                                                  'title': self._nearest_title(seg_text)})
                result.extend(self._split_text_by_size(xdoc))
        return result

    def _split_text_by_size(self, doc: Document) -> List[Document]:
        """纯文本按 chunk_size + overlap 切分（递归字符切分器，行内可细切，避免单行超长）

        复用 langchain 的 RecursiveCharacterTextSplitter，split_documents 会把父 metadata
        复制到每个子片；对短于 chunk_size 的块原样返回。
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            return [doc]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=['\n\n', '\n', ' ', ''],
            keep_separator=True,
        )
        sub = splitter.split_documents([doc])
        # 逐个子片做字节保底，防止极端单行中文超 6000 字节
        out = []
        for s in (sub or [doc]):
            if self._byte_len(s.page_content) > self.TEXT_MAX_LENGTH:
                s = Document(page_content=self._truncate_by_bytes(s.page_content, self.TEXT_MAX_LENGTH, '内容过长，已截断'),
                             metadata={**s.metadata})
            out.append(s)
        return out

    def _clean_metadata(self, doc: Document) -> Document:
        """清理 metadata，只保留 Milvus schema 中定义的字段，防止插入报错"""
        cleaned_metadata = {}
        for key, value in doc.metadata.items():
            if key in self.MILVUS_SCHEMA_FIELDS:
                cleaned_metadata[key] = value

        # category_depth 是 INT64 字段，必须有值
        if 'category_depth' not in cleaned_metadata or cleaned_metadata['category_depth'] is None:
            cleaned_metadata['category_depth'] = 0

        # title 在 Milvus schema 中为非空字段，缺失时补空字符串，避免插入报错
        if not cleaned_metadata.get('title'):
            cleaned_metadata['title'] = ''

        # text 在 Milvus schema 中为 VARCHAR(6000)，按 UTF-8 字节数保底截断
        if self._byte_len(doc.page_content) > self.TEXT_MAX_LENGTH:
            log.warning(f"文档超长(字节{self._byte_len(doc.page_content)})，已截断到 {self.TEXT_MAX_LENGTH}字节: {cleaned_metadata.get('filename', '')}")
            doc.page_content = self._truncate_by_bytes(doc.page_content, self.TEXT_MAX_LENGTH, '内容过长，已截断')

        doc.metadata = cleaned_metadata
        return doc


if __name__ == '__main__':
    file_path = '/Users/zhaozhihua/Downloads/RAG企业知识库项目/RAG_PROJECT/md/替代件详细需求-替代方案（新模板）.md'  # 在项目根目录运行
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)
    for item in docs:
        print(f"元数据: {item.metadata}")
        print(f"标题: {item.metadata.get('title', None)}")
        print(f"块长度: {len(item.page_content)}")
        print(f"doc的内容: {item.page_content}\n")
        print("------" * 10)
