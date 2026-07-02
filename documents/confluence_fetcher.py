import os
import re
import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from utils.env_utils import CONFLUENCE_BASE_URL, CONFLUENCE_TOKEN
from utils.log_utils import log


API_PATH = "/rest/api/content/{id}?expand=body.storage,version"
MD_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "md")


class ConfluenceFetcher:
    """通过 Confluence REST API 获取文档 HTML，清洗噪声后转换为 Markdown"""

    def __init__(self, token: str = None, base_url: str = None,
                 output_dir: str = MD_OUTPUT_DIR):
        self.base_url = (base_url or CONFLUENCE_BASE_URL).rstrip("/")
        self.output_dir = output_dir
        token = token or CONFLUENCE_TOKEN
        if not token:
            raise ValueError(
                "未配置 Confluence token，请在 .env 中设置 CONFLUENCE_TOKEN"
            )
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_content(self, content_id: str) -> dict:
        """调用 Confluence API 获取文档内容"""
        url = self.base_url + API_PATH.format(id=content_id)
        log.info(f"请求 Confluence API: {url}")
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _safe_filename(self, name: str) -> str:
        name = re.sub(r'[\\/:*?"<>|]', "_", name)
        return name.strip().strip(".") or "confluence_doc"

    def html_to_markdown(self, html: str) -> str:
        """清洗 HTML 噪声并转换为 Markdown

        清洗策略：
        - 去除 Confluence 元数据宏（ac:structured-macro、ac:parameter、ac:rich-text-body 等命名空间标签的裸文本）
        - 去除 style、script、注释
        - 折叠多余空行
        """
        soup = BeautifulSoup(html, "html.parser")

        # 移除噪声标签（含删除线：s/del/strike）
        for tag in soup(["script", "style", "ac:structured-macro", "ac:parameter",
                         "ac:rich-text-body", "ac:plain-text-body",
                         "s", "del", "strike"]):
            tag.decompose()

        # 移除 HTML 注释
        for c in soup.find_all(string=lambda t: isinstance(t, NavigableString) and "<!--" in str(t)):
            c.extract()

        lines = []
        self._walk(soup, lines)

        # 合并连续空行
        md_text = "\n".join(lines)
        md_text = re.sub(r"\n{3,}", "\n\n", md_text).strip() + "\n"
        return md_text

    def _walk(self, node, lines: list):
        """递归遍历 HTML 节点，按块输出 Markdown 行"""
        for child in node.children:
            if isinstance(child, NavigableString):
                text = str(child)
                if not text.strip():
                    continue
                # 纯文本直接作为段落
                lines.append(text.strip())
                continue

            if not isinstance(child, Tag):
                continue

            name = child.name

            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(name[1])
                lines.append("")
                lines.append(f"{'#' * level} {child.get_text(strip=True)}")
                lines.append("")
            elif name == "p":
                text = child.get_text(separator=" ", strip=True)
                if text:
                    lines.append(text)
                    lines.append("")
            elif name in ("ul", "ol"):
                self._list_to_md(child, lines, ordered=(name == "ol"), depth=0)
                lines.append("")
            elif name == "li":
                # 兜底：直接 li（不在 ul/ol 内）
                lines.append(f"- {child.get_text(strip=True)}")
            elif name == "table":
                self._table_to_md(child, lines)
                lines.append("")
            elif name == "br":
                lines.append("")
            elif name == "hr":
                lines.append("---")
                lines.append("")
            elif name in ("pre", "code"):
                code = child.get_text()
                fence = "```"
                lines.append(fence)
                lines.append(code.rstrip("\n"))
                lines.append(fence)
                lines.append("")
            elif name == "blockquote":
                inner = child.get_text(separator=" ", strip=True)
                for ln in inner.split("\n"):
                    if ln.strip():
                        lines.append(f"> {ln.strip()}")
                lines.append("")
            elif name in ("div", "section", "article", "span", "a", "strong", "b", "em", "i"):
                # 容器/行内标签：递归处理
                self._walk(child, lines)
            else:
                # 其他标签降级为文本
                text = child.get_text(separator=" ", strip=True)
                if text:
                    lines.append(text)
                    lines.append("")

    def _list_to_md(self, tag: Tag, lines: list, ordered: bool, depth: int):
        idx = 1
        for li in tag.find_all("li", recursive=False):
            indent = "  " * depth
            marker = f"{idx}." if ordered else "-"
            text = li.get_text(separator=" ", strip=True)
            # 处理嵌套列表
            nested = li.find(["ul", "ol"])
            if nested:
                nested.extract()
                text = li.get_text(separator=" ", strip=True)
            lines.append(f"{indent}{marker} {text}")
            if nested:
                self._list_to_md(nested, lines, ordered=(nested.name == "ol"), depth=depth + 1)
            idx += 1

    def _table_to_md(self, table: Tag, lines: list):
        rows = table.find_all("tr")
        if not rows:
            return

        # 构建二维网格，处理 rowspan/colspan
        grid: list[list[str | None]] = []
        for r, row in enumerate(rows):
            while len(grid) <= r:
                grid.append([])
            col_cursor = 0
            cells = row.find_all(["th", "td"], recursive=False)
            for cell in cells:
                while col_cursor < len(grid[r]) and grid[r][col_cursor] is not None:
                    col_cursor += 1
                text = (cell.get_text(separator=" ", strip=True)
                        .replace("\n", " ").replace("|", "\\|"))
                colspan = int(cell.get("colspan", 1) or 1)
                rowspan = int(cell.get("rowspan", 1) or 1)
                for j in range(colspan):
                    while len(grid[r]) <= col_cursor + j:
                        grid[r].append(None)
                    grid[r][col_cursor + j] = text
                for i in range(1, rowspan):
                    while len(grid) <= r + i:
                        grid.append([])
                    for j in range(colspan):
                        while len(grid[r + i]) <= col_cursor + j:
                            grid[r + i].append(None)
                        grid[r + i][col_cursor + j] = ""
                col_cursor += colspan

        max_cols = max((len(r) for r in grid), default=0)
        for r in grid:
            while len(r) < max_cols:
                r.append("")

        md_rows = []
        for i, r in enumerate(grid):
            cells = [c if c is not None else "" for c in r]
            md_rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                md_rows.append("| " + " | ".join(["---"] * max_cols) + " |")
        lines.extend(md_rows)

    def fetch_and_convert(self, content_id: str, filename: str = None) -> str:
        """获取 Confluence 文档并保存为 Markdown 文件，返回保存路径"""
        data = self.fetch_content(content_id)
        title = data.get("title", content_id)
        body = data.get("body", {}).get("storage", {}).get("value", "")
        version = data.get("version", {}).get("number", "?")

        log.info(f"获取文档成功: title={title}, version={version}, html长度={len(body)}")

        md_text = self.html_to_markdown(body)
        header = f"# {title}\n\n> Confluence ID: {content_id} | Version: {version}\n\n"
        md_text = header + md_text

        fname = self._safe_filename(filename or title) + ".md"
        out_path = os.path.join(self.output_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        log.info(f"Markdown 已保存: {out_path}")
        return out_path


if __name__ == "__main__":
    fetcher = ConfluenceFetcher()
    # 示例：传入 Confluence 文档 ID
    doc_id = input("请输入 Confluence 文档 ID: ").strip()
    saved = fetcher.fetch_and_convert(doc_id)
    print(f"已生成: {saved}")
