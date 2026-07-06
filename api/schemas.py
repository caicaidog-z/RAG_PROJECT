"""请求/响应数据模型。"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """问答请求（流式与同步共用）。"""

    question: str = Field(..., min_length=1, description="用户问题")


class SourceItem(BaseModel):
    """检索到的文档来源元信息。"""

    title: str = ""
    source: str = ""
    filename: str = ""
    category: str = ""


class ChatResponse(BaseModel):
    """同步问答接口的响应。"""

    answer: str
    sources: List[SourceItem] = []


class UploadMdResponse(BaseModel):
    """Markdown 文件上传入库结果。"""

    filename: str
    chunks: int
    message: str


class ConfluenceRequest(BaseModel):
    """Confluence 文档上传请求。"""

    content_id: str = Field(..., min_length=1, description="Confluence 文档 ID")
    filename: Optional[str] = Field(
        None, description="自定义保存文件名（不含扩展名），为空则用文档标题"
    )


class ConfluenceUploadResponse(BaseModel):
    """Confluence 文档上传入库结果。"""

    content_id: str
    filename: str
    chunks: int
    message: str
