"""问答路由：流式 SSE + 同步。"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest, ChatResponse, SourceItem
from api.services import qa_service

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """流式问答（SSE）。

    返回 `text/event-stream`，事件载荷见 qa_service.stream_chat。
    前端用 fetch + ReadableStream 读取，逐 token 渲染打字机效果。
    """
    return StreamingResponse(
        qa_service.stream_chat(req.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭 nginx 缓冲，保证实时推送
        },
    )


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """同步问答：走完整 LangGraph（含幻觉打分/重试），返回完整答案+来源。

    比 /stream 慢（多次 LLM 调用），但答案经过质量校验。
    """
    result = qa_service.chat_sync(req.question)
    return ChatResponse(
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
    )
