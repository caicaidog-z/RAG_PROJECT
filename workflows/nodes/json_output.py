import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


def parse_model_from_json_text(text: str, model_cls: type[T]) -> T:
    """Parse a Pydantic model from plain JSON or fenced JSON text."""
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"无法解析模型输出为 JSON: {text}")
        payload = json.loads(raw[start : end + 1])

    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"模型输出 JSON 结构不符合预期: {payload}") from exc
