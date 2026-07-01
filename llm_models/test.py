#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：RAG_PROJECT
@File    ：test
@IDE     ：PyCharm
@Author  ：zhaozhihua
@Date    ：2026/6/30
"""
from langchain_openai import ChatOpenAI
from utils.env_utils import OPENAI_API_KEY, BASE_URL, MODEL

# 打印配置确认
print(f"MODEL: {MODEL}")
print(f"BASE_URL: {BASE_URL}")
print(f"API_KEY: {OPENAI_API_KEY[:8]}..." if OPENAI_API_KEY else "API_KEY: None!")

llm = ChatOpenAI(
    temperature=0,
    model=MODEL,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)


def test_llm_invoke():
    """测试 LLM 基本调用"""
    response = llm.invoke("你好，请用一句话介绍半导体封装。")
    print(f"LLM 响应: {response.content}")


def test_llm_stream():
    """测试 LLM 流式输出"""
    print("流式输出: ", end="")
    for chunk in llm.stream("什么是光刻胶？"):
        print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    test_llm_invoke()
    # test_llm_stream()