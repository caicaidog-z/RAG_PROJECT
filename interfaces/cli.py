import os
import sys
from pprint import pprint

def run_chat():
    from services.qa_service import stream_question

    while True:
        question = input("用户：")
        if question.lower() in ["q", "exit", "quit"]:
            print("对话结束，拜拜！")
            break

        final_answer = ""
        for output in stream_question(question):
            for key, value in output.items():
                pprint(f"Node '{key}':")
                pprint("\n---\n")
                final_answer = value.get("generation", final_answer)
        pprint(final_answer)


def run_ingest(dir_path: str):
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Markdown 目录不存在: {dir_path}")
    from ingestion.milvus_ingest import ingest_directory

    ingest_directory(dir_path)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("用法: python main.py chat | python main.py ingest <markdown_dir>")
        return 1

    command = argv[0]
    try:
        if command == "chat":
            run_chat()
            return 0

        if command == "ingest":
            if len(argv) < 2:
                print("用法: python main.py ingest <markdown_dir>")
                return 1
            run_ingest(argv[1])
            return 0
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    print(f"未知命令: {command}")
    print("用法: python main.py chat | python main.py ingest <markdown_dir>")
    return 1
