import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import requests
from FlagEmbedding import FlagReranker
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

DB_PATH = Path("./milvus_demo.db")
DATASET_PATH = Path("testcase.json")
REPORT_PATH = Path("eval_report.json")
MODEL_NAME = "qwen2.5:7b"
COLLECTION_NAME = "rag_kb"


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[\s\u3000]+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", "", text)
    return text.strip()


def is_refusal_answer(text: str) -> bool:
    if not text:
        return False
    refusal_markers = [
        "根据现有知识库，未找到相关内容",
        "未找到相关内容",
        "无法回答",
        "不能回答",
        "我不能",
        "没有相关信息",
    ]
    return any(marker in text for marker in refusal_markers)


def evaluate_answer(answer: str, ground_truth: str) -> float:
    if not answer or not ground_truth:
        return 0.0

    answer_norm = normalize_text(answer)
    gt_norm = normalize_text(ground_truth)

    if answer_norm == gt_norm:
        return 1.0

    if is_refusal_answer(answer) and is_refusal_answer(ground_truth):
        return 1.0

    if gt_norm in answer_norm or answer_norm in gt_norm:
        return 1.0

    similarity = SequenceMatcher(None, answer_norm, gt_norm).ratio()
    if similarity >= 0.75:
        return similarity

    answer_tokens = set(answer_norm.split())
    gt_tokens = set(gt_norm.split())
    if not answer_tokens or not gt_tokens:
        return 0.0

    overlap = len(answer_tokens & gt_tokens) / max(len(gt_tokens), 1)
    return overlap


def load_test_dataset():
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("testcase.json 必须是列表格式")
    return dataset


def build_rag_context(embedder, milvus, reranker, query: str, top_k: int = 15):
    q_vec = embedder.encode(query).tolist()
    search_res = milvus.search(
        collection_name=COLLECTION_NAME,
        data=[q_vec],
        limit=top_k,
        output_fields=["text", "source"],
    )
    retrieved_texts = [hit["entity"]["text"] for hit in search_res[0]]
    sources = [hit["entity"]["source"] for hit in search_res[0]]

    if not retrieved_texts:
        return [], []

    pairs = [[query, txt] for txt in retrieved_texts]
    scores = reranker.compute_score(pairs)
    scored = sorted(zip(retrieved_texts, sources, scores), key=lambda x: x[2], reverse=True)
    top_texts = [x[0] for x in scored[:5]]
    top_sources = [x[1] for x in scored[:5]]
    return top_texts, top_sources


def call_ollama(question: str, context_text: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个严谨的企业知识库智能助手。请严格根据提供的参考上下文回答用户问题。\n"
                    "如果上下文中没有包含相关答案，请直接回答“根据现有知识库，未找到相关内容”，严禁胡乱编造。"
                ),
            },
            {
                "role": "user",
                "content": f"【参考上下文】\n{context_text}\n\n【用户问题】\n{question}",
            },
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    try:
        session = requests.Session()
        session.trust_env = False
        response = session.post("http://localhost:11434/api/chat", json=payload, timeout=60)
        if response.status_code == 200:
            res_json = response.json()
            return res_json.get("message", {}).get("content", "生成失败").strip()
        return f"Ollama报错 (状态码 {response.status_code}): {response.text}".strip()
    except Exception as exc:
        return f"请求解析异常: {exc}"


def init_eval_env():
    print("正在初始化 RAG 评测环境...")
    try:
        embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        milvus = MilvusClient(str(DB_PATH))
        if not milvus.has_collection(COLLECTION_NAME):
            raise FileNotFoundError(f"未找到集合 '{COLLECTION_NAME}'，请先执行 ingest01.py 建库。")
        milvus.load_collection(COLLECTION_NAME)
        reranker = FlagReranker("BAAI/bge-reranker-large", use_fp16=True)
        return embedder, milvus, reranker
    except Exception as exc:
        print(f"❌ 评测环境初始化失败: {exc}")
        error_text = str(exc).lower()
        if "locked" in error_text or "resource temporarily unavailable" in error_text:
            print("另一个进程正在占用 milvus_demo.db，请关闭 Streamlit / ingest01 / check_db 等进程后再重试。")
        elif "open local milvus failed" in error_text or "connectionconfigexception" in error_text:
            print("Milvus 本地数据库无法启动，请确认数据库目录没有被其他进程占用。")
        raise SystemExit(1)


def main():
    try:
        dataset = load_test_dataset()
    except FileNotFoundError:
        print(f"❌ 找不到测试集文件: {DATASET_PATH}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"❌ 读取测试集失败: {exc}")
        raise SystemExit(1)

    try:
        embedder, milvus, reranker = init_eval_env()
    except SystemExit:
        raise

    print(f"成功加载 {len(dataset)} 条测试题目，开始批量评测...\n" + "=" * 60)

    report_data = []
    passed_count = 0

    for item in dataset:
        q_id = item["id"]
        query = item["question"]
        standard_answer = item["ground_truth"]
        category = item.get("category", "通用")

        print(f"\n【测试题 {q_id}】({category}): {query}")
        top_texts, top_sources = build_rag_context(embedder, milvus, reranker, query)

        print(f"📄 召回关键切片数: {len(top_texts)} 个")
        for s_idx, (txt, src) in enumerate(zip(top_texts[:2], top_sources[:2])):
            snippet = txt.replace("\n", " ")[:70]
            print(f"   └─ [切片 {s_idx + 1}] 来源: {src} | 内容摘要: {snippet}...")

        context_str = "\n\n".join(top_texts)
        llm_answer = call_ollama(query, context_str)
        score = evaluate_answer(llm_answer, standard_answer)
        passed = score >= 0.75

        if passed:
            passed_count += 1

        print(f"🎯 标准答案: {standard_answer}")
        print(f"🤖 大模型回答: {llm_answer.strip()}")
        print(f"📊 相似度/判定分: {score:.3f} => {'通过' if passed else '失败'}")
        print("-" * 60)

        report_data.append(
            {
                "id": q_id,
                "category": category,
                "question": query,
                "ground_truth": standard_answer,
                "retrieved_chunks": top_texts[:3],
                "llm_answer": llm_answer.strip(),
                "score": round(score, 4),
                "passed": passed,
            }
        )

    accuracy = passed_count / len(report_data) if report_data else 0.0
    summary = {
        "total": len(report_data),
        "passed": passed_count,
        "failed": len(report_data) - passed_count,
        "accuracy": round(accuracy, 4),
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": report_data}, f, ensure_ascii=False, indent=2)

    print("\n🎉 批量评测完毕！")
    print(f"✅ 通过: {passed_count} / {len(report_data)}")
    print(f"📈 准确率: {accuracy:.2%}")
    print(f"📁 评测报告已保存至: {REPORT_PATH}")


if __name__ == "__main__":
    main()
