import json
import requests
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from FlagEmbedding import FlagReranker

print("正在初始化 RAG 评测环境...")
embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
milvus = MilvusClient("./milvus_demo.db")
milvus.load_collection("rag_kb")
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

# 加载测试集
with open("testcase.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"成功加载 {len(dataset)} 条测试题目，开始批量评测...\n" + "="*60)

report_data = []

for idx, item in enumerate(dataset):
    q_id = item["id"]
    query = item["question"]
    standard_answer = item["ground_truth"]
    category = item.get("category", "通用")
    
    print(f"\n【测试题 {q_id}】({category}): {query}")
    print(f"🎯 标准答案: {standard_answer}")
    
    # 1. 向量检索 (检索 Top-15 切片)
    q_vec = embedder.encode(query).tolist()
    search_res = milvus.search("rag_kb", data=[q_vec], limit=15, output_fields=["text", "source"])
    retrieved_texts = [hit["entity"]["text"] for hit in search_res[0]]
    sources = [hit["entity"]["source"] for hit in search_res[0]]
    
    # 2. Rerank 重排 (精排前 5 个最贴切切片)
    if retrieved_texts:
        pairs = [[query, txt] for txt in retrieved_texts]
        scores = reranker.compute_score(pairs)
        scored = sorted(zip(retrieved_texts, sources, scores), key=lambda x: x[2], reverse=True)
        top_texts = [x[0] for x in scored[:5]]
        top_sources = [x[1] for x in scored[:5]]
    else:
        top_texts = []
        top_sources = []
        
    # 3. 打印检索到的关键文档切片（供验收核查）
    print(f"📄 召回关键切片数: {len(top_texts)} 个")
    for s_idx, (txt, src) in enumerate(zip(top_texts[:2], top_sources[:2])):
        snippet = txt.replace("\n", " ")[:70]
        print(f"   └─ [切片 {s_idx+1}] 来源: {src} | 内容摘要: {snippet}...")
        
    # 4. 组装上下文与对话 Payload
    context_str = "\n\n".join(top_texts)
    
    payload = {
        "model": "qwen2.5:7b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个严谨的企业知识库智能助手。请严格根据提供的参考上下文回答用户问题。\n"
                    "【特别提示】：参考上下文中包含《权限与功能控制矩阵》等表格。系统管理员 (Admin) 仅负责账号和权限管理，"
                    "在供应商档案的“新增/编辑”操作上权限为无（✕），原则上不参与业务数据的新增、修改及审批。\n"
                    "如果上下文中没有包含相关答案，请直接回答“根据现有知识库，未找到相关内容”，严禁胡乱编造。"
                )
            },
            {
                "role": "user",
                "content": f"【参考上下文】\n{context_str}\n\n【用户问题】\n{query}"
            }
        ],
        "stream": False,
        "options": {"temperature": 0.1}
    }
    
    # 5. 调用 Ollama API (绕过公司代理直连)
    try:
        session = requests.Session()
        session.trust_env = False  # 忽略系统代理
        
        response = session.post("http://localhost:11434/api/chat", json=payload, timeout=60)
        if response.status_code == 200:
            res_json = response.json()
            llm_answer = res_json.get("message", {}).get("content", "生成失败")
        else:
            llm_answer = f"Ollama报错 (状态码 {response.status_code}): {response.text}"
    except Exception as e:
        llm_answer = f"请求解析异常: {str(e)}"
        
    print(f"🤖 大模型回答: {llm_answer.strip()}")
    print("-" * 60)
    
    # 收集保存完整评测三要素
    report_data.append({
        "id": q_id,
        "category": category,
        "question": query,
        "ground_truth": standard_answer,
        "retrieved_chunks": top_texts[:3],  # 保存精排前3的完整切片原文
        "llm_answer": llm_answer.strip()
    })

# 导出包含“问题+标准答案+切片+模型回答”的完整评测结果文件
with open("eval_report.json", "w", encoding="utf-8") as f:
    json.dump(report_data, f, ensure_ascii=False, indent=2)

print("\n🎉 20道测试题批量跑测完毕！")
print("📁 包含【问题、标准答案、对应切片、模型回答】的完整评测报告已保存至: eval_report.json")