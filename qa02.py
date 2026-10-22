import requests
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from FlagEmbedding import FlagReranker
import streamlit as st

# 使用 Streamlit 缓存加载模型，避免重复加载和启动空白
@st.cache_resource
def init_models():
    print("正在加载嵌入模型 (Sentence-Transformers)...")
    embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    
    print("正在连接本地向量数据库 (Milvus)...")
    milvus_client = MilvusClient("./milvus_demo.db")
    
    print("正在加载二次精排模型 (BGE-Reranker)...")
    reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
    
    return embedder, milvus_client, reranker

def ask_rag(query: str):
    # 在函数内部获取组件
    embedder, milvus_client, reranker = init_models()
    collection_name = "rag_kb"
    
    if milvus_client.has_collection(collection_name):
        milvus_client.load_collection(collection_name=collection_name)
    
    # 1. 向量检索 (Top 8)
    query_vector = embedder.encode(query).tolist()
    search_results = milvus_client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=15,
        output_fields=["text", "source"]
    )
    
    retrieved_texts = [hit["entity"]["text"] for hit in search_results[0]]
    sources = [hit["entity"]["source"] for hit in search_results[0]]
    
    if not retrieved_texts:
        return "根据现有知识库，未找到相关内容。", []

    # 2. BGE-Reranker 精排 (Top 5)
    pairs = [[query, txt] for txt in retrieved_texts]
    scores = reranker.compute_score(pairs)
    
    scored_results = sorted(zip(retrieved_texts, sources, scores), key=lambda x: x[2], reverse=True)
    top_items = scored_results[:5]
    # 2. 精排：筛选前 Top 4 给大模型作为参考上下文
    top_texts = [item[0] for item in scored_results[:5]]  # 🌟 取前 4 个最相关的切片
    top_sources = [item[1] for item in scored_results[:5]]

    
    references = list(zip(top_texts, top_sources))

    # 3. 组装 Prompt 调用本地 Ollama
    context_str = "\n".join(top_texts)
    # system_prompt = (
    #     "你是一个严谨的企业知识库智能助手。请严格根据提供的参考上下文回答用户问题。\n"
    #     "如果上下文中没有包含相关答案，请直接回答“根据现有知识库，未找到相关内容”，严禁胡乱编造（幻觉）。"
    # )
    # user_prompt = f"参考以下内容回答：\n{context_str}\n\n问题：{query}"
    
    # payload = {
    #     "model": "qwen2.5:7b",
    #     "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
    #     "stream": False,
    #     "options": {"temperature": 0.1}
    # }


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
    
    try:
        # 在这里加上 proxies 参数，强制不走代理直接连本地
        response = requests.post(
            "http://localhost:11434/api/generate", 
            json=payload, 
            timeout=60,
            proxies={"http": None, "https": None}  # <-- 把这行加进来
        )
        if response.status_code == 200:
            answer = response.json().get("response", "生成失败")
            return answer, references
        else:
            return f"⚠️ Ollama 报错 [状态码: {response.status_code}]", []
    except Exception as e:
        return f"⚠️ 连接异常: {str(e)}", []