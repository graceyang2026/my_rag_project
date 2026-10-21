from pymilvus import MilvusClient

# 1. 连接到本地数据库文件
milvus_client = MilvusClient("./milvus_demo.db")

# 2. 查看当前数据库里有哪些 Collection
collections = milvus_client.list_collections()
print("当前所有的集合列表：", collections)

if "rag_kb" in collections:
    # 🌟 关键修复：在查询前，必须先显式加载集合到内存中！
    milvus_client.load_collection(collection_name="rag_kb")
    
    # 3. 查看集合统计信息
    res = milvus_client.get_collection_stats(collection_name="rag_kb")
    print("rag_kb 集合的统计信息：", res)
    
    # 4. 抽样查看库中的前 5 条切片内容
    query_res = milvus_client.query(
        collection_name="rag_kb",
        filter="id >= 0",
        output_fields=["text", "source"],
        limit=5
    )
    print("\n--- 抽样查看库中的前 5 条切片内容 ---")
    for item in query_res:
        print(f"ID: {item['id']} | 来源: {item['source']}")
        print(f"文本内容片段: {item['text'][:100]}...\n")