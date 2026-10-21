import os
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

# ==========================================
# 1. 初始化各类开源组件（自研配置）
# ==========================================
print("正在加载向量模型...")
# 使用中文效果优秀的开源嵌入模型
embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")

print("正在初始化本地向量数据库 PyMilvus...")
# 在本地生成一个 milvus_demo.db 文件作为向量数据库
milvus_client = MilvusClient("./milvus_demo.db")

print("正在初始化 IBM Docling 解析器...")
converter = DocumentConverter()


# ==========================================
# 2. 自研函数：解析 PDF 并格式转换、向量化入库
# ==========================================
def ingest_pdf(pdf_path: str, collection_name: str = "rag_kb"):
    """
    自研逻辑：接入 Docling 解析结果 -> 格式转换与文本切片 -> 生成向量 -> 存入 Milvus
    """
    if not os.path.exists(pdf_path):
        print(f"错误：找不到文件 {pdf_path}")
        return

    print(f"\n[1/4] 正在使用 IBM Docling 解析 PDF 文件: {pdf_path} ...")
    # 调用 IBM Docling 进行 PDF 版面解析与表格提取
    result = converter.convert(pdf_path)
    # 将解析结果转为 Markdown 格式（格式转换）
    markdown_text = result.document.export_to_markdown()
 
    print("【Docling 提取的文字预览】：\n", markdown_text[:500])
    print("[2/4] 正在进行自研文本切片 (Chunking)...")
    # 自研逻辑：将长文本按 400 字进行切片，相邻切片重叠 50 字（保持上下文连贯）
    chunk_size = 600
    overlap = 150
    chunks = []
    
    start = 0
    while start < len(markdown_text):
        end = start + chunk_size
        chunk = markdown_text[start:end]
        if chunk.strip():  # 过滤空文本
            chunks.append(chunk)
        start += (chunk_size - overlap)
        
    print(f"成功将文档切分为 {len(chunks)} 个切片！")

    print("[3/4] 正在使用 Sentence-Transformers 生成文本向量...")
    # 为每个切片生成向量
    vectors = embedder.encode(chunks)

    print("[4/4] 正在配置 Milvus 索引并写入数据...")
    # 如果数据库集合不存在，则自动创建（维度 512 匹配 bge-small 模型）
    # 如果集合已存在，先删除旧集合，实现“覆盖更新”
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
    if not milvus_client.has_collection(collection_name):
        milvus_client.create_collection(
            collection_name=collection_name,
            dimension=512
        )

    # 组装数据条目（接口胶水代码）
    data = []
    file_name = os.path.basename(pdf_path)
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        data.append({
            "id": idx,
            "vector": vector.tolist(),  # 转换成 list 格式存入
            "text": chunk,              # 保存切片原文
            "source": file_name          # 保存来源文件名
        })

    # 执行写入
    milvus_client.insert(collection_name=collection_name, data=data)
    print(f"✅ 入库成功！共计存入 {len(data)} 条数据到集合 '{collection_name}' 中。")


# ==========================================
# 3. 本地测试运行
# ==========================================
if __name__ == "__main__":
    # 可以在这里放一个你本地的测试 PDF 文件路径进行测试
    # 比如：test.pdf
    test_pdf = "供应商管理系统需求规格说明书.pdf"
    print("hello!!!!")
    print("当前工作目录是：", os.getcwd(), os.path.exists(test_pdf))
    print("是否存在：",  os.path.exists(test_pdf))
    if os.path.exists(test_pdf):
        ingest_pdf(test_pdf)
    else:
        print(f"提示：请在 my_rag_project 文件夹下放一个名为 '{test_pdf}' 的文件，然后运行此代码测试入库。")