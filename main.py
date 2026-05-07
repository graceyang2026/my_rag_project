# /Users/grace/Desktop/PIP2026/python/my_rag_project/main.py
from parse_pdf import convert_pdf_to_md
import os

def start_pipeline():
    # 统一管理路径
    target_pdf = "./data/test.pdf" 
    
    if not os.path.exists(target_pdf):
        print(f"找不到输入文件: {target_pdf}")
        return

    # 调用解析服务
    success, result, duration, content = convert_pdf_to_md(target_pdf)

    if success:
        print(f"解析成功！")
        print(f"耗时: {duration:.2f} 秒")
        print(f"结果已存至: {result}")
        # 这里可以预留：后续对接 Milvus 的逻辑
    else:
        print(f"解析失败: {result}")

if __name__ == "__main__":
    start_pipeline()