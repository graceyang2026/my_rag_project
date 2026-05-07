# /Users/grace/Desktop/PIP2026/python/my_rag_project/parse_pdf.py
import os
import time
from docling.document_converter import DocumentConverter

def convert_pdf_to_md(input_file, output_dir="./output"):
    """
    封装解析逻辑，供 main.py 调用
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[M2 Acceleration] 启动 Docling 解析: {os.path.basename(input_file)}")
    start_time = time.time()

    try:
        # 1. 初始化解析器
        converter = DocumentConverter()
        
        # 2. 执行转换
        result = converter.convert(input_file)
        
        # 3. 导出为 Markdown
        markdown_content = result.document.export_to_markdown()
        
        # 4. 生成输出路径
        base_name = os.path.basename(input_file).replace(".pdf", ".md")
        output_path = os.path.join(output_dir, base_name)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            
        execution_time = time.time() - start_time
        return True, output_path, execution_time, markdown_content
        
    except Exception as e:
        return False, str(e), 0, ""