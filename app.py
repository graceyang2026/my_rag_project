import streamlit as st
from qa02 import ask_rag  # 导入你的 qa 模块（根据你的文件名调整导入方式）

st.title("RAG 智能问答")

if query := st.chat_input("请输入问题..."):
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("正在检索与生成..."):
            # 直接调用 02_qa.py 里的核心函数！
            answer, references = ask_rag(query)
            
        st.write(answer)
        
        # 如果你想在网页上顺便展示引用的切片来源
        if references:
            with st.expander("🔍 查看引用的切片来源"):
                for idx, (txt, src) in enumerate(references):
                    st.markdown(f"**[切片 {idx+1}] 来源: {src}**\n\n> {txt}")