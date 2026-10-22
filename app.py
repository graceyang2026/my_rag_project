import streamlit as st
from qa02 import ask_rag  # 直接调用你带 02 的那个后端文件

# 页面基础配置（居中布局，干净清爽）
st.set_page_config(
    page_title="知识库智能问答",
    page_icon="💬",
    layout="centered"
)

# 极简标题
st.title("💬 智能问答助手")
st.caption("基于本地 RAG 知识库的智能问答")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果有切片来源，渲染折叠面板
        if "references" in message and message["references"]:
            with st.expander("🔍 查看引用的文档切片来源"):
                for idx, (txt, src) in enumerate(message["references"]):
                    st.markdown(f"**[切片 {idx+1}]** 来源: `{src}`")
                    st.caption(txt)

# 底部聊天输入框
if user_query := st.chat_input("请输入您的问题..."):
    
    # 1. 记录并展示用户输入
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # 2. 调用 qa02.py 中的 ask_rag 函数获取回答与切片
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            answer, references = ask_rag(user_query)
            
        # 3. 输出回答
        st.markdown(answer)
        
        # 4. 输出可折叠的切片来源
        if references:
            with st.expander("🔍 查看引用的文档切片来源"):
                for idx, (txt, src) in enumerate(references):
                    st.markdown(f"**[切片 {idx+1}]** 来源: `{src}`")
                    st.caption(txt)
                    
        # 5. 保存到历史记录
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "references": references
        })