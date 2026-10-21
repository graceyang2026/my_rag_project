# import streamlit as st

# st.title("RAG 智能问答系统")

# question = st.text_input("请输入你的问题")

# if question:
#     st.write("你的问题是：", question)
#     st.write("答：","我是RAG智能问答系统（这里接 RAG + Qwen7B 返回结果）")


import streamlit as st

st.set_page_config(page_title="RAG智能问答", page_icon="🤖")

st.title("🤖 RAG 智能问答系统")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框（像 ChatGPT）
question = st.chat_input("请输入你的问题")

if question:

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(question)

    # 保存用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # ===== 这里接你的 RAG + Qwen =====
    # 假设这里是你的模型返回
    answer = f"我是 RAG 智能问答系统，你刚才问的是：{question}"

    answer = f"我是 RAG 智能问答系统"

    # 显示助手消息
    with st.chat_message("assistant"):
        st.markdown(answer)

    # 保存助手消息
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })