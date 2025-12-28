import streamlit as st
import os
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

import dotenv
from dotenv import load_dotenv

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("hf")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("hf")

st.set_page_config(page_title="AI Chatbot Mentor", layout="centered")

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title(" AI Chatbot Mentor")
st.write("Your personalized AI learning assistant")

module = st.selectbox(
    " Select Learning Module",
    [
        "Python",
        "SQL",
        "Power BI",
        "Exploratory Data Analysis (EDA)",
        "Machine Learning",
        "Deep Learning",
        "Generative AI",
        "Agentic AI",
    ],
)

mentor_experience = st.number_input(
    " Select Mentor Experience (Years)",
    min_value=1,
    max_value=50,
    step=1,
)

start_btn = st.button("Start Mentoring Session", type="primary")

if start_btn:
    st.session_state.chat_started = True
    st.session_state.chat_history = []

if st.session_state.chat_started:

    st.subheader(f" Welcome to {module} AI Mentor")
    st.write(
        f"I am your dedicated **{module} mentor** with **{mentor_experience} year(s) of experience**."
    )
    st.write("How can I help you today?")
   
    for role, msg in st.session_state.chat_history:
        st.chat_message(role).write(msg)

    user_question = st.chat_input("Type your query...")

    if user_question:
        st.chat_message("user").write(user_question)
        st.session_state.chat_history.append(("user", user_question))

        system_prompt = f"""
You are an AI Mentor with {mentor_experience} years of experience.
You ONLY answer questions related to the module: {module}.

Rules:
- If the question is NOT related to {module}, reply exactly:
  "Sorry, I don’t know about this question. Please ask something related to the selected module : {module}."
- Keep answers educational, structured, and clear.
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )

        model = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-V3.2",temperature=0.6
        )

        chatmodel = ChatHuggingFace(llm=model)

        chain = prompt | chatmodel
        response = chain.invoke({"question": user_question})
        response = str(response.content)

        st.session_state.chat_history.append(("assistant", response))
        st.chat_message("assistant").write(response)

    st.divider()
    st.subheader(" Download Conversation")

    def create_txt(chat):
        file_path = "chat_history.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for role, msg in chat:
                f.write(f"{role.upper()}: {msg}\n\n")
        return file_path

    def create_pdf(chat):
        file_path = "chat_history.pdf"
        styles = getSampleStyleSheet()
        story = []

        for role, msg in chat:
            story.append(
                Paragraph(f"<b>{role.upper()}:</b> {msg}", styles["Normal"])
            )

        pdf = SimpleDocTemplate(file_path)
        pdf.build(story)
        return file_path

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Download TXT",type="primary"):
            txt_file = create_txt(st.session_state.chat_history)
            with open(txt_file, "rb") as f:
                st.download_button(
                    "Click to Download .TXT", f, file_name="chat_history.txt"
                )

    with col2:
        if st.button("Download PDF",type="primary"):
            pdf_file = create_pdf(st.session_state.chat_history)
            with open(pdf_file, "rb") as f:
                st.download_button(
                    "Click to Download PDF", f, file_name="chat_history.pdf"
                )

    with col3:
        if st.button(" Close Chat",type="primary"):
            st.session_state.chat_started = False
            st.session_state.chat_history = []
            st.rerun()
