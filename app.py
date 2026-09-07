import streamlit as st

from rag import ask_agent
from vector_store import process_uploaded_pdf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Guide AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "action_mode" not in st.session_state:
    st.session_state.action_mode = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎓 Student Guide AI")

st.sidebar.caption(
    "Your AI-powered academic study assistant"
)

st.sidebar.divider()


# ============================================================
# AI AGENT STATUS
# ============================================================

st.sidebar.header("🤖 AI Agent")

st.sidebar.success("🟢 Agent Active")

st.sidebar.caption(
    "The AI Agent understands your request and "
    "uses the RAG system to answer from uploaded documents."
)


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.sidebar.header("📚 Study Material")

uploaded_files = st.sidebar.file_uploader(
    "Upload your PDF notes",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more PDF study materials."
)


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if uploaded_files:

    for uploaded_file in uploaded_files:

        file_key = uploaded_file.name

        if file_key not in st.session_state.processed_files:

            with st.sidebar.spinner(
                f"Processing {uploaded_file.name}..."
            ):

                result = process_uploaded_pdf(
                    uploaded_file
                )

            if result["success"]:

                st.session_state.processed_files.add(
                    file_key
                )

                st.sidebar.success(
                    f"✅ {uploaded_file.name} ready"
                )

                st.sidebar.caption(
                    f"{result['pages']} pages • "
                    f"{result['chunks']} chunks"
                )

            else:

                st.sidebar.error(
                    f"❌ Could not process "
                    f"{uploaded_file.name}"
                )

                st.sidebar.caption(
                    result["error"]
                )

else:

    st.sidebar.info(
        "Upload your study material to get started."
    )


# ============================================================
# UPLOADED FILES
# ============================================================

if uploaded_files:

    with st.sidebar.expander(
        f"📚 Uploaded Files ({len(uploaded_files)})",
        expanded=False
    ):

        for file in uploaded_files:

            st.markdown(
                f"📄 **{file.name}**"
            )


st.sidebar.divider()


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.sidebar.header("⚙️ System Information")

st.sidebar.write(
    f"📚 Documents: "
    f"**{len(uploaded_files) if uploaded_files else 0}**"
)

# CHANGED: Gemini Flash → Groq GPT-OSS 20B
st.sidebar.write(
    "⚡ AI Model: **Groq GPT-OSS 20B**"
)

# KEEP GEMINI FOR EMBEDDINGS
st.sidebar.write(
    "🧠 Embeddings: **Gemini Embedding 001**"
)

st.sidebar.write(
    "🗄️ Vector DB: **ChromaDB**"
)

st.sidebar.write(
    "🔍 Retrieval: **Similarity Search**"
)

st.sidebar.write(
    "🔗 Architecture: **RAG + AI Agent**"
)


st.sidebar.divider()


# ============================================================
# CHAT CONTROLS
# ============================================================

st.sidebar.header("🛠️ Chat Controls")

if st.sidebar.button(
    "🗑️ Clear Chat",
    use_container_width=True
):

    st.session_state.messages = []

    st.session_state.action_mode = None

    st.rerun()


st.sidebar.divider()


# ============================================================
# ABOUT
# ============================================================

st.sidebar.header("ℹ️ About")

st.sidebar.info(
    """
**Student Guide AI**

An AI-powered study assistant that helps
students understand and revise their
academic material.

**Core Technologies**

• Python
• Streamlit
• Google Gemini
• LangChain
• ChromaDB
• RAG
• AI Agent
"""
)


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🎓 Student Guide AI")

st.markdown(
    """
### Your personal AI study assistant

Upload your academic notes and ask questions
in natural language.

The AI Agent can help you with:

**Answers • Explanations • Summaries • Exam Questions**
"""
)


# ============================================================
# TOP STATUS CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    if uploaded_files:

        st.success(
            "📚 Documents Ready"
        )

    else:

        st.warning(
            "📚 No Documents"
        )


with col2:

    st.success(
        "🤖 AI Agent Active"
    )


with col3:

    # CHANGED: Gemini Flash → Groq GPT-OSS 20B
    st.info(
        "⚡ Groq GPT-OSS 20B"
    )


with col4:

    st.info(
        "🗄️ ChromaDB"
    )


# ============================================================
# DOCUMENT STATUS
# ============================================================

st.divider()

if uploaded_files:

    st.success(
        f"✅ {len(uploaded_files)} "
        f"document(s) ready for questions."
    )

else:

    st.warning(
        "📄 Please upload a PDF from the sidebar to begin."
    )


# ============================================================
# QUICK ACTIONS
# ============================================================

st.subheader("🎯 Quick Study Actions")

st.caption(
    "Choose how you want the AI Agent to help you."
)


col1, col2, col3 = st.columns(3)


with col1:

    if st.button(
        "💡 Explain a Topic",
        use_container_width=True
    ):

        st.session_state.action_mode = "explain"


with col2:

    if st.button(
        "📝 Summarize",
        use_container_width=True
    ):

        st.session_state.action_mode = "summarize"


with col3:

    if st.button(
        "❓ Generate Questions",
        use_container_width=True
    ):

        st.session_state.action_mode = "questions"


# ============================================================
# ACTIVE MODE
# ============================================================

if st.session_state.action_mode == "explain":

    st.info(
        "💡 **Explain Mode** — "
        "The Agent will explain the topic "
        "in simple student-friendly language."
    )


elif st.session_state.action_mode == "summarize":

    st.info(
        "📝 **Summary Mode** — "
        "The Agent will create concise "
        "revision points."
    )


elif st.session_state.action_mode == "questions":

    st.info(
        "❓ **Question Mode** — "
        "The Agent will generate important "
        "exam-oriented questions."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

st.divider()

st.subheader("💬 Study Conversation")


if not st.session_state.messages:

    st.info(
        "👋 Start by asking a question about "
        "your uploaded study material."
    )


else:

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(
                message["content"]
            )


            # ------------------------------------------------
            # SHOW AGENT TASK
            # ------------------------------------------------

            if (
                message["role"] == "assistant"
                and message.get("task")
            ):

                task_names = {

                    "question_answering":
                        "Question Answering",

                    "simple_explanation":
                        "Simple Explanation",

                    "summarization":
                        "Summarization",

                    "question_generation":
                        "Question Generation",

                    "advantages_disadvantages":
                        "Advantages / Disadvantages",

                    "error":
                        "Error"
                }


                task_display = task_names.get(
                    message["task"],
                    message["task"]
                )


                st.caption(
                    f"🤖 Agent Task: **{task_display}**"
                )


            # ------------------------------------------------
            # SHOW SOURCES
            # ------------------------------------------------

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                with st.expander(
                    "📚 View Sources"
                ):

                    for source in message["sources"]:

                        filename = source["source"]

                        page = source["page"]


                        if page:

                            st.write(
                                f"📄 {filename} — "
                                f"Page {page}"
                            )

                        else:

                            st.write(
                                f"📄 {filename}"
                            )


# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state.action_mode == "explain":

    placeholder = (
        "Enter a topic to explain..."
    )


elif st.session_state.action_mode == "summarize":

    placeholder = (
        "Enter a topic to summarize..."
    )


elif st.session_state.action_mode == "questions":

    placeholder = (
        "Enter a topic for exam questions..."
    )


else:

    placeholder = (
        "Ask something from your study material..."
    )


question = st.chat_input(
    placeholder
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # CHECK DOCUMENT
    # --------------------------------------------------------

    if not uploaded_files:

        st.warning(
            "📄 Please upload at least one PDF "
            "before asking a question."
        )

        st.stop()


    # --------------------------------------------------------
    # BUILD USER REQUEST
    # --------------------------------------------------------

    if st.session_state.action_mode == "explain":

        final_question = (
            f"Explain the topic '{question}' "
            f"in very simple student-friendly "
            f"language. Include a clear example "
            f"if the information is available "
            f"in the uploaded documents."
        )


    elif st.session_state.action_mode == "summarize":

        final_question = (
            f"Summarize the topic '{question}' "
            f"using concise important points "
            f"suitable for exam revision. "
            f"Use only the uploaded documents."
        )


    elif st.session_state.action_mode == "questions":

        final_question = (
            f"Generate important exam-oriented "
            f"questions about '{question}' "
            f"using only the uploaded documents. "
            f"Include a mixture of short-answer "
            f"and conceptual questions."
        )


    else:

        final_question = question


    # --------------------------------------------------------
    # STORE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # GENERATE AI AGENT RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "⚡ AI Agent is analyzing your study material..."
        ):

            try:

                result = ask_agent(
                    final_question
                )


                answer = result["answer"]

                sources = result.get(
                    "sources",
                    []
                )

                task = result.get(
                    "task",
                    "question_answering"
                )


            except Exception as e:

                answer = (
                    "Sorry, something went wrong "
                    "while processing your request."
                )

                sources = []

                task = "error"

                st.error(
                    str(e)
                )


        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.markdown(answer)


        # ----------------------------------------------------
        # AGENT TASK
        # ----------------------------------------------------

        task_names = {

            "question_answering":
                "Question Answering",

            "simple_explanation":
                "Simple Explanation",

            "summarization":
                "Summarization",

            "question_generation":
                "Question Generation",

            "advantages_disadvantages":
                "Advantages / Disadvantages",

            "error":
                "Error"
        }


        task_display = task_names.get(
            task,
            task
        )


        st.caption(
            f"🤖 Agent Task: **{task_display}**"
        )


        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if sources:

            with st.expander(
                "📚 View Sources"
            ):

                for source in sources:

                    filename = source["source"]

                    page = source["page"]


                    if page:

                        st.write(
                            f"📄 {filename} — "
                            f"Page {page}"
                        )

                    else:

                        st.write(
                            f"📄 {filename}"
                        )


    # --------------------------------------------------------
    # STORE ASSISTANT RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "task": task
        }
    )


    # --------------------------------------------------------
    # RESET ACTION MODE
    # --------------------------------------------------------

    st.session_state.action_mode = None

    st.rerun()