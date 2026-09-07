from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

import time


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "chroma_db"

MAX_HISTORY = 6
RETRIEVAL_K = 3

NO_ANSWER = "I couldn't find the answer in the uploaded documents."


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# ============================================================
# CHROMADB
# ============================================================

vector_store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": RETRIEVAL_K
    },
)


# ============================================================
# GEMINI MODEL
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    max_retries=2,
)


# ============================================================
# CONVERSATION MEMORY
# ============================================================

_history = []


# ============================================================
# CUSTOM LANGCHAIN TOOL
# ============================================================

@tool
def search_study_material(topic: str) -> str:
    """
    Search the uploaded academic study material for information
    relevant to the student's question or topic.
    """

    try:

        docs = retriever.invoke(topic)

        if not docs:
            return "NO_RELEVANT_DOCUMENTS"

        results = []

        for i, doc in enumerate(docs, 1):

            source = doc.metadata.get(
                "source",
                "Unknown document"
            )

            page = doc.metadata.get(
                "page",
                None
            )

            if page is not None:
                try:
                    page = int(page) + 1
                except Exception:
                    pass

            results.append(
                f"[Document {i}]\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content:\n{doc.page_content}"
            )

        return "\n\n".join(results)

    except Exception as e:

        return (
            f"TOOL_ERROR: "
            f"{type(e).__name__}: {str(e)}"
        )


# ============================================================
# TOOL BINDING
# ============================================================

tool_llm = llm.bind_tools(
    [search_study_material]
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Student Guide AI, an academic study assistant.

Your purpose is to help students understand and revise their
uploaded academic study material.

IMPORTANT RULES:

1. Answer ONLY using the uploaded study material.

2. Do not invent facts.

3. Do not use unsupported outside knowledge.

4. Keep answers clear, accurate and student-friendly.

5. For questions, explanations, summaries and study questions,
   use the retrieved document context.

6. If the uploaded documents do not contain the answer, reply:

I couldn't find the answer in the uploaded documents.

7. For follow-up questions such as:
   - explain it
   - explain simply
   - give an example
   - why?
   - how?
   - summarize it

   use the previous topic from the conversation.

8. Do not switch to an unrelated topic.

9. If the user asks for a simple explanation, explain the
   previous topic in easy language instead of changing topics.

Conversation History:
{history}

Retrieved Study Material:
{context}
"""


# ============================================================
# GET LAST USER QUESTION
# ============================================================

def get_last_user_question():

    for item in reversed(_history):

        if item.startswith("User:"):

            return item.replace(
                "User:",
                "",
                1
            ).strip()

    return None


# ============================================================
# FOLLOW-UP DETECTION
# ============================================================

def is_follow_up(question: str):

    q = question.lower().strip()

    follow_up_phrases = [

        "explain it",
        "explain this",
        "explain that",

        "explain simply",
        "explain in simple",
        "explain in simple words",
        "explain in simple language",

        "simple explanation",
        "simple words",

        "easy explanation",
        "easy words",

        "in simple words",
        "in easy words",

        "give an example",
        "give example",
        "give examples",
        "examples",

        "why",
        "how",

        "summarize it",
        "summarise it",

        "summary",

        "advantages",
        "disadvantages",

        "pros",
        "cons",

    ]

    short_question = (
        len(question.split()) <= 7
    )

    phrase_match = any(
        phrase in q
        for phrase in follow_up_phrases
    )

    return (
        short_question
        or phrase_match
    )


# ============================================================
# BUILD RETRIEVAL QUERY
# ============================================================

def build_retrieval_query(
    question: str,
    previous_topic: str = None
):

    if (
        previous_topic
        and is_follow_up(question)
    ):

        return (
            f"Previous topic: {previous_topic}\n"
            f"Follow-up request: {question}"
        )

    return question


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def extract_sources(docs):

    sources = []

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown document"
        )

        page = doc.metadata.get(
            "page",
            None
        )

        if page is not None:

            try:
                page = int(page) + 1

            except Exception:
                pass

        item = {
            "source": source,
            "page": page,
        }

        if item not in sources:
            sources.append(item)

    return sources


# ============================================================
# GEMINI RETRY FUNCTION
# ============================================================

def invoke_with_retry(
    chain,
    payload,
    attempts=3
):

    last_error = None

    for attempt in range(
        1,
        attempts + 1
    ):

        try:

            print(
                f"DEBUG: Gemini generation "
                f"attempt {attempt}/{attempts}"
            )

            return chain.invoke(
                payload
            )

        except Exception as e:

            last_error = e

            error_text = str(e).upper()

            temporary_error = any(
                code in error_text
                for code in [

                    "503",
                    "UNAVAILABLE",

                    "429",
                    "RESOURCE_EXHAUSTED",

                    "TIMEOUT",
                    "READTIMEOUT",

                ]
            )

            if (
                not temporary_error
                or attempt == attempts
            ):

                raise

            wait_time = attempt * 2

            print(
                "DEBUG: Temporary Gemini error."
            )

            print(
                f"DEBUG: Retrying in "
                f"{wait_time} seconds..."
            )

            time.sleep(
                wait_time
            )

    raise last_error


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def ask_question(
    question: str,
    memory_question: str = None
):

    global _history

    try:

        # ----------------------------------------------------
        # IMPORTANT MEMORY FIX
        # ----------------------------------------------------
        # memory_question is the ORIGINAL user question.
        # This prevents internal AI prompts from being saved
        # as conversation history.

        original_question = (
            memory_question
            or question
        )

        # ----------------------------------------------------
        # GET PREVIOUS TOPIC
        # ----------------------------------------------------

        previous_topic = (
            get_last_user_question()
        )

        # ----------------------------------------------------
        # BUILD RETRIEVAL QUERY
        # ----------------------------------------------------

        retrieval_query = build_retrieval_query(
            question,
            previous_topic
        )

        # ----------------------------------------------------
        # DEBUG
        # ----------------------------------------------------

        print(
            "\n========================================"
        )

        print(
            "RAG REQUEST"
        )

        print(
            "========================================"
        )

        print(
            "Question:",
            question
        )

        print(
            "Previous topic:",
            previous_topic
        )

        print(
            "Retrieval query:",
            retrieval_query
        )

        # ====================================================
        # STEP 1: TOOL CALLING
        # ====================================================

        print(
            "DEBUG 1: Asking agent to select tool..."
        )

        tool_instruction = f"""

Student request:
{question}

Previous topic:
{previous_topic or "None"}

Retrieval query:
{retrieval_query}

Use the search_study_material tool to retrieve
relevant information from the uploaded study material.

If the request is a follow-up such as:

"explain it simply"
"give an example"
"why?"
"how?"
"summarize it"

use the Previous topic when selecting the
search topic.

Do not switch to an unrelated topic.
"""

        tool_response = invoke_with_retry(
            tool_llm,
            tool_instruction,
            attempts=2
        )

        print(
            "DEBUG 2: Agent decision completed."
        )

        docs = []

        # ====================================================
        # STEP 2: EXECUTE TOOL
        # ====================================================

        if getattr(
            tool_response,
            "tool_calls",
            None
        ):

            for tool_call in (
                tool_response.tool_calls
            ):

                if (
                    tool_call["name"]
                    == "search_study_material"
                ):

                    tool_args = (
                        tool_call.get(
                            "args",
                            {}
                        )
                    )

                    tool_topic = (
                        tool_args.get(
                            "topic",
                            retrieval_query
                        )
                    )

                    # ------------------------------------------------
                    # FOLLOW-UP PROTECTION
                    # ------------------------------------------------

                    if (
                        previous_topic
                        and is_follow_up(question)
                    ):

                        generic_topics = [

                            "overview",
                            "general",
                            "general overview",
                            "topic",
                            "information",
                            "main concepts",
                            "main concept",

                        ]

                        if (
                            not tool_topic
                            or tool_topic.lower().strip()
                            in generic_topics
                        ):

                            tool_topic = (
                                previous_topic
                            )

                    print(
                        "DEBUG 3: Tool called: "
                        "search_study_material"
                    )

                    print(
                        "DEBUG 4: Tool topic:",
                        tool_topic
                    )

                    docs = retriever.invoke(
                        tool_topic
                    )

                    break

        # ====================================================
        # STEP 3: FALLBACK RETRIEVAL
        # ====================================================

        if not docs:

            print(
                "DEBUG 3B: Tool did not return "
                "documents."
            )

            print(
                "DEBUG 4B: Using direct "
                "ChromaDB retrieval..."
            )

            docs = retriever.invoke(
                retrieval_query
            )

        print(
            "DEBUG 5: Documents found:",
            len(docs)
        )

        # ====================================================
        # NO DOCUMENTS
        # ====================================================

        if not docs:

            answer = NO_ANSWER

            _history.append(
                f"User: {original_question}"
            )

            _history.append(
                f"Assistant: {answer}"
            )

            if len(_history) > 20:
                _history = _history[-20:]

            return {
                "answer": answer,
                "sources": [],
            }

        # ====================================================
        # STEP 4: BUILD CONTEXT
        # ====================================================

        context_parts = []

        for i, doc in enumerate(
            docs,
            1
        ):

            source = doc.metadata.get(
                "source",
                "Unknown document"
            )

            page = doc.metadata.get(
                "page",
                None
            )

            if page is not None:

                try:

                    page = (
                        int(page) + 1
                    )

                except Exception:
                    pass

            context_parts.append(

                f"[Document {i}]\n"

                f"Source: {source}\n"

                f"Page: {page}\n"

                f"Content:\n"
                f"{doc.page_content}"

            )

        context = (
            "\n\n".join(
                context_parts
            )
        )

        # ====================================================
        # CONVERSATION HISTORY
        # ====================================================

        history_text = (
            "\n".join(
                _history[-MAX_HISTORY:]
            )
        )

        # ====================================================
        # STEP 5: FINAL ANSWER GENERATION
        # ====================================================

        prompt = (
            ChatPromptTemplate.from_messages(
                [

                    (
                        "system",
                        SYSTEM_PROMPT
                    ),

                    (
                        "human",
                        "{input}"
                    ),

                ]
            )
        )

        final_chain = (
            prompt | llm
        )

        print(
            "DEBUG 6: Generating final answer..."
        )

        try:

            response = invoke_with_retry(

                final_chain,

                {
                    "input": question,

                    "history": history_text,

                    "context": context,

                },

                attempts=3,

            )

            answer = response.content

        except Exception as e:

            print(
                "DEBUG 6B: Final Gemini call failed."
            )

            print(
                "ERROR:",
                type(e).__name__,
                str(e)
            )

            return {

                "answer": (
                    "The study material was "
                    "retrieved successfully, but "
                    "Gemini is temporarily unavailable. "
                    "Please try again in a few seconds."
                ),

                "sources": extract_sources(
                    docs
                ),

            }

        # ====================================================
        # CLEAN ANSWER
        # ====================================================

        if isinstance(
            answer,
            list
        ):

            parts = []

            for item in answer:

                if isinstance(
                    item,
                    dict
                ):

                    if (
                        item.get("type")
                        == "text"
                    ):

                        parts.append(
                            item.get(
                                "text",
                                ""
                            )
                        )

                elif isinstance(
                    item,
                    str
                ):

                    parts.append(
                        item
                    )

            answer = (
                "\n".join(parts)
            )

        answer = str(
            answer
        ).strip()

        # ====================================================
        # SOURCES
        # ====================================================

        sources = extract_sources(
            docs
        )

        # ====================================================
        # SAVE ORIGINAL USER QUESTION
        # ====================================================

        _history.append(
            f"User: {original_question}"
        )

        _history.append(
            f"Assistant: {answer}"
        )

        if len(_history) > 20:

            _history = (
                _history[-20:]
            )

        print(
            "DEBUG 7: Answer generated successfully."
        )

        print(
            "========================================\n"
        )

        return {

            "answer": answer,

            "sources": sources,

        }

    except Exception as e:

        print(
            "\n========================================"
        )

        print(
            "RAG ERROR"
        )

        print(
            "========================================"
        )

        print(
            "ERROR TYPE:",
            type(e).__name__
        )

        print(
            "ERROR:",
            str(e)
        )

        print(
            "========================================\n"
        )

        return {

            "answer": (
                f"RAG ERROR: "
                f"{type(e).__name__}: "
                f"{str(e)}"
            ),

            "sources": [],

        }


# ============================================================
# AI AGENT
# ============================================================

def ask_agent(question: str):

    try:

        question_lower = (
            question.lower().strip()
        )

        # ====================================================
        # QUESTION GENERATION
        # ====================================================

        if any(

            phrase in question_lower

            for phrase in [

                "generate question",
                "generate questions",

                "create question",
                "create questions",

                "make question",
                "make questions",

                "practice question",
                "practice questions",

                "exam question",
                "exam questions",

                "quiz me",

            ]

        ):

            task = (
                "question_generation"
            )

            final_query = f"""

Student request:
{question}

Generate ONLY 5 useful study/exam questions
based on the uploaded study material.

Include a mixture of:

1. Definition questions
2. Conceptual questions
3. Comparison questions
4. Application-based questions

Do NOT answer the questions.

Use ONLY the uploaded study material.

"""

        # ====================================================
        # SUMMARIZATION
        # ====================================================

        elif any(

            phrase in question_lower

            for phrase in [

                "summarize",
                "summarise",
                "summary",

                "short notes",

                "key points",
                "important points",

            ]

        ):

            task = (
                "summarization"
            )

            final_query = f"""

Student request:
{question}

Give a concise exam-oriented summary.

Include:

- Important concepts
- Key points
- Short explanations

Keep the summary focused,
clear and student-friendly.

Use ONLY the uploaded study material.

"""

        # ====================================================
        # SIMPLE EXPLANATION
        # ====================================================

        elif any(

            phrase in question_lower

            for phrase in [

                "explain simply",

                "explain in simple words",

                "explain in simple language",

                "simple explanation",

                "simple words",

                "easy explanation",

                "easy words",

                "beginner",

                "explain like",

            ]

        ):

            task = (
                "simple_explanation"
            )

            previous = (
                get_last_user_question()
            )

            if previous:

                final_query = f"""

Previous student topic:
{previous}

Student follow-up request:
{question}

Explain the PREVIOUS topic in simple,
easy-to-understand language.

Do NOT switch to another topic.

Use ONLY the uploaded study material.

"""

            else:

                final_query = f"""

Student request:
{question}

Explain the requested topic in simple,
easy-to-understand language.

Use ONLY the uploaded study material.

"""

        # ====================================================
        # ADVANTAGES / DISADVANTAGES
        # ====================================================

        elif any(

            phrase in question_lower

            for phrase in [

                "advantages",
                "disadvantages",

                "benefits",
                "limitations",

                "pros and cons",

                "pros",
                "cons",

            ]

        ):

            task = (
                "advantages_disadvantages"
            )

            final_query = f"""

Student request:
{question}

Give the answer using clear bullet points.

Use ONLY information from
the uploaded study material.

"""

        # ====================================================
        # NORMAL QUESTION
        # ====================================================

        else:

            task = (
                "question_answering"
            )

            final_query = question

        # ====================================================
        # RUN RAG
        # ====================================================

        result = ask_question(
            final_query,
            memory_question=question
        )

        return {

            "answer": result.get(
                "answer",
                "No answer generated."
            ),

            "sources": result.get(
                "sources",
                []
            ),

            "agent": True,

            "task": task,

        }

    except Exception as e:

        print(
            "AGENT ERROR:",
            type(e).__name__,
            str(e)
        )

        return {

            "answer": (
                f"AGENT ERROR: "
                f"{type(e).__name__}: "
                f"{str(e)}"
            ),

            "sources": [],

            "agent": True,

            "task": "error",

        }


# ============================================================
# DIRECT INTERACTIVE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "========================================"
    )

    print(
        "Student Guide AI - Interactive RAG Test"
    )

    print(
        "========================================"
    )

    print(
        "\nType 'exit' to stop."
    )

    print(
        "You can test follow-up questions "
        "without restarting the program."
    )

    print()

    while True:

        question = input(
            "Enter your question: "
        ).strip()

        if question.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print(
                "\nExiting Student Guide AI..."
            )

            break

        if not question:

            print(
                "Please enter a question.\n"
            )

            continue

        print(
            "\nSearching documents..."
        )

        result = ask_agent(
            question
        )

        print(
            "\nANSWER:"
        )

        print(
            result.get(
                "answer",
                "No answer generated."
            )
        )

        print(
            "\nTASK:"
        )

        print(
            result.get(
                "task",
                "Unknown"
            )
        )

        print(
            "\nSOURCES:"
        )

        for source in result.get(
            "sources",
            []
        ):

            print(source)

        print(
            "\n========================================\n"
        )