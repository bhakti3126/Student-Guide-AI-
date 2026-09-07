# 🎓 Student Guide AI

An AI-powered academic study assistant that helps students understand, revise, summarize, and generate questions from their uploaded study material.

Student Guide AI uses **Retrieval-Augmented Generation (RAG)** and an **AI Agent with tool calling** to provide grounded answers from uploaded PDF documents.

---

## 🚀 Features

- 📄 Upload one or multiple PDF study materials
- 🔍 RAG-based document retrieval
- 🤖 AI Agent with tool calling
- 💬 Natural-language question answering
- 💡 Simple student-friendly explanations
- 📝 Summarization for exam revision
- ❓ Exam-oriented question generation
- ⚖️ Advantages and disadvantages
- 🔄 Conversational follow-up questions
- 📚 Multiple-document retrieval
- 📖 Source and page display
- 🧠 Conversation memory
- 🎯 Quick study actions
- 🖥️ Interactive Streamlit interface
- ⚡ Fast AI responses using Groq

---

## 🏗️ System Architecture

```text
                 Student
                    │
                    ▼
             Streamlit UI
                    │
                    ▼
              AI Agent
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
   Task Identification    Tool Calling
          │                   │
          └─────────┬─────────┘
                    ▼
          search_study_material
                    │
                    ▼
              ChromaDB
                    │
                    ▼
          Similarity Retrieval
                    │
                    ▼
          Relevant PDF Chunks
                    │
                    ▼
             Groq LLM
          GPT-OSS 20B
                    │
                    ▼
          Grounded Answer
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
       Answer             Sources
                       (File + Page)