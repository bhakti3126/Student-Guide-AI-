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
                
🧠 RAG Pipeline

The application follows these steps:

Student uploads PDF study material.
PDF text is extracted using PyPDF.
Documents are divided into smaller chunks.
Gemini Embedding 001 converts chunks into vector embeddings.
Embeddings are stored in ChromaDB.
Student asks a question.
The AI Agent identifies the student's request.
The Agent can call the search_study_material tool.
Relevant document chunks are retrieved using similarity search.
Retrieved content is passed to the Groq LLM.
Groq GPT-OSS 20B generates a grounded response.
Relevant source files and page numbers are displayed.
🤖 AI Agent Capabilities

The Agent supports multiple study tasks:

1. Question Answering

Answers questions using the uploaded study material.

2. Simple Explanation

Explains difficult topics in simple, student-friendly language.

3. Summarization

Creates concise revision notes from the uploaded material.

4. Question Generation

Generates study and exam-oriented questions, including:

Definition questions
Conceptual questions
Comparison questions
Application-based questions
5. Advantages / Disadvantages

Provides advantages, disadvantages, benefits, or limitations when supported by the uploaded material.

🔧 Custom Tool

The project includes a custom LangChain tool:

search_study_material(topic)

The tool searches the uploaded academic material and returns relevant document content along with source and page information.

The AI Agent uses this tool when academic information needs to be retrieved from the uploaded documents.

🛡️ Hallucination Reduction

Student Guide AI is designed to reduce unsupported answers by:

Retrieving information from uploaded documents before generation
Using similarity search
Restricting responses to retrieved study material
Using a system prompt that prevents unsupported outside knowledge
Displaying source documents and page numbers
Returning:

I couldn't find the answer in the uploaded documents.

when relevant information is not available.

🧰 Technology Stack
Technology	Purpose
Python	Core programming language
Streamlit	Interactive web interface
LangChain	RAG and Agent framework
Groq	LLM inference
GPT-OSS 20B	AI generation and tool calling
Google Gemini Embedding 001	Text embeddings
ChromaDB	Vector database
PyPDF	PDF processing
python-dotenv	Environment variable management
FastAPI	API backend
Uvicorn	API server

📁 Project Structure
StudentGuide/
│
├── app.py
├── rag.py
├── vector_store.py
├── api.py
├── requirements.txt
├── README.md
│
├── data/
│   └── study_material.pdf
│
└── chroma_db/
File Description

app.py

Streamlit application
PDF upload
Chat interface
Quick study actions
Source display

rag.py

RAG pipeline
ChromaDB retrieval
AI Agent
Tool calling
Conversation handling
Answer generation

vector_store.py

PDF processing
Text chunking
Embedding generation
ChromaDB storage

api.py

FastAPI backend
/ask endpoint
API testing through Swagger

requirements.txt

Project dependencies
🔐 Environment Variables

The application requires API keys.

Create a .env file locally:

GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key


▶️ Run Locally
1. Clone the repository
git clone <your-github-repository-url>
cd StudentGuide
2. Create virtual environment
python -m venv venv
3. Activate the environment

Windows:

venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
5. Add API keys

Create .env:

GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
6. Run Streamlit
streamlit run app.py

The application will open in the browser.

🌐 API

The project also contains a FastAPI backend.

Run:

uvicorn api:app --reload

API documentation is available through Swagger UI:

http://127.0.0.1:8000/docs
💡 Example Usage
Upload

Upload:

OS Notes.pdf
Ask
What is process management?

The Agent retrieves relevant information from the uploaded document and generates the answer.

Follow-up
Explain it simply.

The Agent uses the previous topic to understand the follow-up question.

Summarization
Summarize process management.
Question Generation
Generate exam questions about process management.
🎯 Business Use Case
Problem

Students often have large amounts of academic material in PDF format. Finding important concepts, understanding difficult topics, creating revision notes, and preparing exam questions can be time-consuming.

Solution

Student Guide AI converts static academic PDFs into an interactive AI study assistant.

Target Users
College students
Engineering students
University students
Teachers and educational institutions
Value Proposition

Student Guide AI provides:

Faster study and revision
Personalized explanations
Document-based answers
Exam preparation support
Source-based responses
Interactive learning
⭐ Why Student Guide AI?

Unlike a general-purpose chatbot, Student Guide AI focuses on the student's uploaded academic material.

The RAG architecture retrieves relevant information from the student's documents before generating the response, helping keep answers grounded in the provided study material.

🔮 Future Scope

Possible future improvements include:

🃏 AI flashcards
📝 Interactive quizzes
📊 Student progress tracking
📅 AI study planner
🎤 Voice-based interaction
🌐 Multilingual support
📈 Personalized learning analytics
📱 Mobile application
👩‍💻 Project

Student Guide AI — Subject Guide & Question Bank Assistant AI Agent

Developed using:

RAG + AI Agent + Tool Calling + LangChain + ChromaDB + Groq + Streamlit

📌 Project Status

✅ RAG Pipeline Working
✅ Multiple PDF Support
✅ ChromaDB Retrieval
✅ AI Agent Working
✅ Custom Tool Calling Working
✅ Follow-up Questions Working
✅ Summarization Working
✅ Question Generation Working
✅ Source/Page Display Working
✅ Streamlit UI Working
✅ FastAPI Endpoint Working
🚀 Cloud Deployment — Final Step