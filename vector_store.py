from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "chroma_db"


# ============================================================
# EMBEDDING MODEL
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# ============================================================
# TEXT SPLITTER
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# ============================================================
# CREATE FILE HASH
# ============================================================

def get_file_hash(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    return hashlib.md5(file_bytes).hexdigest()


# ============================================================
# PROCESS UPLOADED PDF
# ============================================================

def process_uploaded_pdf(uploaded_file):

    temp_path = None

    try:

        # ----------------------------------------------------
        # Validate file
        # ----------------------------------------------------

        if uploaded_file is None:

            return {
                "success": False,
                "error": "No file was uploaded."
            }

        if not uploaded_file.name.lower().endswith(".pdf"):

            return {
                "success": False,
                "error": "Only PDF files are supported."
            }


        # ----------------------------------------------------
        # Create file hash
        # ----------------------------------------------------

        file_hash = get_file_hash(uploaded_file)


        # ----------------------------------------------------
        # Connect to ChromaDB
        # ----------------------------------------------------

        vector_db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )


        # ----------------------------------------------------
        # CHECK FOR DUPLICATE PDF
        # ----------------------------------------------------

        existing = vector_db.get(
            where={
                "file_hash": file_hash
            },
            limit=1
        )

        if existing and existing.get("ids"):

            return {
                "success": True,
                "filename": uploaded_file.name,
                "pages": 0,
                "chunks": 0,
                "file_hash": file_hash,
                "message": "This PDF is already stored in ChromaDB."
            }


        # ----------------------------------------------------
        # Save temporary PDF
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name


        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        loader = PyPDFLoader(temp_path)

        documents = loader.load()


        if not documents:

            return {
                "success": False,
                "error": "The PDF does not contain any readable pages."
            }


        # ----------------------------------------------------
        # Add metadata
        # ----------------------------------------------------

        for document in documents:

            document.metadata["source"] = uploaded_file.name

            document.metadata["file_hash"] = file_hash


        # ----------------------------------------------------
        # Split into chunks
        # ----------------------------------------------------

        docs = splitter.split_documents(
            documents
        )


        if not docs:

            return {
                "success": False,
                "error": "No readable text was found in the PDF."
            }


        # ----------------------------------------------------
        # CREATE UNIQUE IDS
        # ----------------------------------------------------

        ids = [
            f"{file_hash}_{i}"
            for i in range(len(docs))
        ]


        # ----------------------------------------------------
        # ADD DOCUMENTS TO CHROMADB
        # ----------------------------------------------------

        vector_db.add_documents(
            documents=docs,
            ids=ids
        )


        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return {
            "success": True,
            "filename": uploaded_file.name,
            "pages": len(documents),
            "chunks": len(docs),
            "file_hash": file_hash,
            "message": "PDF successfully added to ChromaDB."
        }


    except Exception as e:

        return {
            "success": False,
            "error": f"Unable to process PDF: {str(e)}"
        }


    finally:

        # ----------------------------------------------------
        # Delete temporary file
        # ----------------------------------------------------

        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)

            except Exception:
                pass