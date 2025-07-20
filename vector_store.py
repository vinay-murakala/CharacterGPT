import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

load_dotenv()

VECTOR_STORE_PATH = "faiss_index"

@st.cache_resource
def get_vector_store():
    """
    Loads documents, splits them into chunks, creates embeddings,
    and stores them in a FAISS vector store.
    Saves the store locally for reuse.
    Uses @st.cache_resource to avoid reloading/recreating on each script run.
    """
    try:
        if os.path.exists(VECTOR_STORE_PATH):
            return FAISS.load_local(
                VECTOR_STORE_PATH,
                GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                allow_dangerous_deserialization=True
            )
        else:
            loader = DirectoryLoader('./documents', glob="*.txt")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            text_chunks = text_splitter.split_documents(documents)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
            vector_store.save_local(VECTOR_STORE_PATH)
            return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        raise

def invalidate_vector_store_cache():
    """
    Removes the FAISS index directory to force cache refresh on next load.
    """
    import shutil
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH) 