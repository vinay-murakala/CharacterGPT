import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Configure the Google API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API Key not found. Please set it in your .env file.")
    st.stop()

# --- DATA LOADING AND PROCESSING ---

# Define the path for the local FAISS vector store
VECTOR_STORE_PATH = "faiss_index"

@st.cache_resource
def get_vector_store():
    """
    Loads documents, splits them into chunks, creates embeddings,
    and stores them in a FAISS vector store.
    Saves the store locally for reuse.
    Uses @st.cache_resource to avoid reloading/recreating on each script run.
    """
    if os.path.exists(VECTOR_STORE_PATH):
        # Load the existing vector store
        return FAISS.load_local(
            VECTOR_STORE_PATH, 
            GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
            allow_dangerous_deserialization=True
        )
    else:
        # Load documents from the 'documents' directory
        loader = DirectoryLoader('./documents', glob="*.txt")
        documents = loader.load()

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(documents)

        # Create embeddings and the FAISS vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        
        # Save the vector store locally
        vector_store.save_local(VECTOR_STORE_PATH)
        return vector_store

# --- RAG CHAIN WITH MEMORY ---

def get_conversation_chain(_vector_store, persona):
    """
    Creates a conversational RAG chain with memory and a custom persona prompt.
    """
    # Initialize the Gemini Pro model
    # llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, google_api_key=api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7, google_api_key=api_key)
    
    # Setup memory
    # The 'chat_history' memory key is important for the chain
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Create the conversational chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vector_store.as_retriever(),
        memory=memory,
        # Customizing the prompt to enforce the persona
        combine_docs_chain_kwargs={
            "prompt": st.session_state.prompt_templates[persona]
        }
    )
    return conversation_chain

# --- STREAMLIT UI ---
def main():
    """
    The main function to run the Streamlit web app.
    """
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    st.set_page_config(page_title="CharacterGPT 🤖", layout="wide")
    st.title("CharacterGPT — Chat with a Persona 🧠")
    st.markdown("---")

    # --- Persona Prompts and Selection ---
    # Define custom prompts for each persona
    # This is done once and stored in session state
    if "prompt_templates" not in st.session_state:
        from langchain.prompts import PromptTemplate
        kalam_template = """You are acting as the persona of APJ Abdul Kalam. Answer the user's question in a way that is consistent with his known ideas, beliefs, and humble, inspiring communication style. Use the provided context and chat history to ground your answer. Always maintain his persona.

        Chat History:
        {chat_history}
        
        Context:
        {context}

        User's Question: {question}

        Your Response (as APJ Abdul Kalam):"""

        musk_template = """You are acting as the persona of Elon Musk. Answer the user's question in a way that is consistent with his known ideas, direct communication style, and focus on first principles, engineering, and ambitious goals. Use the provided context and chat history to ground your answer. Always maintain his persona.

        Chat History:
        {chat_history}
        
        Context:
        {context}

        User's Question: {question}

        Your Response (as Elon Musk):"""

        einstein_template = """You are acting as the persona of Albert Einstein. Answer the user's question in a thoughtful, philosophical, and scientific manner consistent with his known ideas and personality. Reflect on the deeper implications. Use the provided context and chat history to ground your answer. Always maintain his persona.

        Chat History:
        {chat_history}
        
        Context:
        {context}

        User's Question: {question}

        Your Response (as Albert Einstein):"""
        
        st.session_state.prompt_templates = {
            "APJ Abdul Kalam": PromptTemplate.from_template(kalam_template),
            "Elon Musk": PromptTemplate.from_template(musk_template),
            "Albert Einstein": PromptTemplate.from_template(einstein_template),
        }

    # Persona selection
    persona_options = ["APJ Abdul Kalam", "Elon Musk", "Albert Einstein"]
    selected_persona = st.sidebar.selectbox("Choose a Persona:", persona_options, key="persona_select")

    # Initialize session state for chat history and conversation chain
    if "conversation" not in st.session_state or st.session_state.persona_select != st.session_state.get('last_persona'):
        vector_store = get_vector_store()
        st.session_state.conversation = get_conversation_chain(vector_store, selected_persona)
        st.session_state.chat_history = []
        st.session_state.last_persona = selected_persona

    st.sidebar.button("Clear Chat History", on_click=lambda: setattr(st.session_state, 'chat_history', []))
    
    # Display chat history
    st.header(f"Conversation with {selected_persona}")
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(message['question'])
        with st.chat_message("assistant"):
            st.write(message['answer'])

    # User input
    user_question = st.chat_input(f"Ask {selected_persona} a question...")

    if user_question:
        with st.spinner("Thinking..."):
            # The chain now manages history automatically
            response = st.session_state.conversation({'question': user_question})
            
            # Store question and answer in history
            st.session_state.chat_history.append({'question': user_question, 'answer': response['answer']})
            
            # Rerun to display the new message
            st.rerun() # Use st.rerun() which is the modern replacement for st.experimental_rerun()

if __name__ == "__main__":
    main()