import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

API_KEY = os.getenv("GOOGLE_API_KEY")
PROMPT_DIR = "prompts"


def get_prompt_templates():
    """
    Loads persona prompt templates from the prompts directory.
    Returns a dict of persona name to PromptTemplate.
    """
    templates = {}
    try:
        for fname in os.listdir(PROMPT_DIR):
            if fname.endswith(".txt"):
                persona = os.path.splitext(fname)[0].replace('_', ' ').title()
                with open(os.path.join(PROMPT_DIR, fname), encoding="utf-8") as f:
                    template = f.read()
                templates[persona] = PromptTemplate.from_template(template)
    except Exception as e:
        st.error(f"Error loading prompt templates: {e}")
        raise
    return templates

def get_conversation_chain(_vector_store, persona):
    """
    Creates a conversational RAG chain with memory and a custom persona prompt.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7, google_api_key=API_KEY)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=_vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={
                "prompt": st.session_state.prompt_templates[persona]
            }
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        raise 