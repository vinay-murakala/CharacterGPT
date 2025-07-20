import streamlit as st
from vector_store import get_vector_store
from persona import get_prompt_templates, get_conversation_chain
import os

# --- STREAMLIT UI ---
def main():
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    st.set_page_config(page_title="CharacterGPT 🤖", layout="wide")
    st.title("CharacterGPT — Chat with a Persona 🧠")
    st.markdown("---")

    # --- Persona Prompts and Selection ---
    if "prompt_templates" not in st.session_state:
        st.session_state.prompt_templates = get_prompt_templates()

    persona_options = list(st.session_state.prompt_templates.keys())
    selected_persona = st.sidebar.selectbox("Choose a Persona:", persona_options, key="persona_select")

    # Initialize session state for chat history and conversation chain
    if "conversation" not in st.session_state or st.session_state.persona_select != st.session_state.get('last_persona'):
        vector_store = get_vector_store()
        st.session_state.conversation = get_conversation_chain(vector_store, selected_persona)
        st.session_state.chat_history = []
        st.session_state.last_persona = selected_persona

    st.sidebar.button("Clear Chat History", on_click=lambda: setattr(st.session_state, 'chat_history', []))
    
    # Display chat history (limit to last 20 messages for performance)
    st.header(f"Conversation with {selected_persona}")
    for i, message in enumerate(st.session_state.chat_history[-20:]):
        with st.chat_message("user"):
            st.write(message['question'])
        with st.chat_message("assistant"):
            st.write(message['answer'])

    user_question = st.chat_input(f"Ask {selected_persona} a question...")

    if user_question:
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.conversation({'question': user_question})
                st.session_state.chat_history.append({'question': user_question, 'answer': response['answer']})
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()