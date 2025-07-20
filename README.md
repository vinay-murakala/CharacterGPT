# CharacterGPT 🤖 — RAG-Powered Persona Simulator

This project is a Retrieval-Augmented Generation (RAG) application that allows you to interact with the simulated personas of famous personalities like **APJ Abdul Kalam**, **Elon Musk**, and **Albert Einstein**.

The application uses a knowledge base created from texts about each individual. When you ask a question, the system retrieves relevant information from this knowledge base and uses Google's Gemini Pro model to generate an answer in the character's specific style and from their perspective.

## ✨ Features

- **Persona Simulation**: Chat with AI versions of historical and modern figures.
- **RAG Pipeline**: Combines the power of large language models (LLMs) with a specific knowledge base for grounded, context-aware answers.
- **Interactive UI**: A simple and clean web interface built with Streamlit.
- **Local Vector Store**: Uses FAISS to store document embeddings locally, ensuring fast and efficient retrieval without repeated processing.

## 🛠️ Tech Stack

- **LLM**: Google Gemini Pro
- **Framework**: LangChain
- **UI**: Streamlit
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Google Generative AI Embeddings

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

- Python 3.9+
- A Google API Key for Gemini. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 2. Clone the Repository

```bash
git clone [https://github.com/your-username/CharacterGPT.git](https://github.com/your-username/CharacterGPT.git)
cd CharacterGPT
```

### 3. Install Dependencies

Create a virtual environment (recommended) and install the required packages.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install packages
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory of the project and add your Google API key:

```
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

### 5. Run the Application

Once the setup is complete, run the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your web browser. The first time you run it, it will process the documents and create a `faiss_index` folder. Subsequent runs will be faster as they will load the pre-built index.

## ⚙️ How It Works

1.  **Data Loading**: Text files from the `documents/` directory are loaded.
2.  **Indexing**: The documents are split into smaller chunks. These chunks are then converted into numerical vectors (embeddings) using Google's embedding model and stored in a FAISS vector index. This index is saved locally in the `faiss_index` folder.
3.  **User Query**: You select a persona and ask a question through the Streamlit UI.
4.  **Retrieval**: The system searches the FAISS index to find the text chunks most relevant to your question.
5.  **Generation**: The user's question, the selected persona, and the retrieved text chunks are passed to the Gemini Pro model via a carefully crafted prompt.
6.  **Response**: The model generates a response that mimics the chosen persona, grounded in the retrieved context, and displays it on the screen.

## 🧩 Project Structure

- `app.py` — Streamlit UI logic only.
- `vector_store.py` — Handles loading, saving, and cache invalidation for the FAISS vector store.
- `persona.py` — Loads persona prompt templates and creates conversation chains.
- `prompts/` — Contains external persona prompt templates (one .txt file per persona).
- `documents/` — Source documents for each persona.
- `faiss_index/` — Local FAISS vector store (auto-generated).

## 🔒 Security Notes

- **API Keys:** Never commit your `.env` file or API keys to version control. The app loads your Google API key from `.env`.
- **Sensitive Data:** No sensitive data is logged or exposed in the UI.

## 🧰 Troubleshooting

- **Missing API Key:** If you see an error about the Google API key, ensure your `.env` file is present and correct.
- **Dependency Issues:** If you encounter errors during installation, ensure you are using the correct Python version and that all dependencies are installed with the pinned versions in `requirements.txt`.
- **Prompt Loading Errors:** If you add new personas, ensure you create a corresponding `.txt` file in `prompts/`.
- **Cache Issues:** If you update documents and want to refresh the FAISS index, delete the `faiss_index/` directory or use the `invalidate_vector_store_cache()` function in `vector_store.py`.

## 🏗️ Extending the App

- To add a new persona:
  1. Add a new `.txt` file in `documents/` with relevant content.
  2. Add a new prompt template in `prompts/` (e.g., `marie_curie.txt`).
  3. The app will automatically detect and load new personas at startup.

---
