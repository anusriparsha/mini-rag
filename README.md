# Mini Retrieval-Augmented Generation (RAG) System

This project is a *small-scale Retrieval-Augmented Generation (RAG) pipeline* built using:
- [LangChain](https://python.langchain.com/)
- [Ollama](https://ollama.ai/) (for embeddings + LLM)
- [FAISS](https://faiss.ai/) (vector database)

It loads a text file, splits it into chunks, embeds the chunks, stores them in FAISS, and allows the user to ask questions in natural language. The system retrieves the most relevant chunks and generates answers using an LLM.

---
 How It Works

1.Load text file
   The script loads Sample.txt from the project folder.

2. Chunking
   The text is split into smaller overlapping chunks using RecursiveCharacterTextSplitter.

3. Embeddings
   Each chunk is converted into embeddings using OllamaEmbeddings (model = llama3.2).

4. Vector Store (FAISS)
   The embeddings are stored in a FAISS index for similarity search.

5. Retrieval + QA
   A RetrievalQA chain uses:
   - FAISS retriever → fetches top-3 relevant chunks.  
   - OllamaLLM → generates the final answer.  

6. Interactive Q&A
   The user can type any question, and the system prints:
   - The generated answer.  
   - The source text snippets used.  

---

#Setup Instructions

# 1. Install dependencies
Make sure you have Python 3.9+ and a virtual environment. Then install:

2.pip install langchain langchain-community langchain-ollama faiss-cpu

Time spent : 2-4 hours as requested.

VEDIO LINK : https://1drv.ms/v/c/2347956e4f99a8a5/EfoPCiUOkP5EqsQ4XGAB_wIBq5XXofJkHiJDhM9896qR9A?e=Ggfevi
