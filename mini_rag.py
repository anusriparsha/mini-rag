
import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

DATA_FILE = "Sample.txt"

def build_rag_pipeline():
    # step 1: Load the text File
    here = Path(__file__).resolve().parent
    data_path = (here / DATA_FILE).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    loader = TextLoader(str(data_path), encoding="utf-8")
    documents = loader.load()

    #  Step 2: Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    #  Step 3: Use LLM for embeddings
    # NOTE: OllamaEmbeddings with "llama3.2" here
    embeddings = OllamaEmbeddings(model="llama3.2")

    #Step 4: Store vectors in FAISS
    vectordb = FAISS.from_documents(chunks, embeddings)

    #  Step 5: Setup same LLM for QA
    llm = OllamaLLM(model="llama3.2")

    # Step 6: Build Retrieval-QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        return_source_documents=True
    )

    return qa


def pretty_print(result):
    answer = result.get("result") or result
    print("\nAnswer:\n", answer)
    sources = result.get("source_documents") or []
    if sources:
        print("\nSources:")
        for i, doc in enumerate(sources, 1):
            snippet = doc.page_content.strip().replace("\n", " ")
            print(f"[{i}] {snippet[:180]}{'...' if len(snippet) > 180 else ''}")


if __name__ == "__main__":
    print("🔍 Mini RAG (LLM-only Embeddings + QA)")
    print("Type 'exit' to quit.\n")

    try:
        qa_pipeline = build_rag_pipeline()
    except Exception as e:
        print("Setup error:", e)
        raise SystemExit(1)

    while True:
        query = input("Ask a question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye! 👋")
            break
        if not query:
            continue

        result = qa_pipeline.invoke({"query": query})
        pretty_print(result)