import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def main():
    print("Loading PDF...")
    loader = PyPDFLoader("AWS.pdf")
    docs = loader.load()

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    print("Saving BM25 Corpus...")
    with open("bm25_corpus.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Generating embeddings and storing in Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
