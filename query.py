import os
import pickle
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_mistralai import ChatMistralAI
from sentence_transformers import CrossEncoder

load_dotenv()

# Setup retrievers
print("Loading embeddings and vector store...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

print("Loading BM25 corpus...")
with open("bm25_corpus.pkl", "rb") as f:
    bm25_docs = pickle.load(f)
sparse_retriever = BM25Retriever.from_documents(bm25_docs)
sparse_retriever.k = 5

print("Loading Reranker...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def hybrid_retrieval_and_rerank(query: str):
    # Retrieve documents
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = sparse_retriever.invoke(query)
    
    # Merge and deduplicate by content
    all_docs = {}
    for doc in dense_docs + sparse_docs:
        all_docs[doc.page_content] = doc
        
    unique_docs = list(all_docs.values())
    
    if not unique_docs:
        return []
        
    # Prepare pairs for reranking
    pairs = [[query, doc.page_content] for doc in unique_docs]
    scores = reranker.predict(pairs)
    
    def get_score(item):
        return item[1]
        
    scored_docs = []
    for i in range(len(unique_docs)):
        doc = unique_docs[i]
        score = scores[i]
        scored_docs.append((doc, score))
        
    scored_docs.sort(key=get_score, reverse=True)
    return [doc for doc, score in scored_docs[:5]]

def get_answer(question: str):
    best_docs = hybrid_retrieval_and_rerank(question)
    context = "\n\n".join([doc.page_content for doc in best_docs])
    
    template = f"""Answer the question based only on the following context:
{context}

Question: {question}
"""
    
    llm = ChatMistralAI(model="mistral-large-latest", mistral_api_key=os.getenv("MISTRAL_API_KEY"))
    response = llm.invoke(template)
    
    return response.content

while True:
    try:
        question = input("\nQuestion: ")
        if not question.strip():
            continue
        if question.strip().lower() in ['quit', 'exit', 'q']:
            break
            
        print("Thinking...")
        answer = get_answer(question)
        print("\nAnswer:", answer)
    except EOFError:
        break
