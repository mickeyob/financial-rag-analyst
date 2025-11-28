import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from qdrant_client import QdrantClient

# Load env
load_dotenv()

# --- CONFIGURATION ---
# Must match your ingestion script exactly
QDRANT_PATH = "./qdrant_local_data" 
COLLECTION_NAME = "finance_10k"

# Use the same embedding model
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

def test_retrieval():
    print(f"📂 Connecting to local Qdrant at {QDRANT_PATH}...")
    client = QdrantClient(path=QDRANT_PATH)
    
    # 1. Re-connect to the existing index
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # 2. Create a Query Engine
    # similarity_top_k=5 means "Get me the best 5 chunks"
    query_engine = index.as_query_engine(similarity_top_k=5)

    # 3. The "Golden" Question
    # This is a specific fact hidden in a table in the 10-K
    question = "What were the Net Sales for 2023?"
    
    print(f"\n❓ Question: {question}")
    print("Searching...")
    
    response = query_engine.query(question)

    # 4. The "Data Engineer" Audit
    # We don't just print the answer; we inspect the SOURCE.
    print(f"\n🤖 LLM Answer: {str(response)}\n")
    print("--- 🔍 RETRIEVAL EVIDENCE (The Proof) ---")
    
    for i, node in enumerate(response.source_nodes):
        score = node.score
        page_num = node.node.metadata.get('page_label', 'N/A') # LlamaParse usually adds this
        content = node.node.get_content()[:200] # Show first 200 chars
        
        print(f"Chunk {i+1} (Score: {score:.4f} | Page: {page_num})")
        print(f"Content Preview: {content}...")
        print("-" * 50)

if __name__ == "__main__":
    test_retrieval()