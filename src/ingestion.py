import os
import hashlib
from dotenv import load_dotenv

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)

from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# --- CONFIGURATION ---
COLLECTION_NAME = "finance_10k"
QDRANT_PATH = "./qdrant_local_data" 
DATA_DIR = "./data"

# 1. Setup Embeddings (Local & Free)
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

def get_file_hash(filepath: str) -> str:
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def ingest_documents():
    # 1. Cleanup Old Data (Optional but recommended for a fresh start)
    if os.path.exists(QDRANT_PATH):
        print(f"🧹 Clearing old database at {QDRANT_PATH}...")
        import shutil
        shutil.rmtree(QDRANT_PATH)

    print(f"📂 Connecting to local Qdrant at {QDRANT_PATH}...")
    client = QdrantClient(path=QDRANT_PATH)
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory '{DATA_DIR}' not found.")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    
    if not files:
        print(f"⚠️ No PDF files found in {DATA_DIR}.")
        return

    for file_name in files:
        file_path = os.path.join(DATA_DIR, file_name)
        file_hash = get_file_hash(file_path)
        
        print(f"🔍 Processing {file_name}...")
        
        # 3. LlamaParse
        print(f"⏳ Parsing PDF with LlamaCloud... (This sends data to API)")
        parser = LlamaParse(
            result_type="markdown",
            verbose=True,
            language="en"
        )
        
        documents = parser.load_data(file_path)
        
        # 4. Enrich Metadata
        try:
            ticker = file_name.split("_")[0]
            year = file_name.split("_")[1]
        except:
            ticker, year = "UNKNOWN", "UNKNOWN"
            
        for doc in documents:
            doc.metadata["file_hash"] = file_hash
            doc.metadata["ticker"] = ticker
            doc.metadata["year"] = year
            doc.metadata["file_name"] = file_name

        # 5. Chunking (Standard Markdown Parser)
        # This one is reliable and doesn't need OpenAI
        print("⚙️  Chunking text...")
        node_parser = MarkdownNodeParser()
        nodes = node_parser.get_nodes_from_documents(documents)
        
        # 6. Indexing
        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        print(f"🚀 Upserting {len(nodes)} chunks to local storage...")
        
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )
        
        print(f"✅ Finished ingesting {file_name}")

if __name__ == "__main__":
    ingest_documents()