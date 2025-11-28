import os
import chainlit as cl
from dotenv import load_dotenv

# --- 1. SETUP ENV & KEYS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("❌ ERROR: GROQ_API_KEY is missing!")
else:
    print("✅ Keys loaded.")

# --- 2. IMPORTS ---
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatMemoryBuffer
from qdrant_client import QdrantClient

# --- 3. CONFIGURATION ---
QDRANT_PATH = "./qdrant_local_data"
COLLECTION_NAME = "finance_10k"

# Load models once at startup
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

# --- 4. CHAT STARTUP (FAST) ---
@cl.on_chat_start
async def start():
    msg = cl.Message(content="Loading Financial Knowledge Base...")
    await msg.send()

    try:
        # Connect to Database
        client = QdrantClient(path=QDRANT_PATH)
        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # Create Logic Engine
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=ChatMemoryBuffer.from_defaults(token_limit=3000),
            system_prompt=(
                "You are a Senior Financial Analyst. "
                "You are analyzing the Apple (AAPL) 10-K filing for 2022. "
                "ALWAYS cite the specific page number or table row if you find it in the context. "
                "If the context does not contain the answer, say 'I cannot find that in the 2022 10-K'."
            )
        )
        cl.user_session.set("chat_engine", chat_engine)
        
        msg.content = "✅ **System Ready.** Ask me about Apple's 2022 Financials!"
        await msg.update()

    except Exception as e:
        msg.content = f"❌ **Startup Error:** {str(e)}"
        await msg.update()

# --- 5. MAIN CHAT LOOP ---
@cl.on_message
async def main(message: cl.Message):
    chat_engine = cl.user_session.get("chat_engine")
    msg = cl.Message(content="")
    
    try:
        response_stream = chat_engine.stream_chat(message.content)
        
        for token in response_stream.response_gen:
            await msg.stream_token(token)

        if response_stream.source_nodes:
            sources_text = "\n\n**Sources Used:**"
            for node in response_stream.source_nodes:
                page = node.metadata.get('page_label', 'N/A')
                file_name = node.metadata.get('file_name', 'Doc')
                sources_text += f"\n- {file_name} (Page {page})"
            
            await msg.stream_token(sources_text)

        await msg.send()
        
    except Exception as e:
        await cl.Message(content=f"❌ **Error:** {str(e)}").send()