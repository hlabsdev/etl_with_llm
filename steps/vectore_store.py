from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
from typing import Dict, List, Any
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
from typing import Dict, List

from multiprocessing import Pool, cpu_count



import os
import uuid
from datetime import datetime
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
from typing import Dict, List
from multiprocessing import Pool, cpu_count

from steps.embedder import embed_chunks_streaming, gpu_embed_batch

def generate_collection_name(file_path: str) -> str:
    # Nom du fichier sans extension
    base = os.path.splitext(os.path.basename(file_path))[0]

    # Timestamp formaté
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ID court
    uid = uuid.uuid4().hex[:6]

    # Format final
    return f"{base}_{timestamp}_{uid}"


# =============================
# CPU nettoyage (global)
# =============================
def cpu_clean(doc):
    try:
        text = str(doc["page_content"]).replace("\n", " ").strip()
        return {
            "page_content": text,
            "metadata": doc["metadata"]
        }
    except Exception as e:
        print(f"❌ Erreur document : {e}")
        return None


# =============================
# Streaming insertion Qdrant
# =============================
def insert_qdrant_streaming(store, batches, embeddings):
    total = 0
    for batch in batches:
        # Embedding GPU du batch
        embedded = gpu_embed_batch(batch, embeddings)

        # Transformation en LangChain documents
        docs = [
            Document(page_content=c["page_content"], metadata=c["metadata"])
            for c in embedded
        ]

        # Insertion dans Qdrant
        store.add_documents(docs)

        total += len(docs)
        print(f"✅ {total} documents indexés (streaming)")

# =============================
# Pipeline principal
# =============================
def create_vector_store(
    split_docs: List[Dict],
    embeddings,
    collection_name: str = "excel_documents"
):
    print("\n⚡ Pipeline streaming CPU → GPU → Qdrant")

    client = QdrantClient(host="localhost", port=6333)

    # Création collection
    vector_size = len(embeddings.embed_query("test"))
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"✅ Collection '{collection_name}' créée")
    except:
        print(f"ℹ️ Collection '{collection_name}' existe déjà")

    # Vector store
    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    # 1) CPU multiprocessing
    print("🧵 Nettoyage CPU multiprocessing...")
    with Pool(processes=max(2, cpu_count() // 2)) as pool:
        prepared = pool.map(cpu_clean, split_docs)
    prepared = [d for d in prepared if d is not None]
    print(f"✅ {len(prepared)} documents prêts")

    # 2) Streaming GPU + Qdrant
    print("🚀 Démarrage streaming GPU → Qdrant")
    batches = embed_chunks_streaming(
        prepared,
        embeddings,
        batch_size=256,
     
    )

    insert_qdrant_streaming(store, batches, embeddings)

    print("✅ Pipeline terminé avec succès")
    return store



