from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from typing import List, Dict, Generator
from tqdm import tqdm

# =============================
# Embedding model (GPU)
# =============================
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda:0", },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 2048
        }
    )

# =============================
# Streaming chunk generator
# =============================
def embed_chunks_streaming(
    chunks,
    embeddings,
    batch_size=1024,
    max_words=2500
):
    """
    Version ultra-rapide :
    - Pas de tokenizer
    - Découpage par mots (beaucoup plus rapide)
    - Batch GPU large
    """

    print(f"⚡ Streaming embedding ultra-rapide...")

    buffer = []

    for c in chunks:
        text = c["page_content"]
        words = text.split()

        for i in range(0, len(words), max_words):
            sub_text = " ".join(words[i:i+max_words])

            buffer.append({
                "page_content": sub_text,
                "metadata": c["metadata"]
            })

            if len(buffer) >= batch_size:
                yield buffer
                buffer = []

    if buffer:
        yield buffer




# =============================
# GPU Embed + Injection vecteurs
# =============================
def gpu_embed_batch(batch: List[Dict], embeddings) -> List[Dict]:
    texts = [c["page_content"] for c in batch]
    vectors = embeddings.embed_documents(texts)

    for i in range(len(batch)):
        batch[i]["embedding"] = vectors[i]

    return batch
