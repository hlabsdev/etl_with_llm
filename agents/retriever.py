# =========================================================
# retriever.py
# Retriever universel : Vector Search + Similarity + Rerank + Context Engineering
# =========================================================

from typing import List
from langchain_core.documents import Document
import numpy as np

# ============================
# Simple Reranker local
# ============================

class Reranker:
    """
    Reranker simple basé sur matching score local.
    Peut être remplacé par Cohere / CrossEncoder / BGE-Reranker.
    """

    def _score(self, query: str, text: str) -> float:
        query_tokens = set(query.lower().split())
        text_tokens = set(text.lower().split())

        if not query_tokens:
            return 0.0

        common = query_tokens.intersection(text_tokens)
        return len(common) / len(query_tokens)

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int = 30
    ) -> List[Document]:

        scored = []

        for d in docs:
            score = self._score(query, d.page_content[:2000])
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)

        reranked = [doc for _, doc in scored[:top_k]]

        return reranked


# ============================
# Context Engineering Tool
# ============================

def build_context(docs: List[Document], token_limit: int = 6000) -> str:
    """
    Construit un contexte propre pour le LLM :
    - Déduplication
    - Nettoyage
    - Respect d’une limite de taille
    """

    seen = set()
    context_blocks = []
    total_len = 0

    for d in docs:
        content = d.page_content.strip()

        if not content:
            continue

        # déduplication simple
        hashed = hash(content[:200])
        if hashed in seen:
            continue
        seen.add(hashed)

        block = f"\n--- SOURCE ---\n{content}\n"
        block_len = len(block)

        if total_len + block_len > token_limit:
            break

        context_blocks.append(block)
        total_len += block_len

    return "\n".join(context_blocks)


# ============================
# Retriever Universel
# ============================

def retrieve_docs(
    store,
    query: str,
    k: int = 5000,          # ⬅️ Augmenter par défaut
    method: str = "mmr",
    rerank_top_k: int = 100  # ⬅️ Augmenter
) -> List[Document]:
    
    print(f"\n🔍 [Retriever] Query : {query[:80]}...")
    print(f"📈 Paramètres: k={k}, rerank_top_k={rerank_top_k}")

    # MODIFIER cette partie pour augmenter les limites
    try:
        retriever = store.as_retriever(
            search_type=method,
            search_kwargs={
                "k": min(k, 200000),        # ⬅️ Augmenter à 2000
                "fetch_k": min(k * 3, 600000),  # ⬅️ Augmenter
                "lambda_mult": 0.5        # ⬅️ Plus de diversité
            }
        )
        
        docs = retriever.invoke(query)
        print(f"✅ VectorSearch(MMR) : {len(docs)} chunks récupérés")
        
    except Exception as e:
        print(f"⚠️ MMR indisponible : {e}")
        # Fallback direct
        docs = store.similarity_search(query=query, k=min(k, 1000))
    

    # -------------------------------------------------
    # 2. Fallback Similarity Search
    # -------------------------------------------------
    if not docs:
        try:
            print("🔁 Fallback : similarity_search")
            docs = store.similarity_search(query=query, k=k)
            print(f"✅ similarity_search : {len(docs)} chunks")

        except Exception as e:
            print(f"❌ similarity_search échoué : {e}")
            return []

    # -------------------------------------------------
    # 3. Reranking
    # -------------------------------------------------
    if docs:
        print("🔄 Reranking...")

        reranker = Reranker()
        docs = reranker.rerank(query, docs, top_k=rerank_top_k)

        print(f"🎯 Après rerank : {len(docs)} chunks")

    # -------------------------------------------------
    # 4. Context Engineering
    # -------------------------------------------------
    print("🧠 Construction du contexte...")

    context = build_context(docs)

    # -------------------------------------------------
    # 5. Debug / Metrics Tools
    # -------------------------------------------------
    total_lines = 0
    country_lines = 0

    for d in docs[:10]:
        lines = d.page_content.split("\n")
        total_lines += len(lines)
        country_lines += sum(1 for l in lines if "Pays:" in l)

    print(f"📊 Lignes estimées : {total_lines}")
    print(f"🌍 Lignes 'Pays:' : {country_lines}")
    print(f"📦 Taille du contexte : {len(context)} caractères")

    # Injection du contexte dans les métadonnées
    for d in docs:
        d.metadata["context_ready"] = True

    return docs
from langchain_core.documents import Document
from qdrant_client.models import PointStruct
def dump_all_documents(vector_store, collection_name, sample_size=1000):
    """
    Récupère un petit échantillon de documents pour test.
    Version simple et séquentielle.
    """
    all_docs = []
    batch_size = min(500, sample_size)  # Adapter la taille du batch
    offset = 0
    collection_name = collection_name
    client = vector_store.client
    
    print(f"🔬 Récupération d'un échantillon de {sample_size} documents...")
    
    while len(all_docs) < sample_size:
        try:
            result = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if isinstance(result, tuple):
                points, next_offset = result
            else:
                points = result
                next_offset = None
            
            if not points:
                print("⚠️ Aucun point trouvé dans la collection")
                break
            
            for point in points:
                # Extraire le payload
                if hasattr(point, 'payload'):
                    payload = point.payload
                elif isinstance(point, dict):
                    payload = point.get('payload', point)
                else:
                    payload = {}
                
                # Extraire contenu
                if isinstance(payload, dict):
                    content = payload.get("page_content", "")
                    if not content:  # Essayer 'text' si 'page_content' est vide
                        content = payload.get("text", "")
                    
                    metadata = {k: v for k, v in payload.items() 
                              if k not in ["page_content", "text"]}
                else:
                    content = str(payload)
                    metadata = {}
                
                # Filtrer les documents trop courts ou vides
                if len(content.strip()) > 50:  # Ignorer docs de moins de 50 caractères
                    all_docs.append(Document(
                        page_content=content.strip(), 
                        metadata=metadata
                    ))
                
                if len(all_docs) >= sample_size:
                    break
            
            print(f"📥 {len(all_docs)}/{sample_size} documents récupérés...")
            
            if next_offset is None or offset == next_offset:
                break
                
            offset = next_offset
            
        except Exception as e:
            print(f"⚠️ Erreur lors de la récupération: {e}")
            break
    
    print(f"\n✅ Échantillon récupéré : {len(all_docs)} documents")
    
    # Afficher un aperçu des documents
    if all_docs:
        print("\n🔍 Aperçu des 3 premiers documents:")
        for i, doc in enumerate(all_docs[:3]):
            print(f"\n--- Document {i+1} ---")
            print(f"Contenu (100 premiers caractères): {doc.page_content[:100]}...")
            print(f"Métadonnées: {doc.metadata}")
            print(f"Longueur: {len(doc.page_content)} caractères")
    
    return all_docs