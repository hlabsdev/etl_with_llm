
# =========================================================
# extractor.py
# Extractor robuste multi-chunks avec LLM direct
# =========================================================

import re
import json
import time
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from multiprocessing import Process, Queue, cpu_count

# ==========================
# Context Engineering
# ==========================
def build_context(docs: List[Document], token_limit: int = 6000) -> str:
    """Déduplication + nettoyage + limite sur le contexte"""
    seen = set()
    blocks = []
    total_len = 0

    for d in docs:
        content = d.page_content.strip()
        if not content:
            continue
        hashed = hash(content[:200])
        if hashed in seen:
            continue
        seen.add(hashed)
        block = f"\n--- SOURCE ---\n{content}\n"
        if total_len + len(block) > token_limit:
            break
        blocks.append(block)
        total_len += len(block)
    return "\n".join(blocks)

# ==========================
# Reranker performant
# ==========================
class AdvancedReranker:
    def __init__(self, method: str = "local"):
        self.method = method

    def score(self, query: str, text: str) -> float:
        q_tokens = set(query.lower().split())
        t_tokens = set(text.lower().split())
        if not q_tokens:
            return 0.0
        return len(q_tokens.intersection(t_tokens)) / len(q_tokens)

    def rerank(self, query: str, docs: List[Document], top_k: int = 30) -> List[Document]:
        scored = [(self.score(query, d.page_content[:2000]), d) for d in docs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

# ==========================
# Extraction JSON avec LLM direct
# ==========================


def extract_json_from_context(context: str, batch_num: int = 1) -> List[Dict]:
    """Extrait toutes les informations structurées depuis le contexte de manière universelle"""
    
    

    prompt = f"""
Tu es un moteur d'extraction de données. 
Ta seule tâche est d'extraire TOUTES les informations quantitatives OU factuelles présentes explicitement dans le texte suivant.

Règles STRICTES :
- N'invente rien.
- Ne déduis rien.
- Ne reformule rien.
- N'inclus AUCUN champ absent du texte.
- Tu ne fais AUCUNE classification.
- Tu ne fais AUCUNE interprétation.
- Tu produis UNIQUEMENT du JSON valide.

Format de sortie OBLIGATOIRE :

{{
  "donnees": [
      {{
        "indicateur": "...",
        "valeur": ...,
        "unite": "...",
        "pays": "...",
        "region": "...",
        "annee": "...",
        "source_document": "..."
        "code_iso":"...",
        "pourcentage": "...",
        "population_milliers": "...",
      "francophones_milliers": "...",
      }}
  ]
}}

Champs optionnels :
- Si le champ n'existe pas → NE PAS l'inclure.

Règles numériques :
- Si un nombre est explicitement écrit en milliers ou millions → convertis-le.
- Sinon garde-le tel quel.

Voici le texte à analyser :

{context}

Ta réponse doit contenir EXCLUSIVEMENT un JSON.
"""



    try:
        llm = ChatOllama(model="llama3.2", temperature=0)
        response = llm.invoke(prompt)
        content = response.content.strip()

        
            # Extraire le JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                print(f"⚠️  Pas de JSON trouvé dans la réponse du lot {batch_num}")
                return []
        
        json_str = json_str.strip()
        data = json.loads(json_str)
        
        if "donnees" in data and isinstance(data["donnees"], list):
            return data["donnees"]
        else:
            print(f"⚠️  Clé 'donnees' manquante ou invalide dans le JSON du lot {batch_num}")
            return []
        

    except json.JSONDecodeError as e:
        print(f"❌ Erreur de parsing JSON pour le lot {batch_num}: {e}")
        print(f"Contenu reçu: {content[:500]}")
        return []
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction pour le lot {batch_num}: {e}")
        return []


# ==========================
# Extraction par lot
# ==========================
def extract_with_agent(
    docs: List[Document],
    query: str,
    batch_size: int = 50,        
    token_limit: int = 100000,      
    rerank_top_k: int = 100000       
) -> List[Dict]:
    """
    Extraction des données par lots
    """
    print(f"\n🔄 Extraction par lots : {len(docs)} chunks")
    
    all_results = []
    total_batches = len(docs) // batch_size + (1 if len(docs) % batch_size else 0)
    
    reranker = AdvancedReranker(method="local")
    
    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end = start + batch_size
        batch_docs = docs[start:end]
        
        print(f"\n📦 Lot {batch_num+1}/{total_batches} (docs {start+1} → {min(end, len(docs))})")
        
        # Rerank pour filtrer les meilleurs chunks
        batch_docs = reranker.rerank(query, batch_docs, top_k=rerank_top_k)
        
        # Construire le contexte
        context = build_context(batch_docs, token_limit=token_limit)
        
        if not context.strip():
            print(f"⚠️  Contexte vide pour le lot {batch_num+1}")
            continue
        
        print(f"   Contexte: {len(context)} caractères, ~{len(context.split())} mots")
        
        # Extraction avec LLM
        batch_results = extract_json_from_context(context, batch_num + 1)
        
        if batch_results:
            all_results.extend(batch_results)
            print(f"   ✅ {len(batch_results)} entrées extraites")
        else:
            print(f"   ⚠️  Aucune donnée extraite de ce lot")
        
        # Pause pour éviter de surcharger le LLM
        time.sleep(0.5)
    
    print(f"\n✅ Extraction terminée : {len(all_results)} entrées récupérées")
    
    # Déduplication
    unique_results = []
    seen = set()
    for result in all_results:
        key = f"{result.get('pays', '')}-{result.get('annee', '')}"
        if key not in seen:
            seen.add(key)
            unique_results.append(result)
    
    print(f"📊 Après déduplication : {len(unique_results)} entrées uniques")
    return unique_results







def worker_extract(queue_in: Queue, queue_out: Queue, query: str, token_limit: int):
    """
    Worker multiprocessing :
    - lit des batches depuis queue_in
    - extrait via LLM
    - renvoie les résultats dans queue_out
    """
    while True:
        item = queue_in.get()

        if item is None:
            break

        batch_num, docs = item

        try:
            context = build_context(docs, token_limit=token_limit)
            results = extract_json_from_context(context, batch_num)

            queue_out.put(results)

        except Exception as e:
            print(f"❌ Worker erreur batch {batch_num}: {e}")
            queue_out.put([])


def extract_with_agent_parallel(
    docs: List[Document],
    query: str,
    batch_size: int = 50,
    token_limit: int = 60000,
    num_workers: int = None
) -> List[Dict]:

    print(f"\n⚡ Extraction parallèle activée")

    if num_workers is None:
        num_workers = max(2, cpu_count() - 1)

    queue_in = Queue()
    queue_out = Queue()

    # Démarrer les workers
    workers = []
    for _ in range(num_workers):
        p = Process(
            target=worker_extract,
            args=(queue_in, queue_out, query, token_limit)
        )
        p.start()
        workers.append(p)

    # Découpage en lots
    total_batches = len(docs) // batch_size + (1 if len(docs) % batch_size else 0)

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end = start + batch_size
        batch_docs = docs[start:end]

        queue_in.put((batch_num + 1, batch_docs))

    # Stop signals
    for _ in workers:
        queue_in.put(None)

    # Collecte résultats
    results = []
    for _ in range(total_batches):
        results.extend(queue_out.get())

    # Join
    for p in workers:
        p.join()

    print(f"✅ Extraction parallèle terminée : {len(results)} entrées")

    return results



