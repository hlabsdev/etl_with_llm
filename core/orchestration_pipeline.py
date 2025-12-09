# =========================================================
# orchestration_pipeline.py
# Orchestration complète du pipeline Excel
# =========================================================

import os
import json
from typing import List, Dict
from langchain_core.documents import Document

from steps.embedder import get_embeddings
from steps.vectore_store import create_vector_store, generate_collection_name
from agents.retriever import retrieve_docs
from steps.splitter import split_excel_documents
from loader.parallel_excel_loader import load_excel_with_sheets_parallel as load_excel_with_sheets
from preprocess import preprocess_excel_data
from agents.extractor import extract_with_agent


def run_pipeline(
    excel_file: str,
    output_file: str,
    batch_size: int = 10,
    token_limit: int = 6000,
    rerank_top_k: int = 30
) -> List[Dict]:
    """
    Exécute le pipeline complet sur un fichier Excel et retourne les données extraites.
    """

    # --- 1. Collection Qdrant ---
    vector_collection = generate_collection_name(excel_file)
    print(f"📁 Collection Qdrant utilisée : {vector_collection}")

    # --- 2. Charger le fichier Excel ---
    sheets_data = load_excel_with_sheets(excel_file)
    if not sheets_data:
        raise ValueError("Erreur : Excel vide ou introuvable.")

    # --- 3. Prétraiter les données ---
    stats = preprocess_excel_data(sheets_data)
    print(f"\n📊 Statistiques prétraitement : {json.dumps(stats, indent=2)}")

    # --- 4. Conversion DataFrame -> Documents ---
    documents = []
    for sheet_name, df in sheets_data.items():
        for idx, row in df.iterrows():
            content = " | ".join(str(cell) for cell in row)
            documents.append(Document(
                page_content=content,
                metadata={"sheet_name": sheet_name, "row_index": idx}
            ))

    print(f"\n🔖 Total Documents générés : {len(documents)}")

    # --- 5. Découper en chunks ---
    chunks = split_excel_documents([{"page_content": d.page_content, "metadata": d.metadata} for d in documents])
    print(f"🔪 Total chunks après découpage : {len(chunks)}")

    # --- 6. Embeddings & Vector Store ---
    embeddings = get_embeddings()
    vector_store = create_vector_store(
        split_docs=chunks,
        embeddings=embeddings,
        collection_name=vector_collection,
    )

    # --- 7. Retrieval et reranking ---
    query = "extrait toutes les informations disponible"
    retrieved_docs = retrieve_docs(
        store=vector_store,
        query=query,
        k=100,
        method="mmr",
        rerank_top_k=rerank_top_k
    )
    print(f"\n📥 Chunks récupérés pour extraction : {len(retrieved_docs)}")

    # --- 8. Extraction ---
    extracted_data = extract_with_agent(
        docs=retrieved_docs,
        query=query,
        batch_size=batch_size,
        token_limit=token_limit,
        rerank_top_k=rerank_top_k
    )

    print(f"\n✅ Extraction terminée : {len(extracted_data)} entrées")
    print(json.dumps(extracted_data[:5], indent=2))  # Aperçu

    # --- 9. Sauvegarde ---
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Résultats sauvegardés dans {output_file}")
    return extracted_data
