# =============================================
# splitters/multiprocess_splitter.py
# =============================================

from multiprocessing import Pool, cpu_count
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

def _split_single_document(doc):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\nFEUILLE:", "\nEN-TÊTES:", "\nLigne", "\n\n", "\n", " ", ""],
        add_start_index=True
    )

    chunks = splitter.split_text(doc["page_content"])
    split_docs = []

    for i, chunk in enumerate(chunks):
        split_docs.append({
            "page_content": chunk,
            "metadata": {
                **doc["metadata"],
                "chunk_id": i,
                "total_chunks": len(chunks)
            }
        })

    return split_docs


def split_excel_documents(documents: List[Dict]) -> List[Dict]:
    print(f"🔪 Splitter multiprocessing : {len(documents)} documents")

    workers = min(len(documents), cpu_count())

    print(f"Nombre de workers généré: {workers}")

    with Pool(processes=workers) as pool:
        results = pool.map(_split_single_document, documents)

    # Flatten
    split_docs = [chunk for sublist in results for chunk in sublist]

    print(f"🔪 Total chunks : {len(split_docs)}")
    return split_docs
