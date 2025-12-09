# main.py
import argparse
from core.orchestration_pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Pipeline d'extraction Excel → JSON")
    parser.add_argument("--input", "-i", required=True, help="Chemin du fichier Excel")
    parser.add_argument("--output", "-o", required=True, help="Chemin du fichier JSON de sortie")
    parser.add_argument("--batch_size", type=int, default=10, help="Taille de batch pour l'extraction")
    parser.add_argument("--token_limit", type=int, default=6000, help="Limite de tokens par batch")
    parser.add_argument("--rerank_top_k", type=int, default=30, help="Top-k pour reranking")
    args = parser.parse_args()

    run_pipeline(
        excel_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size,
        token_limit=args.token_limit,
        rerank_top_k=args.rerank_top_k
    )

if __name__ == "__main__":
    main()
