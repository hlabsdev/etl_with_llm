cet pipeline est juste une partie permettant de tester le rag avec multiprocessing 
 ## Agent regroupe les agnes 
 - extracteur
 - retiever
 - saver

 ## steps
  liste les differentes etapes en partantt du splitting au vectorestore

## loader: 
ici on ingest le fichier a traiter

## core:
 l'orchestrateur du pipeline

 main.py vous permet d'executer le pipeline 
 python3  main.py -i data/population.xlsx -o outputs/nombre_extracted.json

comment debuter

git clone git@github.com:mtchalim3/etl_with_llm.git
cd pipeline
python3 -m venv venv
activer le venv
 puis
 pip install -r requirements.txt
 