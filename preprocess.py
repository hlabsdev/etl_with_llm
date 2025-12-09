"""
pipeline_excel_ameliore.py
Pipeline d'extraction Excel avec extraction par lots COMPLÈTE
"""
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
import json
import os
import re
import traceback

# LangChain pour le traitement avancé
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

# Importer les constantes
from constants import KEYWORDS

# === 1. PRÉ-TRAITEMENT DES DONNÉES EXCEL ===
def preprocess_excel_data(sheets_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Prétraite les données Excel avant l'extraction
    Extrait des statistiques et métadonnées pour enrichir le prompt
    """
    print("\n🔧 Étape 0: Pré-traitement des données Excel...")
    
    stats = {
        "nombre_feuilles": len(sheets_data),
        "feuilles": list(sheets_data.keys()),
        "details_feuilles": {}
    }
    
    for sheet_name, df in sheets_data.items():
        if df.empty:
            continue
        
        sheet_stats = {
            "lignes": df.shape[0],
            "colonnes": df.shape[1],
            "premiere_ligne": df.iloc[0].tolist() if df.shape[0] > 0 else [],
            "types_donnees": {}
        }
        
        # Analyser le contenu
        try:
            # Détecter les en-têtes (première ligne non vide)
            for i in range(min(3, df.shape[0])):
                row = df.iloc[i].tolist()
                # Vérifier si la ligne semble être un en-tête (contient des mots clés)
                row_text = " ".join(str(cell) for cell in row).lower()
                if any(keyword in row_text for keyword in KEYWORDS):
                    sheet_stats["ligne_entetes"] = i
                    sheet_stats["entetes"] = row
                    break
            
            # Détecter les types de données
            if df.shape[0] > 5 and df.shape[1] > 2:
                # Échantillon de données
                sample_data = {}
                for col_idx in range(min(5, df.shape[1])):
                    col_data = df.iloc[:20, col_idx].dropna().tolist()
                    if col_data:
                        # Détecter le type
                        if any(isinstance(val, (int, float)) or (isinstance(val, str) and re.match(r'^-?\d+\.?\d*$', str(val))) for val in col_data[:10]):
                            sheet_stats["types_donnees"][f"colonne_{col_idx}"] = "numerique"
                        elif any(isinstance(val, str) and len(val) > 3 for val in col_data[:10]):
                            sheet_stats["types_donnees"][f"colonne_{col_idx}"] = "texte"
            
        except Exception as e:
            print(f"   ⚠️  Erreur d'analyse feuille '{sheet_name}': {e}")
        
        stats["details_feuilles"][sheet_name] = sheet_stats
        
        print(f"   📊 Feuille '{sheet_name}': {df.shape[0]}×{df.shape[1]}")
    
    # Analyser spécifiquement la feuille "Nombre francophones"
    all_years = set()
    if "Nombre francophones" in sheets_data:
        df_fr = sheets_data["Nombre francophones"]
        try:
            # Chercher les années dans TOUTES les colonnes
            for col in range(df_fr.shape[1]):
                for i in range(min(100, df_fr.shape[0])):
                    val = str(df_fr.iloc[i, col]).strip()
                    # Recherche plus large pour détecter 2030
                    year_patterns = [
                        r'^\d{4}$',  # 2025, 2030, etc.
                        r'^\d{4}-\d{4}$',  # 2025-2030
                        r'^\d{4}/\d{4}$',  # 2025/2030
                        r'2030'  # Spécifiquement 2030
                    ]
                    
                    for pattern in year_patterns:
                        if re.search(pattern, val):
                            # Extraire l'année
                            year_match = re.search(r'20\d{2}', val)
                            if year_match:
                                year = year_match.group()
                                if 2000 <= int(year) <= 2100:
                                    all_years.add(year)
            
            # Vérifier spécifiquement pour 2030
            if "2030" not in all_years:
                print("   🔍 Recherche spécifique de 2030...")
                # Scanner tout le DataFrame pour 2030
                for i in range(min(200, df_fr.shape[0])):
                    for col in range(min(10, df_fr.shape[1])):
                        cell_value = str(df_fr.iloc[i, col]).strip()
                        if "2030" in cell_value:
                            all_years.add("2030")
                            print(f"   ✅ 2030 trouvé à la ligne {i+1}, colonne {col+1}")
                            break
                    if "2030" in all_years:
                        break
            
            if all_years:
                stats["annees_disponibles"] = sorted(list(all_years))
                print(f"   📅 Années détectées: {', '.join(sorted(stats['annees_disponibles']))}")
                
                # Vérifier les années manquantes
                expected_years = ["2010", "2014", "2018", "2022", "2024", "2025", "2026", "2030"]
                missing_years = [year for year in expected_years if year not in all_years]
                if missing_years:
                    print(f"   ⚠️  Années attendues mais non détectées: {', '.join(missing_years)}")
            
            # Compter les pays uniques (colonne 0 probablement)
            if df_fr.shape[1] > 0:
                pays_set = set()
                for i in range(min(200, df_fr.shape[0])):
                    pays = str(df_fr.iloc[i, 0]).strip()
                    if pays and len(pays) > 2 and not any(keyword in pays.lower() for keyword in ["pays", "total", "année"]):
                        pays_set.add(pays)
                
                stats["nombre_pays_estime"] = len(pays_set)
                print(f"   🌍 Pays estimés: {len(pays_set)}")
                
        except Exception as e:
            print(f"   ⚠️  Erreur analyse francophones: {e}")
    
    return stats
