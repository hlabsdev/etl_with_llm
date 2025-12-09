import re
import json
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ExcelMetadataClassifier:
    """
    Classificateur spécialisé pour les fichiers Excel de métadonnées d'indicateurs.
    Analyse la matrice "Matrice Indicateurs-Source données" pour mapper vers l'annexe des indicateurs.
    """

    def __init__(self, catalog: Dict):
        self.catalog = catalog  # catalogue des indicateurs de l'annexe
        
        # Créer une base de connaissances à partir du catalogue
        self.indicator_base = {}
        for category, indicators in catalog.items():
            for indicator_name, details in indicators.items():
                # Normaliser le nom
                normalized_name = self._normalize(indicator_name)
                self.indicator_base[normalized_name] = {
                    "id": f"{category}.{indicator_name}",
                    "categorie": category,
                    "description_complete": indicator_name,
                    "details": details,
                    "mots_cles": self._extract_keywords(indicator_name)
                }
        
        # Préparer pour TF-IDF
        self.indicator_descriptions = [self._normalize(name) for name in self.indicator_base.keys()]
        self.indicator_ids = list(self.indicator_base.keys())
        
        # TF-IDF
        self.vectorizer = TfidfVectorizer().fit(self.indicator_descriptions)
        self.desc_matrix = self.vectorizer.transform(self.indicator_descriptions)
        
        print(f"✅ Classificateur initialisé ({len(self.indicator_ids)} indicateurs)")

    def _normalize(self, text: str) -> str:
        """Normalise le texte pour les comparaisons"""
        text = text.lower()
        text = re.sub(r"[^a-z0-9éèêàâôûùüç\s]", " ", text)  # Garder les accents français
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_keywords(self, text: str) -> List[str]:
        """Extrait les mots-clés d'un indicateur"""
        text = self._normalize(text)
        # Supprimer les mots vides
        stop_words = {"de", "la", "le", "les", "des", "du", "en", "à", "au", "aux", 
                     "dans", "pour", "par", "sur", "avec", "et", "ou", "où", "qui", 
                     "que", "quoi", "quand", "comment", "combien"}
        words = text.split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords

    def _parse_excel_row(self, row_text: str) -> Dict:
        """Parse une ligne de la matrice Excel pour en extraire les informations"""
        data = {}
        
        # Détecter les sections
        section_patterns = {
            "Section 1 – Francophones": "denombrement",
            "Section 2-1- Cartographie Enseignement - Données générales": "enseignement",
            "Section 2-2- Cartographie Enseignement - Enseignement du et en français": "enseignement",
            "Section 3 - Economie et numérique": "themes_specifiques",
            "Section 4 - Culture, Médias et découvrabilité": "themes_specifiques"
        }
        
        # Chercher la section
        for section_name, category in section_patterns.items():
            if section_name in row_text:
                data["section"] = section_name
                data["categorie"] = category
                break
        
        # Chercher les indicateurs (colonne B probablement)
        indicator_match = re.search(r'Indicateur.*?\n(.*?)(?=\nFichier|$)', row_text, re.DOTALL | re.IGNORECASE)
        if indicator_match:
            indicators_text = indicator_match.group(1)
            # Détecter les indicateurs multiples (séparés par des tirets ou numéros)
            indicators = []
            
            # Chercher des listes
            bullet_pattern = re.compile(r'[-•]\s*(.+?)(?=\n[-•]|\nFichier|\n|$)')
            bullets = bullet_pattern.findall(indicators_text)
            if bullets:
                indicators.extend(bullets)
            else:
                # Prendre toute la ligne comme indicateur
                lines = indicators_text.split('\n')
                for line in lines:
                    if line.strip() and not line.strip().startswith('Fichier'):
                        indicators.append(line.strip())
            
            data["indicateurs_excel"] = indicators
        
        # Chercher les fichiers sources
        file_match = re.search(r'Fichier.*?\n(.*?)(?=\nRemarques|\n|$)', row_text, re.DOTALL | re.IGNORECASE)
        if file_match:
            files_text = file_match.group(1)
            files = [f.strip() for f in files_text.split('\n') if f.strip()]
            data["fichiers_sources"] = files
        
        # Chercher les remarques
        remarques_match = re.search(r'Remarques.*?\n(.*?)(?=\n|$)', row_text, re.DOTALL | re.IGNORECASE)
        if remarques_match:
            data["remarques"] = remarques_match.group(1).strip()
        
        return data

    def _match_to_annexe(self, excel_indicator: str) -> List[Dict]:
        """Match un indicateur Excel avec les indicateurs de l'annexe"""
        excel_norm = self._normalize(excel_indicator)
        
        # 1. Matching exact ou partiel
        exact_matches = []
        for base_id, base_info in self.indicator_base.items():
            base_norm = self._normalize(base_info["description_complete"])
            
            # Calculer la similarité
            similarity = SequenceMatcher(None, excel_norm, base_norm).ratio()
            
            if similarity > 0.6:  # Seuil de similarité
                exact_matches.append({
                    "indicateur_id": base_info["id"],
                    "score": similarity,
                    "categorie": base_info["categorie"],
                    "description": base_info["description_complete"],
                    "match_type": "exact" if similarity > 0.8 else "partial"
                })
        
        # 2. Matching par mots-clés
        keyword_matches = []
        excel_keywords = set(self._extract_keywords(excel_indicator))
        
        for base_id, base_info in self.indicator_base.items():
            base_keywords = set(base_info["mots_cles"])
            
            if excel_keywords and base_keywords:
                intersection = excel_keywords.intersection(base_keywords)
                union = excel_keywords.union(base_keywords)
                
                if intersection:
                    jaccard = len(intersection) / len(union) if union else 0
                    if jaccard > 0.3:  # Seuil Jaccard
                        keyword_matches.append({
                            "indicateur_id": base_info["id"],
                            "score": jaccard,
                            "categorie": base_info["categorie"],
                            "description": base_info["description_complete"],
                            "match_type": "keywords",
                            "mots_communs": list(intersection)
                        })
        
        # 3. Matching sémantique TF-IDF
        semantic_matches = []
        if excel_norm:
            vec = self.vectorizer.transform([excel_norm])
            cosine_scores = cosine_similarity(vec, self.desc_matrix)[0]
            
            for idx, score in enumerate(cosine_scores):
                if score > 0.2:  # Seuil bas pour la sémantique
                    base_info = self.indicator_base[self.indicator_ids[idx]]
                    semantic_matches.append({
                        "indicateur_id": base_info["id"],
                        "score": float(score),
                        "categorie": base_info["categorie"],
                        "description": base_info["description_complete"],
                        "match_type": "semantic"
                    })
        
        # Combiner et trier tous les matches
        all_matches = {}
        
        for match in exact_matches + keyword_matches + semantic_matches:
            match_id = match["indicateur_id"]
            if match_id not in all_matches or match["score"] > all_matches[match_id]["score"]:
                all_matches[match_id] = match
        
        # Convertir en liste et trier
        sorted_matches = sorted(all_matches.values(), key=lambda x: x["score"], reverse=True)
        
        return sorted_matches[:5]  # Top 5

    def classify_excel_content(self, excel_content: str) -> Dict:
        """
        Analyse le contenu Excel et classe tous les indicateurs trouvés
        """
        print("\n🔍 Analyse du fichier Excel de métadonnées...")
        
        # Séparer par lignes (basé sur la structure Markdown/HTML)
        lines = excel_content.split('\n')
        
        # Reconstruire les lignes de tableau
        current_row = ""
        rows = []
        
        for line in lines:
            if line.strip().startswith('|') and '---' not in line:
                current_row += line + '\n'
            elif current_row and not line.strip().startswith('|'):
                rows.append(current_row.strip())
                current_row = ""
        
        if current_row:
            rows.append(current_row.strip())
        
        print(f"📊 Lignes détectées dans le tableau: {len(rows)}")
        
        # Analyser chaque ligne non-en-tête
        results = {
            "denombrement": [],
            "enseignement": [],
            "vade_mecum": [],
            "themes_specifiques": [],
            "statistiques": {
                "total_lignes": len(rows),
                "lignes_analysées": 0,
                "indicateurs_trouves": 0,
                "matches": 0
            }
        }
        
        # Ignorer la première ligne (en-tête)
        for i, row in enumerate(rows[1:], start=1):
            if not row.strip() or '---' in row:
                continue
            
            # Parser la ligne
            row_data = self._parse_excel_row(row)
            
            if "indicateurs_excel" in row_data:
                for excel_indicator in row_data["indicateurs_excel"]:
                    matches = self._match_to_annexe(excel_indicator)
                    
                    if matches:
                        best_match = matches[0]
                        
                        # Ajouter aux résultats
                        result_entry = {
                            "indicateur_excel": excel_indicator,
                            "match_annexe": {
                                "id": best_match["indicateur_id"],
                                "description": best_match["description"],
                                "score": best_match["score"],
                                "match_type": best_match["match_type"]
                            },
                            "section_excel": row_data.get("section", "Inconnue"),
                            "fichiers_sources": row_data.get("fichiers_sources", []),
                            "remarques": row_data.get("remarques", "")
                        }
                        
                        # Classer par catégorie
                        categorie = best_match["categorie"]
                        if categorie in results:
                            results[categorie].append(result_entry)
                            results["statistiques"]["matches"] += 1
                        
                        results["statistiques"]["indicateurs_trouves"] += 1
                
                results["statistiques"]["lignes_analysées"] += 1
        
        # Calculer les statistiques
        total_indicators_ref = sum(len(self.catalog.get(cat, {})) for cat in ["denombrement", "enseignement", "vade_mecum", "themes_specifiques"])
        if total_indicators_ref > 0:
            coverage_rate = (results["statistiques"]["matches"] / total_indicators_ref) * 100
            results["statistiques"]["taux_couverture"] = f"{coverage_rate:.1f}%"
        
        # Ajouter des détails par catégorie
        for categorie in ["denombrement", "enseignement", "vade_mecum", "themes_specifiques"]:
            count = len(results[categorie])
            total_ref = len(self.catalog.get(categorie, {}))
            results["statistiques"][f"{categorie}_matches"] = count
            results["statistiques"][f"{categorie}_total_ref"] = total_ref
            
            if total_ref > 0:
                taux = (count / total_ref) * 100
                results["statistiques"][f"{categorie}_taux"] = f"{taux:.1f}%"
        
        return results

    def generate_report(self, classification_results: Dict) -> str:
        """Génère un rapport détaillé de la classification"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("📊 RAPPORT DE CLASSIFICATION DES INDICATEURS")
        report_lines.append("=" * 80)
        
        stats = classification_results["statistiques"]
        report_lines.append(f"\n📈 STATISTIQUES GLOBALES:")
        report_lines.append(f"   • Lignes analysées: {stats['lignes_analysées']}/{stats['total_lignes']}")
        report_lines.append(f"   • Indicateurs Excel détectés: {stats['indicateurs_trouves']}")
        report_lines.append(f"   • Matches avec l'annexe: {stats['matches']}")
        
        if 'taux_couverture' in stats:
            report_lines.append(f"   • Taux de couverture: {stats['taux_couverture']}")
        
        # Détails par catégorie
        report_lines.append(f"\n🎯 DÉTAILS PAR CATÉGORIE:")
        for categorie in ["denombrement", "enseignement", "vade_mecum", "themes_specifiques"]:
            matches = stats.get(f"{categorie}_matches", 0)
            total_ref = stats.get(f"{categorie}_total_ref", 0)
            taux = stats.get(f"{categorie}_taux", "0%")
            
            nom_categorie = {
                "denombrement": "DÉNOMBREMENT",
                "enseignement": "ENSEIGNEMENT", 
                "vade_mecum": "VADE-MECUM",
                "themes_specifiques": "THÈMES SPÉCIFIQUES"
            }.get(categorie, categorie.upper())
            
            report_lines.append(f"\n   {nom_categorie}:")
            report_lines.append(f"     • Matches: {matches}/{total_ref} ({taux})")
            
            # Lister les matches
            if classification_results[categorie]:
                for i, match in enumerate(classification_results[categorie][:3], 1):
                    excel_ind = match["indicateur_excel"][:50] + "..." if len(match["indicateur_excel"]) > 50 else match["indicateur_excel"]
                    annexe_desc = match["match_annexe"]["description"][:50] + "..." if len(match["match_annexe"]["description"]) > 50 else match["match_annexe"]["description"]
                    score = match["match_annexe"]["score"]
                    
                    report_lines.append(f"     {i}. Excel: '{excel_ind}'")
                    report_lines.append(f"        ➡ Annexe: '{annexe_desc}'")
                    report_lines.append(f"        Score: {score:.2f} ({match['match_annexe']['match_type']})")
                
                if len(classification_results[categorie]) > 3:
                    report_lines.append(f"     ... et {len(classification_results[categorie]) - 3} autres")
        
        # Indicateurs non matchés
        report_lines.append(f"\n⚠️  INDICATEURS EXCEL SANS MATCH:")
        unmatched_count = 0
        for categorie in ["denombrement", "enseignement", "vade_mecum", "themes_specifiques"]:
            for match in classification_results[categorie]:
                if match["match_annexe"]["score"] < 0.5:
                    unmatched_count += 1
                    if unmatched_count <= 5:
                        report_lines.append(f"   • {match['indicateur_excel'][:60]}...")
        
        if unmatched_count > 5:
            report_lines.append(f"   ... et {unmatched_count - 5} autres")
        
        report_lines.append(f"\n" + "=" * 80)
        
        return "\n".join(report_lines)


# === FONCTION D'UTILISATION ===
def load_annexe_catalog() -> Dict:
    """Charge le catalogue des indicateurs de l'annexe"""
    # Structure basée sur l'annexe fournie
    catalog = {
        "denombrement": {
            "Nombre de francophones": {
                "description": "Nombre total de francophones",
                "type": "quantitatif",
                "unite": "personnes"
            },
            "Densité francophone de la population": {
                "description": "Pourcentage de francophones dans la population",
                "type": "pourcentage", 
                "unite": "%"
            },
            "Nombre de locuteurs pour les autres langues officielles ou nationales": {
                "description": "Locuteurs d'autres langues officielles",
                "type": "quantitatif",
                "unite": "personnes"
            },
            "Evolutions démographiques des populations francophones": {
                "description": "Evolution temporelle des populations francophones",
                "type": "tendance",
                "unite": "variation"
            },
            "Prospective de la population francophone dans le monde": {
                "description": "Projections futures des populations francophones",
                "type": "prospective",
                "unite": "personnes"
            },
            "Usage du français": {
                "description": "Utilisation du français dans différents contextes",
                "type": "qualitatif",
                "unite": "texte"
            }
        },
        "enseignement": {
            "Nombre d'apprenants en français": {
                "description": "Apprenants étudiant en français",
                "type": "quantitatif",
                "unite": "personnes"
            },
            "Nombre d'apprenants scolarisés dans des établissements français du réseau AEFE": {
                "description": "Élèves dans les établissements AEFE",
                "type": "quantitatif",
                "unite": "personnes"
            },
            "Nombre d'apprenants du français": {
                "description": "Personnes apprenant le français",
                "type": "quantitatif", 
                "unite": "personnes"
            },
            "Nombre d'apprenants en enseignement bilingue": {
                "description": "Élèves en programmes bilingues",
                "type": "quantitatif",
                "unite": "personnes"
            },
            # ... ajouter tous les autres indicateurs de l'annexe
        },
        "vade_mecum": {
            "Langue officielle enregistrée dans les OI par les délégations issues de pays membres, associés ou observateurs de l'OIF": {
                "description": "Langues officielles dans les organisations internationales",
                "type": "qualitatif",
                "unite": "texte"
            },
            # ... ajouter les autres
        },
        "themes_specifiques": {
            "Statut institutionnel de la langue française dans le pays": {
                "description": "Statut légal du français",
                "type": "qualitatif",
                "unite": "texte"
            },
            # ... ajouter les autres
        }
    }
    
    return catalog


def main_classification():
    """Fonction principale pour classifier le fichier Excel"""
    # 1. Charger le catalogue de l'annexe
    catalog = load_annexe_catalog()
    
    # 2. Initialiser le classificateur
    classifier = ExcelMetadataClassifier(catalog)
    
    # 3. Lire le contenu du fichier Excel (à remplacer par votre contenu réel)
    with open("matrice_indicateurs.txt", "r", encoding="utf-8") as f:
        excel_content = f.read()
    
    # 4. Classifier le contenu
    results = classifier.classify_excel_content(excel_content)
    
    # 5. Générer le rapport
    report = classifier.generate_report(results)
    print(report)
    
    # 6. Sauvegarder les résultats
    with open("classification_resultats.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Résultats sauvegardés dans 'classification_resultats.json'")
    
    return results


if __name__ == "__main__":
    main_classification()