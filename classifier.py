import re
import json
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleIndicatorClassifier:
    """
    Classificateur V1 – simple, robuste, performant.
    Basé sur :
    - matching exact
    - regex & mots-clés
    - similarité TF-IDF
    """

    def __init__(self, catalog: Dict):
        self.catalog = catalog  # catalogue aplati indicateur → {description, categorie}

        self.indicator_ids = list(catalog.keys())
        self.normalized_descriptions = [
            self._normalize(catalog[i]["description"]) for i in self.indicator_ids
        ]

        # TF-IDF vectorisation des descriptions officielles
        self.vectorizer = TfidfVectorizer().fit(self.normalized_descriptions)
        self.desc_matrix = self.vectorizer.transform(self.normalized_descriptions)

        # Patterns utiles
        self.year_pattern = re.compile(r"\b(19|20)\d{2}\b")
        self.percent_pattern = re.compile(r"%|pourcent|pourcentage", re.I)

        print(f"✅ Classificateur V1 initialisé ({len(self.indicator_ids)} indicateurs)")

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9 %]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ----------------------------------------------------------
    # 1. Matching EXACT + alias
    # ----------------------------------------------------------
    def _match_exact(self, text: str) -> Tuple[str, float]:
        t = self._normalize(text)

        for ind_id in self.indicator_ids:
            item = self.catalog[ind_id]
            # description exacte
            if t == self._normalize(item["description"]):
                return ind_id, 1.0

            # alias exact
            for alias in item.get("aliases", []):
                if t == self._normalize(alias):
                    return ind_id, 0.95

        return None, 0.0

    # ----------------------------------------------------------
    # 2. Matching MOTS-CLÉS / REGEX
    # ----------------------------------------------------------
    def _keyword_match(self, text: str) -> List[Tuple[str, float]]:
        score_list = []
        t = self._normalize(text)

        for ind_id in self.indicator_ids:
            info = self.catalog[ind_id]
            score = 0.0

            desc_words = self._normalize(info["description"]).split()
            for word in desc_words:
                if word in t:
                    score += 0.05  # petit bonus par mot en commun

            # bonus catégorie (selon cahier des charges)
            if info["categorie"] == "denombrement":
                if any(k in t for k in ["population", "densité", "francophone", "locuteur", "nombre"]):
                    score += 0.2

            if info["categorie"] == "enseignement":
                if any(k in t for k in ["apprenant", "enseignement", "école", "enseignant", "université", "french"]):
                    score += 0.2

            if info["categorie"] == "vade_mecum":
                if any(k in t for k in ["oi", "organisation internationale", "traduction", "interprétation"]):
                    score += 0.2

            if info["categorie"] == "themes_specifiques":
                if any(k in t for k in ["presse", "internet", "radio", "film", "tv", "livres"]):
                    score += 0.2

            if score > 0:
                score_list.append((ind_id, score))

        # tri décroissant
        return sorted(score_list, key=lambda x: x[1], reverse=True)

    # ----------------------------------------------------------
    # 3. Matching SEMANTIQUE Simplifié (TF-IDF)
    # ----------------------------------------------------------
    def _semantic_match(self, text: str) -> List[Tuple[str, float]]:
        t = self._normalize(text)
        vec = self.vectorizer.transform([t])
        cosine_scores = cosine_similarity(vec, self.desc_matrix)[0]

        results = []
        for idx, score in enumerate(cosine_scores):
            results.append((self.indicator_ids[idx], float(score)))

        return sorted(results, key=lambda x: x[1], reverse=True)

    # ----------------------------------------------------------
    # API principale
    # ----------------------------------------------------------
    def classify(self, text: str, topk: int = 3) -> List[Dict]:
        """
        Retourne une liste des meilleurs indicateurs (id + score)
        """

        # 1️⃣ exact
        exact_id, exact_score = self._match_exact(text)
        if exact_id:
            return [{
                "indicateur_id": exact_id,
                "score": exact_score,
                "categorie": self.catalog[exact_id]["categorie"],
                "description": self.catalog[exact_id]["description"]
            }]

        # 2️⃣ mots-clés
        keyword_matches = self._keyword_match(text)

        # 3️⃣ TF-IDF
        semantic_matches = self._semantic_match(text)

        # fusionner résultats
        combined = {}

        for ind, s in keyword_matches[:5]:
            combined[ind] = combined.get(ind, 0) + s

        for ind, s in semantic_matches[:10]:
            combined[ind] = combined.get(ind, 0) + s

        # convertir en liste
        results = [
            {
                "indicateur_id": ind,
                "score": combined[ind],
                "categorie": self.catalog[ind]["categorie"],
                "description": self.catalog[ind]["description"]
            }
            for ind in combined
        ]

        # trier
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:topk]
