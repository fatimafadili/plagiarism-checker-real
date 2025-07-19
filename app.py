import ollama
import streamlit as st
import numpy as np
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langdetect import detect, DetectorFactory
from typing import List, Dict, Any, Tuple
import time
import pandas as pd
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import hashlib
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configurer la détection de langue
DetectorFactory.seed = 0

# 🚀 Initialisation du système
@st.cache_resource(show_spinner=False)
def initialize_system():
    try:
        embeddings = OllamaEmbeddings(
            model="mxbai-embed-large:latest",
            temperature=0.01,
            top_k=50
        )
        
        vecdb = Chroma(
            persist_directory="philo_db",
            embedding_function=embeddings,
            collection_name="rag-chroma"
        )
        
        dataset_contents = []
        if hasattr(vecdb, '_collection'):
            dataset_contents = vecdb._collection.get(include=['documents'])['documents']
        
        # Vectorizer multilingue (français/anglais)
        tfidf_vectorizer = TfidfVectorizer(
            stop_words=None,  # Désactiver les stop words pour gérer plusieurs langues
            ngram_range=(1, 3),
            analyzer='word'
        )
        if dataset_contents:
            tfidf_vectorizer.fit(dataset_contents)
        
        return vecdb, embeddings, dataset_contents, tfidf_vectorizer
    
    except Exception as e:
        st.error(f"Erreur d'initialisation: {str(e)}")
        st.stop()

vecdb, embeddings, dataset_contents, tfidf_vectorizer = initialize_system()

# 🔍 Vérification de copie exacte multilingue
def check_exact_match(input_text: str, dataset: List[str]) -> List[Tuple[str, float]]:
    def normalize(text):
        text = re.sub(r'[^\w\s]', '', text.strip().lower())
        return re.sub(r'\s+', ' ', text)
    
    normalized_input = normalize(input_text)
    input_hash = hashlib.md5(normalized_input.encode('utf-8')).hexdigest()
    matches = []
    
    for doc in dataset:
        normalized_doc = normalize(doc)
        doc_hash = hashlib.md5(normalized_doc.encode('utf-8')).hexdigest()
        
        if input_hash == doc_hash:
            return [(doc, 1.0)]
        
        # Similarité textuelle indépendante de la langue
        match_ratio = SequenceMatcher(None, normalized_input, normalized_doc).ratio()
        if match_ratio > 0.7:
            matches.append((doc, match_ratio))
        
        # Vérification des segments longs
        input_words = normalized_input.split()
        doc_words = normalized_doc.split()
        
        for i in range(len(input_words) - 8 + 1):  # Fenêtre de 8 mots
            segment = ' '.join(input_words[i:i+8])
            if segment in normalized_doc:
                matches.append((doc, max(match_ratio, 0.85)))
                break
    
    unique_matches = {match[0]: match[1] for match in matches}
    return sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)

# 🌐 Traduction intelligente (si nécessaire)
@st.cache_data(ttl=3600, show_spinner=False)
def translate_text(text: str, target_lang: str) -> str:
    try:
        if len(text) < 50:  # Ne pas traduire les textes trop courts
            return text
            
        response = ollama.chat(
            model="llama3.1",
            messages=[{
                "role": "system",
                "content": f"Traduis ce texte en {target_lang} en conservant le sens original:\n{text}"
            }],
            options={'temperature': 0.1}
        )
        return response["message"]["content"]
    except Exception as e:
        st.warning(f"Traduction partielle: {str(e)}")
        return text

# 🧠 Similarité sémantique améliorée
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def calculate_similarity(text1: str, text2: str) -> float:
    try:
        # Similarité lexicale (TF-IDF)
        vectors = tfidf_vectorizer.transform([text1, text2])
        tfidf_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        # Similarité sémantique (Cross-Encoder)
        cross_score = cross_encoder.predict([[text1, text2]])[0]
        
        # Combinaison pondérée
        return (cross_score * 0.7) + (tfidf_sim * 0.3)
    except:
        return SequenceMatcher(None, text1, text2).ratio()

# 🔎 Recherche hybride multilingue
def hybrid_search(query: str, dataset: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
    try:
        # Détection de la langue de la requête
        query_lang = detect(query) if len(query) > 20 else 'en'
        
        # 1. Vérifier les copies exactes
        exact_matches = check_exact_match(query, dataset)
        if exact_matches:
            return [{
                "content": match[0],
                "similarity": match[1],
                "match_type": "exact",
                "metadata": {},
                "combined_score": match[1]
            } for match in exact_matches[:top_k]]
        
        # 2. Recherche dans la langue d'origine
        vector_results = vecdb.similarity_search_with_score(query, k=top_k*2)
        
        # 3. Si la requête est en français, chercher aussi en anglais et vice versa
        translated_results = []
        if query_lang == 'fr':
            translated_query = translate_text(query, 'en')
            if translated_query != query:
                translated_results = vecdb.similarity_search_with_score(translated_query, k=top_k)
        elif query_lang == 'en':
            translated_query = translate_text(query, 'fr')
            if translated_query != query:
                translated_results = vecdb.similarity_search_with_score(translated_query, k=top_k)
        
        # Combiner les résultats
        all_results = []
        
        # Ajouter les résultats originaux
        for doc, score in vector_results:
            sim_score = calculate_similarity(query, doc.page_content)
            all_results.append({
                "content": doc.page_content,
                "similarity": sim_score,
                "match_type": "semantic",
                "metadata": doc.metadata,
                "combined_score": sim_score
            })
        
        # Ajouter les résultats traduits
        for doc, score in translated_results:
            translated_content = translate_text(doc.page_content, query_lang)
            sim_score = calculate_similarity(query, translated_content)
            all_results.append({
                "content": doc.page_content,
                "similarity": sim_score,
                "match_type": "translated",
                "metadata": doc.metadata,
                "combined_score": sim_score * 0.9  # Légère pénalité pour la traduction
            })
        
        # Éliminer les doublons et trier
        unique_results = {}
        for res in all_results:
            if res["content"] not in unique_results or res["combined_score"] > unique_results[res["content"]]["combined_score"]:
                unique_results[res["content"]] = res
        
        return sorted(unique_results.values(), key=lambda x: x["combined_score"], reverse=True)[:top_k]
    
    except Exception as e:
        st.error(f"Erreur de recherche: {str(e)}")
        return []

# 📊 Analyse des similarités d'idées
def analyze_ideas(input_text: str, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ideas = []
    sentences = [s.strip() for s in re.split(r'[.!?]', input_text) if len(s.strip().split()) > 5]
    
    for match in matches:
        if match["combined_score"] < 0.4:  # Seuil pour les idées similaires
            continue
            
        match_sentences = [s.strip() for s in re.split(r'[.!?]', match["content"]) if len(s.strip().split()) > 5]
        
        for sent in sentences:
            for match_sent in match_sentences:
                sim_score = calculate_similarity(sent, match_sent)
                if sim_score > 0.5:  # Seuil pour similarité d'idée
                    ideas.append({
                        "source_sentence": sent,
                        "matched_sentence": match_sent,
                        "similarity": sim_score,
                        "source_content": match["content"][:200] + "...",
                        "metadata": match.get("metadata", {})
                    })
    
    # Regrouper les idées similaires
    grouped_ideas = defaultdict(list)
    for idea in ideas:
        key = idea["source_sentence"][:50]  # Regrouper par phrase source
        grouped_ideas[key].append(idea)
    
    # Garder la meilleure correspondance pour chaque groupe
    return [max(group, key=lambda x: x["similarity"]) for group in grouped_ideas.values()]

# 🎨 Interface Streamlit (inchangée)
def main():
    st.set_page_config(
        page_title="🔍 Detecteur de Plagiat Expert",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
        .header {
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .exact-match { border-left: 6px solid #ef4444; background-color: rgba(239, 68, 68, 0.05); }
        .partial-match { border-left: 6px solid #f59e0b; background-color: rgba(245, 158, 11, 0.05); }
        .semantic-match { border-left: 6px solid #10b981; background-color: rgba(16, 185, 129, 0.05); }
        .sentence-match { background-color: rgba(255, 237, 213, 0.7); padding: 2px 6px; border-radius: 4px; }
        .idea-match { background-color: rgba(173, 216, 230, 0.3); padding: 10px; border-radius: 8px; margin: 5px 0; }
    </style>
    """, unsafe_allow_html=True)
    
   
