import ollama
import streamlit as st
import numpy as np
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langdetect import detect, DetectorFactory
from typing import List, Dict, Any, Tuple
import re
import hashlib
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# Configurer la détection de langue
DetectorFactory.seed = 0

# 🚀 Initialisation du système avec des optimisations
@st.cache_resource(show_spinner=False)
def initialize_system():
    try:
        # Initialize embeddings with optimized parameters
        embeddings = OllamaEmbeddings(
            model="mxbai-embed-large:latest",
            temperature=0.01,
            top_k=50
        )
        
        # Load Chroma DB with optimized settings
        vecdb = Chroma(
            persist_directory="philo_db",
            embedding_function=embeddings,
            collection_name="rag-chroma"
        )
        
        # Preload dataset contents
        dataset_contents = []
        if hasattr(vecdb, '_collection'):
            dataset_contents = vecdb._collection.get(include=['documents'])['documents']
        
        # Initialize TF-IDF vectorizer with optimized parameters
        tfidf_vectorizer = TfidfVectorizer(
            stop_words=None,
            ngram_range=(1, 2),  # Reduced from (1,3) for better performance
            analyzer='word',
            max_features=5000  # Limit features to improve performance
        )
        
        if dataset_contents:
            tfidf_vectorizer.fit(dataset_contents[:1000])  # Fit on a subset for faster initialization
        
        # Preload cross encoder model
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        return vecdb, embeddings, dataset_contents, tfidf_vectorizer, cross_encoder
    
    except Exception as e:
        st.error(f"Erreur d'initialisation: {str(e)}")
        st.stop()

# Load all resources at startup
vecdb, embeddings, dataset_contents, tfidf_vectorizer, cross_encoder = initialize_system()

# 🔍 Optimized exact match checking
def check_exact_match(input_text: str, dataset: List[str]) -> List[Tuple[str, float]]:
    @st.cache_data(max_entries=1000)
    def normalize(text):
        text = re.sub(r'[^\w\s]', '', text.strip().lower())
        return re.sub(r'\s+', ' ', text)
    
    normalized_input = normalize(input_text)
    input_hash = hashlib.md5(normalized_input.encode('utf-8')).hexdigest()
    matches = []
    
    # Check only a subset of documents for performance
    for doc in dataset[:1000]:  # Limit to first 1000 documents
        normalized_doc = normalize(doc)
        doc_hash = hashlib.md5(normalized_doc.encode('utf-8')).hexdigest()
        
        if input_hash == doc_hash:
            return [(doc, 1.0)]
        
        # Only calculate ratio if first 20 chars match
        if normalized_input[:20] in normalized_doc:
            match_ratio = SequenceMatcher(None, normalized_input, normalized_doc).ratio()
            if match_ratio > 0.7:
                matches.append((doc, match_ratio))
    
    unique_matches = {match[0]: match[1] for match in matches}
    return sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)[:10]  # Return only top 10

# 🌐 Optimized translation with caching
@st.cache_data(ttl=3600, show_spinner=False, max_entries=100)
def translate_text(text: str, target_lang: str) -> str:
    try:
        if len(text) < 50:  # Skip translation for short texts
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
    except Exception:
        return text

# 🧠 Optimized similarity calculation
def calculate_similarity(text1: str, text2: str) -> float:
    try:
        # Use cached TF-IDF vectors
        vectors = tfidf_vectorizer.transform([text1, text2])
        tfidf_sim = np.dot(vectors[0].toarray(), vectors[1].toarray().T)[0][0]
        
        # Use cross encoder (already initialized)
        cross_score = cross_encoder.predict([[text1, text2]])[0]
        
        return (cross_score * 0.7) + (tfidf_sim * 0.3)
    except:
        return SequenceMatcher(None, text1[:200], text2[:200]).ratio()  # Compare only first 200 chars

# 🔎 Optimized hybrid search
def hybrid_search(query: str, dataset: List[str], top_k: int = 5) -> List[Dict[str, Any]]:  # Reduced default top_k
    try:
        # Detect language (cached)
        query_lang = detect(query) if len(query) > 20 else 'en'
        
        # 1. Check exact matches (optimized)
        exact_matches = check_exact_match(query, dataset)
        if exact_matches:
            return [{
                "content": match[0],
                "similarity": match[1],
                "match_type": "exact",
                "metadata": {},
                "combined_score": match[1]
            } for match in exact_matches[:top_k]]
        
        # 2. Semantic search with score
        vector_results = vecdb.similarity_search_with_score(query, k=top_k)
        
        # 3. Translated search (only if high confidence)
        translated_results = []
        if query_lang == 'fr' and len(query) > 30:
            translated_query = translate_text(query, 'en')
            if translated_query != query:
                translated_results = vecdb.similarity_search_with_score(translated_query, k=top_k//2)
        elif query_lang == 'en' and len(query) > 30:
            translated_query = translate_text(query, 'fr')
            if translated_query != query:
                translated_results = vecdb.similarity_search_with_score(translated_query, k=top_k//2)
        
        # Combine and process results
        all_results = []
        
        for doc, score in vector_results:
            sim_score = calculate_similarity(query, doc.page_content)
            all_results.append({
                "content": doc.page_content,
                "similarity": sim_score,
                "match_type": "semantic",
                "metadata": doc.metadata,
                "combined_score": sim_score
            })
        
        for doc, score in translated_results:
            translated_content = translate_text(doc.page_content, query_lang)
            sim_score = calculate_similarity(query, translated_content)
            all_results.append({
                "content": doc.page_content,
                "similarity": sim_score,
                "match_type": "translated",
                "metadata": doc.metadata,
                "combined_score": sim_score * 0.9
            })
        
        # Deduplicate and sort
        unique_results = {}
        for res in all_results:
            content_key = res["content"][:100]  # Use first 100 chars as key
            if content_key not in unique_results or res["combined_score"] > unique_results[content_key]["combined_score"]:
                unique_results[content_key] = res
        
        return sorted(unique_results.values(), key=lambda x: x["combined_score"], reverse=True)[:top_k]
    
    except Exception as e:
        st.error(f"Erreur de recherche: {str(e)}")
        return []

# 📊 Optimized idea analysis
def analyze_ideas(input_text: str, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Extract sentences only once
    sentences = [s.strip() for s in re.split(r'[.!?]', input_text) if 5 < len(s.strip().split()) <= 50]  # Limit sentence length
    
    ideas = []
    for match in matches[:5]:  # Only analyze top 5 matches
        if match["combined_score"] < 0.4:
            continue
            
        match_sentences = [s.strip() for s in re.split(r'[.!?]', match["content"]) if 5 < len(s.strip().split()) <= 50]
        
        for sent in sentences:
            for match_sent in match_sentences[:10]:  # Only check first 10 sentences
                sim_score = calculate_similarity(sent, match_sent)
                if sim_score > 0.5:
                    ideas.append({
                        "source_sentence": sent,
                        "matched_sentence": match_sent,
                        "similarity": sim_score,
                        "source_content": match["content"][:200] + "...",
                        "metadata": match.get("metadata", {})
                    })
    
    # Group and return top ideas
    grouped_ideas = defaultdict(list)
    for idea in ideas:
        key = idea["source_sentence"][:30]  # Smaller key for grouping
        grouped_ideas[key].append(idea)
    
    return [max(group, key=lambda x: x["similarity"]) for group in grouped_ideas.values()][:5]  # Return top 5 ideas

# 🎨 Optimized Streamlit interface
def main():
    st.set_page_config(
        page_title="🔍 Detecteur de Plagiat Expert",
        page_icon="🔍",
        layout="wide"
    )
    
    # CSS with simplified selectors
    st.markdown("""
    <style>
        .header { background: #1e3a8a; color: white; padding: 1rem; }
        .exact-match { border-left: 4px solid #ef4444; }
        .partial-match { border-left: 4px solid #f59e0b; }
        .semantic-match { border-left: 4px solid #10b981; }
        .sentence-match { background: #fff5e6; padding: 2px 4px; }
        .idea-match { background: #e6f3ff; padding: 8px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔍 Detecteur de Plagiat Expert")
    
    with st.form("search_form"):
        input_text = st.text_area("Entrez votre texte à analyser:", height=200)
        submitted = st.form_submit_button("Analyser")
    
    if submitted and input_text:
        with st.spinner("Analyse en cours..."):
            start_time = time.time()
            
            # Perform search
            matches = hybrid_search(input_text, dataset_contents)
            
            # Analyze ideas
            similar_ideas = analyze_ideas(input_text, matches)
            
            st.success(f"Analyse terminée en {time.time() - start_time:.2f} secondes")
            
            # Display results
            if matches:
                st.subheader("📝 Correspondances trouvées")
                
                for match in matches:
                    match_class = ""
                    if match["match_type"] == "exact":
                        match_class = "exact-match"
                    elif match["similarity"] > 0.7:
                        match_class = "partial-match"
                    else:
                        match_class = "semantic-match"
                    
                    with st.container():
                        st.markdown(f'<div class="{match_class}">', unsafe_allow_html=True)
                        st.write(f"**Similarité:** {match['similarity']:.2f}")
                        st.write(match["content"][:500] + "...")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                if similar_ideas:
                    st.subheader("💡 Idées similaires")
                    for idea in similar_ideas:
                        with st.container():
                            st.markdown('<div class="idea-match">', unsafe_allow_html=True)
                            st.write(f"**Votre phrase:** {idea['source_sentence']}")
                            st.write(f"**Phrase similaire:** {idea['matched_sentence']}")
                            st.write(f"**Similarité:** {idea['similarity']:.2f}")
                            st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Aucune correspondance significative trouvée.")

if __name__ == "__main__":
    main()