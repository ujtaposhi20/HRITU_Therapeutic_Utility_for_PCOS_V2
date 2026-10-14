"""
knowledge.py — Tier-1 Local Verified Knowledge Engine.
Embeds 16 curated medical FAQ cards from ujt_codes/knowledge.json
and provides instant (<5ms) TF-IDF matching for standard PCOS questions.
Architecture:
  Tier 1 (this file): CPU-only TF-IDF char_wb(2,4) against verified cards.
  Tier 2 (fallback):  Live DuckDuckGo RAG in agent.py when no card matches.
The threshold (MIN_SCORE) determines when a card is "confident enough"
to use directly vs. falling through to live search.
"""

import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIN_SCORE = 0.12  # Minimum TF-IDF cosine similarity to trust a local card
TOP_K = 2         # Max cards to return per query

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_KNOWLEDGE_PATH = os.path.join(_BASE_DIR, "knowledge_cards.json")

# Fallback: try ujt_codes location if local copy not found
_UJT_PATH = os.path.join(os.path.dirname(_BASE_DIR), "ujt_codes", "knowledge.json")


def _load_cards():
    """Load knowledge cards from the local copy or ujt_codes fallback."""
    for path in [_KNOWLEDGE_PATH, _UJT_PATH]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    # If no file found, return embedded minimal set
    print("⚠️  knowledge.py: No knowledge_cards.json found. Using empty card set.")
    return []


CARDS = _load_cards()

_documents = []
_card_indices = []

for i, card in enumerate(CARDS):
    doc_parts = [
        card.get("title", ""),
        card.get("en", ""),
        card.get("bn", ""),
        card.get("keywords", ""),
    ]
    _documents.append(" ".join(doc_parts).lower())
    _card_indices.append(i)

_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4),
    max_features=8000,
)

_tfidf_matrix = None
if _documents:
    _tfidf_matrix = _vectorizer.fit_transform(_documents)

def search(query: str, k: int = TOP_K, min_score: float = MIN_SCORE):
    """
    Search the knowledge base for cards matching the query.

    Returns:
        list of (card_dict, score) tuples, sorted by descending score.
        Only cards with score >= min_score are returned.
        Returns empty list if no match found (signals Tier-2 fallback).
    """
    if not CARDS or _tfidf_matrix is None:
        return []

    query_vec = _vectorizer.transform([query.lower()])
    scores = cosine_similarity(query_vec, _tfidf_matrix).flatten()

    # Get top-k indices sorted by score
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for idx, score in ranked:
        if score >= min_score:
            results.append((CARDS[idx], float(score)))

    return results


def get_card_context(query: str, lang: str = "en") -> str:
    """
    High-level API for the agent: returns a formatted context string
    from the best matching knowledge card, or None if no match.

    Args:
        query: User's message text.
        lang:  "en" or "bn" for language preference.

    Returns:
        Formatted string with the card content, or None if no match found
        (indicating the agent should fall back to Tier-2 DuckDuckGo search).
    """
    hits = search(query)
    if not hits:
        return None

    card, score = hits[0]
    title = card.get("title", "")
    content = card.get(lang, card.get("en", ""))

    context = (
        f"[Verified Medical Knowledge — {title}]\n"
        f"{content}\n"
        f"(Confidence: {score:.2f})"
    )

    # If there's a strong second match, include it too
    if len(hits) > 1:
        card2, score2 = hits[1]
        title2 = card2.get("title", "")
        content2 = card2.get(lang, card2.get("en", ""))
        context += (
            f"\n\n[Additional Reference — {title2}]\n"
            f"{content2}\n"
            f"(Confidence: {score2:.2f})"
        )

    return context

def get_all_card_titles() -> list:
    """Return a list of all knowledge card titles (for debugging/display)."""
    return [card.get("title", "Unknown") for card in CARDS]

if __name__ == "__main__":
    print(f"Loaded {len(CARDS)} knowledge cards:")
    for title in get_all_card_titles():
        print(f"  • {title}")

    test_queries = [
        "What is PCOS?",
        "Can I get pregnant with PCOS?",
        "আমার চুল পড়ছে",
        "PCOS ki valo hoy?",
        "insulin resistance ki?",
        "Tell me about Python programming",
    ]
    print("\nSearch results:")
    for q in test_queries:
        hits = search(q)
        if hits:
            card, score = hits[0]
            print(f"  Q: {q!r:40} → {card['title']} (score={score:.3f})")
        else:
            print(f"  Q: {q!r:40} → No match (→ Tier-2 DuckDuckGo)")
