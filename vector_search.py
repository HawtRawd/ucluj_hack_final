import unicodedata
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# ==========================================
# 1. CONFIGURATION & INITIALIZATION
# ==========================================
# Load these globally so the model is ready in memory when the backend calls it.
PINECONE_API_KEY = "here goes key"
INDEX_NAME = "u-cluj-scouting"

print("Starting Vector Engine: Loading Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def clean_name_input(raw_name):
    """Formats the user's input to match the exact database format (Title Case)."""
    # Turns "alex chipciu" into "Alex Chipciu"
    clean_name = str(raw_name).strip().title()
    clean_name = unicodedata.normalize('NFKD', clean_name).encode('ASCII', 'ignore').decode('utf-8')
    return clean_name


# ==========================================
# 3. CORE SEARCH LOGIC (THE HANDOFF)
# ==========================================
def hybrid_player_search(user_query: str, search_mode: str = "semantic", top_k: int = 5):
    """
    Returns a list of Player IDs (and match scores) based on text or name search.
    The backend developer takes these IDs to run his News/LLM pipeline.
    """

    if search_mode == "exact":
        search_results = index.query(
            vector=[0.0] * 384,
            top_k=1,
            include_metadata=False,  # We don't need metadata, just the ID
            filter={
                "name": {"$eq": clean_name_input(user_query)}
            }
        )

    else:
        # Semantic search
        query_vector = model.encode(user_query).tolist()
        search_results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=False  # Handing just the ID to the backend dev
        )

    # Return a clean list of dictionaries containing just the ID and score
    results = [{"player_id": match['id'], "score": round(match['score'], 3)} for match in
               search_results.get('matches', [])]
    return results


def get_similar_players(player_id: str, top_k: int = 5):
    """
    Returns a list of Player IDs similar to the provided ID.
    """
    try:
        search_results = index.query(
            id=player_id,
            top_k=top_k + 1,
            include_metadata=False
        )
    except Exception as e:
        print(f"Error finding player: {e}")
        return []

    recommendations = []
    for match in search_results.get('matches', []):
        if match['id'] == player_id:
            continue

        recommendations.append({
            "player_id": match['id'],
            "score": round(match['score'], 3)
        })

    return recommendations[:top_k]


# ==========================================
# 4. TESTING THE OUTPUT
# ==========================================
if __name__ == "__main__":

    print("\n--- Testing Exact Hand-off ---")
    exact_ids = hybrid_player_search("Alex Chipciu", search_mode="exact")
    print(exact_ids)
    # Output: [{'player_id': 'alex_chipciu_u_cluj', 'score': 0.0}]

    print("\n--- Testing Semantic Hand-off ---")
    semantic_ids = hybrid_player_search("aggressive enforcer midfielder", search_mode="semantic", top_k=3)
    print(semantic_ids)
    # Output: [{'player_id': 'player_A', 'score': 0.85}, {'player_id': 'player_B', 'score': 0.79}]

    if exact_ids:
        print("\n--- Testing Recommendations Hand-off ---")
        similar_ids = get_similar_players(exact_ids[0]['player_id'], top_k=2)
        print(similar_ids)
        # Output: [{'player_id': 'player_C', 'score': 0.91}, {'player_id': 'player_D', 'score': 0.88}]
