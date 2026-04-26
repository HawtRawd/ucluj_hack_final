import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from google import genai
from google.genai import types
import feedparser
import urllib.parse
from fastapi import BackgroundTasks
import requests
from typing import Optional

# --- IMPORT THE AI EXPERT'S ENGINE ---
import vector_search

news_cache = {}
CACHE_TTL = 300

app = FastAPI(title="AI Scouting Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GOOGLE CLOUD VERTEX AI SETUP ---
client = genai.Client("here goes key")

# --- LOAD AND MERGE THE DATABASES ---
print("Loading the Databases...")
lookup_dict = {}

try:
    # 1. Load the raw stats (v3 database)
    with open('v6 updated_player_database.json', 'r', encoding='utf-8') as f:
        v3_raw = json.load(f)

        # [THE FIX] Convert the v3 list into a dictionary so we can search it instantly
        if isinstance(v3_raw, list):
            for player in v3_raw:
                # 1. Get the raw name from the v3 stats
                raw_name = player.get("shortName", "")

                if raw_name:
                    # 2. Format it to match "d_bota" exactly
                    pid = raw_name.lower().replace(". ", "_").replace(".", "").replace(" ", "_")
                    lookup_dict[pid] = player
        else:
            lookup_dict = v3_raw  # Fallback just in case it's already a dictionary

    print(f"✅ Loaded {len(lookup_dict)} players from v3 stats database!")


    # 2. Load the AI text descriptions (Superliga AI Profiles)
    with open('superliga_ai_profiles.json', 'r', encoding='utf-8') as f:
        ai_profiles_list = json.load(f)

        merged_count = 0
        # 3. Inject the AI text directly into the stats dictionary
        for profile in ai_profiles_list:
            player_id = profile["id"]

            # If the player exists in the v3 stats, glue the text to them
            if player_id in lookup_dict:
                lookup_dict[player_id]["ai_description"] = profile["text"]
                merged_count += 1
            else:
                # If they are in the AI file but NOT in v3, add them anyway
                lookup_dict[player_id] = profile

    print(f"✅ Successfully fused AI descriptions into {merged_count} v3 stat profiles!")

    # --- ADD THIS TO SAVE THE RESULTING FILE ---
    print("Saving merged database to file...")
    with open('merged_ultimate_database.json', 'w', encoding='utf-8') as outfile:
        json.dump(lookup_dict, outfile, indent=4, ensure_ascii=False)
    print("✅ Saved to 'merged_ultimate_database.json'!")

except Exception as e:
    print(f"❌ FAILED TO LOAD DATABASES: {e}")


# --- REQUEST MODELS ---
class ScoutRequest(BaseModel):
    query: str
    position: Optional[str] = None
    max_age: Optional[int] = None
    foot: Optional[str] = None


# --- THE API ENDPOINTS ---

@app.post("/api/rag-scout")
def rag_scout_players(request: ScoutRequest):
    user_query = request.query

    # 1. Ask the expert's code for the matching IDs (fetch top 20 to allow room for filtering)
    pinecone_results = vector_search.hybrid_player_search(user_query, search_mode="semantic", top_k=20)

    if not pinecone_results:
        return {"report": "No matching players found in the database.", "players": []}

    hydrated_players = []

    for match in pinecone_results:
        player_id = match['player_id']
        match_score = match['score']

        if player_id in lookup_dict:
            raw_data = lookup_dict[player_id]

            # Grab variables for filtering
            p_pos = str(raw_data.get("role.name", "")).lower()
            birth_date = str(raw_data.get("birthDate", ""))
            p_age = 2026 - int(birth_date[:4]) if len(birth_date) >= 4 and birth_date[:4].isdigit() else 0
            p_foot = str(raw_data.get("foot", "")).lower()

            # --- APPLY FILTERS ---
            if request.position and request.position != "Any":
                if request.position.lower() not in p_pos:
                    continue
            if request.max_age and p_age > request.max_age:
                continue
            if request.foot and request.foot != "Any":
                if request.foot.lower() != p_foot:
                    continue

            # --- FORMAT FOR UI ---
            formatted_player = {
                "raw": raw_data,
                "id": player_id,
                "player_id": player_id,
                "name": raw_data.get("shortName", raw_data.get("name", "Unknown")),
                "position": raw_data.get("role.name", "Unknown"),  # Top-level position fix
                "age": p_age if p_age > 0 else "N/A",
                "team": raw_data.get("currentTeamId", "Unknown"),
                "minutes_played": raw_data.get("total_minutesOnField", 0),
                "goals": raw_data.get("total_goals", 0),
                "assists": raw_data.get("total_assists", 0),
                "yellow_cards": raw_data.get("total_yellowCards", 0),
                "text": raw_data.get("ai_description", "No AI description available."),
                "ai_match_score": match_score,
            }

            hydrated_players.append(formatted_player)

            # Limit the final returned list to the top 5 filtered results
            if len(hydrated_players) >= 5:
                break

    return {
        "report": "Successfully retrieved pre-generated player profiles based on your query.",
        "players": hydrated_players
    }


def update_pinecone_background(player_id: str, name: str, hype_score: int, summary: str):
    """This runs silently AFTER the user already got their news results!"""
    try:
        stats_str = ""
        if player_id in lookup_dict:
            # Grab just a few base stats to keep the vector lean
            base_data = {
                "Position": lookup_dict[player_id].get("role.name", ""),
                "Minutes": lookup_dict[player_id].get("total_minutesOnField", 0)
            }
            stats_str = f"Base Stats: {json.dumps(base_data)}."

        updated_rag_text = f"Player Profile: {name}. {stats_str} Latest Media Sentiment & Form: {summary} (Hype Score: {hype_score}/100)."

        new_vector = vector_search.model.encode(updated_rag_text).tolist()

        vector_search.index.upsert(
            vectors=[{
                "id": player_id,
                "values": new_vector,
                "metadata": {
                    "name": name,
                    "hype_score": hype_score,
                    "latest_news": summary
                }
            }]
        )
        print(f"✅ [BACKGROUND] Pinecone updated for {name}!")
    except Exception as e:
        print(f"❌ [BACKGROUND] Failed to update Pinecone: {e}")

@app.get("/api/similar")
def find_similar_players(player_id: str):
    similar_ids = vector_search.get_similar_players(player_id, top_k=3)
    similar_players = []

    for match in similar_ids:
        pid = match['player_id']
        if pid in lookup_dict:
            raw_data = lookup_dict[pid]

            # Calculate Age safely so the UI receives it properly
            birth_date = str(raw_data.get("birthDate", ""))
            p_age = 2026 - int(birth_date[:4]) if len(birth_date) >= 4 and birth_date[:4].isdigit() else 0

            # Formatted IDENTICALLY to /rag-scout for frontend consistency
            formatted_player = {
                "raw": raw_data,
                "id": pid,
                "player_id": pid,
                "name": raw_data.get("shortName", raw_data.get("name", "Unknown")),
                "position": raw_data.get("role.name", "Unknown"),
                "age": p_age if p_age > 0 else "N/A",
                "team": raw_data.get("currentTeamId", "Unknown"),
                "minutes_played": raw_data.get("total_minutesOnField", 0),
                "goals": raw_data.get("total_goals", 0),
                "assists": raw_data.get("total_assists", 0),
                "yellow_cards": raw_data.get("total_yellowCards", 0),
                "text": raw_data.get("ai_description", "No AI description available."),
                "ai_match_score": match['score'], # Map Pinecone's score here so the UI catches it
            }
            similar_players.append(formatted_player)

    return {"similar_players": similar_players}


@app.get("/api/news")
def get_player_news_score(name: str, player_id: str, background_tasks: BackgroundTasks):
    current_time = time.time()

    if name in news_cache:
        cached_entry = news_cache[name]
        if current_time - cached_entry["timestamp"] < CACHE_TTL:
            print(f"⚡ [CACHE HIT] Instantly returning saved news for {name}!")
            return cached_entry["data"]

    def fetch_rss(query_string):
        safe_query = urllib.parse.quote(query_string)
        rss_url = f"https://news.google.com/rss/search?q={safe_query}&hl=ro&gl=RO&ceid=RO:ro"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml'
        }
        try:
            print(f"🌐 Fetching RSS: {rss_url}")
            rss_response = requests.get(rss_url, timeout=5, headers=headers)

            # --- DEBUGGING PRINT ---
            print(f"📡 RSS Status Code: {rss_response.status_code}")

            # If Google hits you with a 429, you know you're rate-limited
            if rss_response.status_code != 200:
                print(f"❌ RSS Blocked/Failed. Text: {rss_response.text[:100]}")
                return None

            feed = feedparser.parse(rss_response.content)
            print(f"✅ Found {len(feed.entries)} articles for query: {query_string}")
            return feed
        except Exception as e:
            print(f"❌ RSS Fetch Error: {e}")
            return None

    targeted_ro_query = f'"{name}" fotbal'
    print(f"DEBUG: Attempting to fetch news for: {targeted_ro_query}")

    # Try just the name and "fotbal" first
    feed = fetch_rss(targeted_ro_query)
    is_recent_news = True

    # Fallback: Just the player's name
    if not feed or not feed.entries:
        print("⚠️ No results for targeted query. Trying fallback just with name...")
        feed = fetch_rss(f'"{name}"')
        is_recent_news = False

    if not feed or not feed.entries:
        print(f"🚨 ABSOLUTELY NO NEWS FOUND FOR {name}.")
        return {"name": name, "hype_score": 50, "summary": "No media presence found.", "headlines": []}

    sorted_entries = sorted(
        feed.entries,
        key=lambda x: x.published_parsed if hasattr(x, 'published_parsed') and x.published_parsed else (9999,),
        reverse=True
    )

    headlines_data = [{"title": entry.title, "link": entry.link} for entry in sorted_entries[:5]]
    just_titles = [item["title"] for item in headlines_data]
    headlines_text = "\n- ".join(just_titles)

    # --- DEBUGGING PRINT ---
    print(f"🧠 Sending these headlines to Gemini:\n{headlines_text}")
    time_context = "These are recent headlines from the last two weeks reflecting current form." if is_recent_news else "These are older/historical headlines, as this player has not been in the news recently."

    prompt = f"""
            You are an elite football scout and media sentiment analyst. Your task is to analyze the following news headlines for the player "{name}".

            HEADLINES:
            {headlines_text if headlines_text.strip() else "NO HEADLINES AVAILABLE"}

            TIME CONTEXT: {time_context}

            INSTRUCTIONS:
            1. Analyze the sentiment, form, and narrative surrounding the player. 
            2. FLEXIBLE ANALYSIS: If the headlines are mostly generic team match results rather than specific player news, do NOT reject them. Instead, infer a baseline summary based on the team's activity (e.g., "Currently active with their squad in recent fixtures, maintaining a stable presence."). 
            3. NO DATA FALLBACK: Only if the headlines section literally says "NO HEADLINES AVAILABLE", set the score to 50 and the summary to: "Insufficient media coverage to determine a meaningful sentiment or form."
            4. PROFESSIONAL TONE: Write the summary like a professional scouting report. Do NOT use filler meta-phrases like "The media consensus is..." or "These articles suggest...". State the facts directly.
            5. Keep the summary to 1-2 punchy, insightful sentences.

            Return a strict JSON object with exactly two keys:
            - "score": An integer from 1 to 100 representing the positive hype or form (100 = exceptional form/high praise, 10 = heavy criticism, 50 = neutral, stable, or mostly team-based news).
            - "summary": The insightful, direct summary of the media narrative.
            """

    try:
        # Reverted model back to gemini-2.0-flash (or 1.5-flash) to prevent API crashes.
        response = client.models.generate_content(
            model='gemini-2.5-flash',  # <--- The safest, most stable string
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        ai_analysis = json.loads(response.text)
        hype_score = ai_analysis.get("score", 50)
        summary = ai_analysis.get("summary", "Analysis complete.")

        # Pass the SAFE player_id to the background task
        background_tasks.add_task(update_pinecone_background, player_id, name, hype_score, summary)

        final_result = {
            "name": name,
            "hype_score": hype_score,
            "summary": summary,
            "headlines": headlines_data
        }

        news_cache[name] = {
            "timestamp": current_time,
            "data": final_result
        }

        return final_result

    except Exception as e:
        print(f"❌ AI ERROR for {name}: {str(e)}")
        return {"name": name, "hype_score": 50, "summary": "AI Error.", "headlines": headlines_data}


@app.get("/api/filter")
def filter_players_database(
        position: Optional[str] = None,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        min_height: Optional[int] = None,
        max_height: Optional[int] = None,
        foot: Optional[str] = None
):
    filtered_results = []

    for player_id, stats in lookup_dict.items():
        p_pos = str(stats.get("role.name", "")).lower()

        # Calculate Age safely from birthDate (e.g. '1998-05-12')
        birth_date = str(stats.get("birthDate", ""))
        p_age = 2026 - int(birth_date[:4]) if len(birth_date) >= 4 and birth_date[:4].isdigit() else 0

        p_height = int(stats.get("height", 0)) if str(stats.get("height", 0)).isdigit() else 0
        p_foot = str(stats.get("foot", "")).lower()

        if position and position.lower() not in p_pos:
            continue
        if min_age and p_age < min_age:
            continue
        if max_age and p_age > max_age:
            continue
        if min_height and p_height < min_height:
            continue
        if max_height and p_height > max_height:
            continue
        if foot and foot.lower() != p_foot:
            continue

        filtered_results.append(stats)

    return {
        "total_matches": len(filtered_results),
        "players": filtered_results
    }


def is_position_compatible(target_pos: str, candidate_pos: str) -> bool:
    """
    Evaluates if a candidate's position is a reasonable match for the target.
    Prevents goalkeepers/defenders from bleeding into midfield/attack searches.
    """
    target = target_pos.lower()
    candidate = candidate_pos.lower()

    # 1. Goalkeepers are strictly isolated
    if "goalkeeper" in target or "gk" in target:
        return "goalkeeper" in candidate or "gk" in candidate
    if "goalkeeper" in candidate or "gk" in candidate:
        return False

    # 2. Defenders are mostly isolated from attackers
    is_target_def = "def" in target or "back" in target
    is_candidate_def = "def" in candidate or "back" in candidate

    if is_target_def:
        # If looking for a defender, allow other defenders (or defensive mids, but kept simple here)
        return is_candidate_def
    if is_candidate_def:
        # If target is NOT a defender, don't suggest a defender
        return False

        # 3. Midfielders and Attackers (Wingers, Flanks, Strikers)
    # Since the user specifically requested allowing mid players to be found for flanks:
    # If we reached this point, neither player is a GK or Defender. They are safe to mix.
    return True


@app.get("/api/search")
def search_players_by_name_and_similar(name: str):
    """
    1. Finds the specific player by name.
    2. Grabs their AI Description and removes the first 7 words (to drop the name).
    3. Uses vector search to find similar players based on the cleaned description.
    4. Filters out incompatible positions.
    """
    if not name:
        return {"report": "No name provided.", "players": []}

    search_query = name.lower().strip()
    target_player_raw = None
    target_player_id = None

    # --- STEP 1: FIND THE TARGET PLAYER IN THE DB ---
    for pid, stats in lookup_dict.items():
        short_name = str(stats.get("shortName", "")).lower()
        full_name = str(stats.get("name", "")).lower()

        if search_query in short_name or search_query in full_name or search_query in pid:
            target_player_raw = stats
            target_player_id = pid
            break  # Grab the first solid match and stop

    if not target_player_raw:
        return {"report": f"Could not find a player matching '{name}'.", "players": []}

    # Helper function to format players for the UI
    def format_player(raw_data, p_id, match_score):
        birth_date = str(raw_data.get("birthDate", ""))
        p_age = 2026 - int(birth_date[:4]) if len(birth_date) >= 4 and birth_date[:4].isdigit() else 0

        return {
            "raw": raw_data,
            "id": p_id,
            "player_id": p_id,
            "name": raw_data.get("shortName", raw_data.get("name", "Unknown")),
            "position": raw_data.get("role.name", "Unknown"),
            "age": p_age if p_age > 0 else "N/A",
            "team": raw_data.get("currentTeamId", "Unknown"),
            "minutes_played": raw_data.get("total_minutesOnField", 0),
            "goals": raw_data.get("total_goals", 0),
            "assists": raw_data.get("total_assists", 0),
            "yellow_cards": raw_data.get("total_yellowCards", 0),
            "text": raw_data.get("ai_description", "No AI description available."),
            "ai_match_score": match_score,
        }

    final_results = [format_player(target_player_raw, target_player_id, 1.0)]

    # --- STEP 2: CLEAN TEXT & PERFORM RAG LOOKUP ---
    raw_ai_text = target_player_raw.get("ai_description", "")
    target_position = str(target_player_raw.get("role.name", "")).lower()

    if raw_ai_text:
        # [THE FIX]: Split into words, drop the first 7, and rejoin.
        # (Fallback to the original text if it's strangely short).
        words = raw_ai_text.split()
        clean_ai_text = " ".join(words[7:]) if len(words) > 7 else raw_ai_text

        # Use the CLEANED text for the vector search
        pinecone_results = vector_search.hybrid_player_search(clean_ai_text, search_mode="semantic", top_k=20)

        for match in pinecone_results:
            candidate_id = match['player_id']

            if candidate_id == target_player_id:
                continue

            if candidate_id in lookup_dict:
                candidate_raw = lookup_dict[candidate_id]
                candidate_pos = str(candidate_raw.get("role.name", "")).lower()

                # --- STEP 3: FILTER OUT INCOMPATIBLE POSITIONS ---
                if not is_position_compatible(target_position, candidate_pos):
                    continue

                final_results.append(format_player(candidate_raw, candidate_id, match['score']))

                if len(final_results) >= 6:
                    break

    return {
        "report": f"Found {target_player_raw.get('shortName')} and filtered for tactically similar profiles.",
        "total_matches": len(final_results),
        "players": final_results
    }
