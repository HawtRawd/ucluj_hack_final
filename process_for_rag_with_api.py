import pandas as pd
import json
import time # Added to prevent rate-limit crashes
from google import genai
from google.genai import types

# Initialize the Gemini Client
# Make sure to get a free API key from Google AI Studio
client = genai.Client("here goes key")

def generate_ai_profile(player_name, stats_dict):
    """Hits Gemini 2.5 Flash to generate a specific playstyle profile."""
    
    prompt = f"""
    You are an elite football scout analyzing the Romanian Superliga. 
    Analyze this player: {player_name}.
    Here are their per-90 stats: {json.dumps(stats_dict)}
    
    Write a 3-sentence scouting report focusing ONLY on their playstyle, mentality, and tactical tendencies based on these numbers. Do not just list the numbers. Tell me who they are as a player.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a professional football data scout.",
                temperature=0.4, # Keeps it analytical and grounded
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating profile for {player_name}: {e}")
        return "Balanced playstyle with standard tactical discipline." # Safe fallback

def process_superliga_database(csv_path):
    df = pd.read_csv(csv_path)
    
    # FILTER: Only process Superliga players to save time!
    df = df[df['league'] == 'ROU-Liga I'] 
    
    pinecone_payload = []
    
    for index, row in df.iterrows():
        name = row['player']
        minutes = row.get('Playing Time_Min', 0)
        
        # Skip benchwarmers to keep your DB elite
        if minutes < 200:
            continue
            
        print(f"Analyzing {name}...")
        
        # 1. Grab their key stats
        stats = {
            "Tackles Won": row.get('Performance_TklW', 0),
            "Fouls": row.get('Performance_Fls', 0),
            "Yellow Cards": row.get('Performance_CrdY', 0),
            "Goals": row.get('Performance_Gls', 0),
            "Assists": row.get('Performance_Ast', 0)
        }
        
        # 2. Let Gemini do the heavy lifting
        ai_scout_report = generate_ai_profile(name, stats)
        
        # 3. Build the RAG text
        rag_text = f"Player Profile: {name}. {ai_scout_report}"
        
        # 4. Save for Pinecone
        pinecone_payload.append({
            "id": str(name).replace(" ", "_").lower(),
            "text": rag_text,
            "metadata": {
                "name": name,
                "team": row['team'],
                "position": row['pos_']
            }
        })
        
        # 5. Pause briefly to avoid hitting Google's rate limits
        time.sleep(1) 
        
    # Save the final AI-enriched database!
    with open('superliga_ai_profiles.json', 'w') as f:
        json.dump(pinecone_payload, f, indent=4)
        
    print(f"Success! Generated {len(pinecone_payload)} AI profiles.")

# Execute
process_superliga_database('v3 ultimate_player_database_final.csv')
