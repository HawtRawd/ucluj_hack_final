import streamlit as st
import requests
import re
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import date, time

st.set_page_config(page_title="U Cluj Scout AI", layout="wide")

BACKEND_URL = "http://10.249.66.190:8000"
USE_BACKEND = True
SHOW_DEBUG = False
CSV_FILE = "v3 ultimate_player_database_final.csv"

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
    color: #e5e7eb;
}
h1, h2, h3, h4, p, span, label {
    color: #e5e7eb !important;
}
.block-container {
    padding-top: 3rem;
    max-width: 1400px;
}
[data-testid="stContainer"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 18px;
}
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
}
textarea, input {
    background-color: rgba(15, 23, 42, 0.85) !important;
    color: #f8fafc !important;
    border-radius: 14px !important;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 14px;
}
a {
    color: #60a5fa !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INIT -----------------
if "page" not in st.session_state:
    st.session_state["page"] = "scout"

# ----------------- NEW CODE LOGIC -----------------
fallback_players = [
    {
        "player_id": "p1",
        "player": "Alex Salom",
        "team": "Botoșani",
        "pos_": "MF,DF",
        "age_": 28,
        "Playing Time_Min": 672,
        "ai_match_score": 0.86,
    },
    {
        "player_id": "p2",
        "player": "R. Fotin",
        "team": "Botoșani",
        "pos_": "DF",
        "age_": 18,
        "Playing Time_Min": 310,
        "ai_match_score": 0.81,
    },
    {
        "player_id": "p3",
        "player": "G. Dumitru",
        "team": "Voluntari",
        "pos_": "DF",
        "age_": 20,
        "Playing Time_Min": 540,
        "ai_match_score": 0.75,
    },
]

def normalize_player(p, idx):
    raw = p.get("raw") if isinstance(p.get("raw"), dict) else p

    player_id = str(p.get("id") or p.get("player_id") or f"player_{idx}")

    age_from_backend = p.get("age")

    birth_date = str(
        p.get("birthDate")
        or raw.get("birthDate")
        or p.get("birth_date")
        or raw.get("birth_date")
        or ""
    )

    if len(birth_date) >= 4 and birth_date[:4].isdigit():
        age = date.today().year - int(birth_date[:4])
    elif age_from_backend not in [None, "", "N/A", "nan"]:
        age = age_from_backend
    else:
        age = "N/A"

    return {
        "id": player_id,
        "player_id": player_id,
        "name": p.get("name") or p.get("shortName") or raw.get("shortName") or "Unknown Player",
        "team": p.get("teamName") or p.get("team") or raw.get("teamName") or raw.get("team") or "Unknown",
        "position": p.get("position") or p.get("role.name") or raw.get("role.name") or "Unknown",
        "age": age,
        "minutes_played": p.get("minutes_played") or p.get("total_minutesOnField") or raw.get("total_minutesOnField") or 0,
        "goals": p.get("goals") if p.get("goals") is not None else raw.get("total_goals", "N/A"),
        "assists": p.get("assists") if p.get("assists") is not None else raw.get("total_assists", "N/A"),
        "yellow_cards": p.get("yellow_cards") if p.get("yellow_cards") is not None else raw.get("total_yellowCards", "N/A"),
        "text": p.get("text") or raw.get("ai_description") or "No AI description available.",
        "ai_match_score": p.get("ai_match_score") or p.get("score") or 0,
        "raw": raw,
    }

def normalize_players(players):
    return [normalize_player(p, i) for i, p in enumerate(players)]


def rag_search(query, position=None, max_age=None, foot=None):
    if not USE_BACKEND:
        return {
            "report": "Fallback report: these players match the requested profile.",
            "players": normalize_players(fallback_players),
        }

    payload = {"query": query}
    if position and position != "Any":
        payload["position"] = position
    if max_age and max_age < 50:
        payload["max_age"] = int(max_age)
    if foot and foot != "Any":
        payload["foot"] = foot

    response = requests.post(
        f"{BACKEND_URL}/api/rag-scout",
        json=payload,
        timeout=40,
    )

    if response.status_code != 200:
        raise Exception(f"{response.status_code} - {response.text}")

    data = response.json()

    if SHOW_DEBUG:
        st.write("RAW BACKEND RESPONSE:", data)

    return {
        "report": data.get("report", "No report returned."),
        "players": normalize_players(data.get("players", [])),
    }


def get_similar_players(player_id):
    if not USE_BACKEND:
        return normalize_players(fallback_players[:3])

    response = requests.get(
        f"{BACKEND_URL}/api/similar",
        params={"player_id": player_id},
        timeout=30,
    )

    if response.status_code != 200:
        raise Exception(f"{response.status_code} - {response.text}")

    data = response.json()
    return normalize_players(data.get("similar_players", []))


def get_player_news(player_name, player_id):
    if not USE_BACKEND:
        return {"name": player_name, "hype_score": 72, "summary": "...", "headlines": []}

    response = requests.get(
        f"{BACKEND_URL}/api/news",
        params={"name": player_name, "player_id": player_id},
        timeout=35,
    )

    if response.status_code != 200:
        raise Exception(f"{response.status_code} - {response.text}")

    return response.json()


def match_percent(score):
    score = float(score)
    if score <= 1:
        score = score * 100

    # boost mai mare pentru scoruri mici
    boosted = score + (100 - score) * 0.4
    return round(boosted)


def safe_value(value):
    if value in [None, "", "nan"]:
        return "N/A"
    return value

def clean_ai_text(player):
    ai_text = player.get("text", "No AI description available.")
    name = player.get("name", "")

    if ai_text.startswith("Player Profile:"):
        ai_text = ai_text.split(":", 1)[1].strip()

    if name and ai_text.startswith(name):
        ai_text = ai_text[len(name):].strip()

    return ai_text.lstrip(". ").strip()

def player_specific_overview(player, full_report):
    name = player["name"]
    raw = player["raw"]

    position = safe_value(player["position"])
    team = safe_value(player["team"])
    age = safe_value(player["age"])
    minutes = safe_value(player["minutes_played"])
    match = match_percent(player.get("ai_match_score", 0))

    goals = (
        raw.get("goals")
        if raw.get("goals") is not None
        else raw.get("total_goals")
        if raw.get("total_goals") is not None
        else raw.get("Performance_Gls")
        if raw.get("Performance_Gls") is not None
        else "N/A"
    )

    assists = (
        raw.get("assists")
        if raw.get("assists") is not None
        else raw.get("total_assists")
        if raw.get("total_assists") is not None
        else raw.get("Performance_Ast")
        if raw.get("Performance_Ast") is not None
        else "N/A"
    )

    cards = (
        raw.get("yellow_cards")
        if raw.get("yellow_cards") is not None
        else raw.get("total_yellowCards")
        if raw.get("total_yellowCards") is not None
        else raw.get("Performance_CrdY")
        if raw.get("Performance_CrdY") is not None
        else "N/A"
    )

    return f"""
**{name}** is a **{age}-year-old {position}** currently playing for **{team}**.

**Why he matches:** The player has an AI semantic match score of **{match}%**.

**Profile:** He has played **{minutes} minutes**, with **{goals} goals**, **{assists} assists**, and **{cards} yellow cards**.

**Mentality angle:** Based on the available indicators, this profile should be reviewed for tactical fit and reliability.
"""


def render_news_section(player):
    player_name = player["name"]
    player_id = player["id"]

    if "news_cache" not in st.session_state:
        st.session_state["news_cache"] = {}

    st.subheader("📰 Media & News Sentiment")

    if player_name not in st.session_state["news_cache"]:
        with st.spinner(f"Fetching news for {player_name}..."):
            try:
                st.session_state["news_cache"][player_name] = get_player_news(player_name, player_id)
            except Exception as e:
                st.session_state["news_cache"][player_name] = {"summary": f"News unavailable: {e}", "headlines": []}

    news = st.session_state["news_cache"][player_name]
    hype_score = news.get("hype_score")

    col_news_1, col_news_2 = st.columns([1, 3])

    with col_news_1:
        if hype_score is None:
            st.info("Hype Score: N/A")
        elif hype_score >= 70:
            st.success(f"Hype Score: {hype_score}/100")
        elif hype_score >= 40:
            st.warning(f"Hype Score: {hype_score}/100")
        else:
            st.error(f"Hype Score: {hype_score}/100")

    with col_news_2:
        st.write(news.get("summary", "No summary available."))

    headlines = news.get("headlines", [])

    if headlines:
        st.markdown("**Latest headlines:**")
        for item in headlines[:5]:
            title = item.get("title", "Untitled article")
            link = item.get("link", "")
            if link:
                st.markdown(f"- [{title}]({link})")
            else:
                st.markdown(f"- {title}")
    else:
        st.caption("No media headlines found for this player.")

    if st.button("🔄 Refresh news sentiment", key=f"refresh_news_{player['id']}"):
        with st.spinner("Refreshing news sentiment..."):
            try:
                st.session_state["news_cache"][player_name] = get_player_news(player_name)
                st.rerun()
            except Exception as e:
                st.error(f"Could not refresh news: {e}")


# ----------------- 3D DASHBOARD (CARRIED OVER & CLEANED) -----------------
@st.cache_data
def load_traditional_data():
    df = pd.read_csv(CSV_FILE)
    if "total_minutesOnField" in df.columns:
        df = df[df["total_minutesOnField"] >= 90].copy()
    return df


def render_traditional_dashboard():
    if st.button("⬅ Back to AI Scout"):
        st.session_state["page"] = "scout"
        st.rerun()

    st.markdown("""
    <h1 style='font-size: 42px;'>🧊 3D Player Scout Map</h1>
    <p style='font-size: 18px; color: #94a3b8 !important;'>
    Visualize the traditional database metrics mapped in 3-dimensional space.
    </p>
    """, unsafe_allow_html=True)

    try:
        df = load_traditional_data()
    except Exception as e:
        st.error(f"Could not load CSV file: {e}")
        st.info(f"Make sure `{CSV_FILE}` is in the same folder as app.py.")
        return

    if df.empty:
        st.warning("CSV loaded, but no players matched the minimum minutes filter.")
        return

    st.success(f"Loaded {len(df)} players from the traditional database.")

    stat_options = [
        col for col in df.columns
        if any(prefix in col for prefix in ["avg_", "total_", "true_pct_"])
    ]
    stat_options.sort()

    if len(stat_options) < 3:
        st.warning("Not enough stat columns found for 3D plot.")
    else:
        default_x = "avg_goals_per90" if "avg_goals_per90" in stat_options else stat_options[0]
        default_y = "avg_assists_per90" if "avg_assists_per90" in stat_options else stat_options[1]
        default_z = "avg_successfulDribbles_per90" if "avg_successfulDribbles_per90" in stat_options else stat_options[
            2]

        plot_col1, plot_col2, plot_col3 = st.columns(3)

        with plot_col1:
            x_axis = st.selectbox(
                "X Axis",
                stat_options,
                index=stat_options.index(default_x)
            )

        with plot_col2:
            y_axis = st.selectbox(
                "Y Axis",
                stat_options,
                index=stat_options.index(default_y)
            )

        with plot_col3:
            z_axis = st.selectbox(
                "Z Axis",
                stat_options,
                index=stat_options.index(default_z)
            )

        color_options = [c for c in ["role.name", "foot", "birthArea.name"] if c in df.columns]
        size_options = [c for c in ["total_minutesOnField", "height", "weight"] if c in df.columns]

        c_col, s_col = st.columns(2)

        with c_col:
            color_by = st.selectbox("Color by", color_options) if color_options else None

        with s_col:
            size_by = st.selectbox("Size by", size_options) if size_options else None

        hover_data = {}

        for col in ["role.name", "height", "weight", "birthDate", x_axis, y_axis, z_axis]:
            if col in df.columns:
                hover_data[col] = True

        hover_name = "shortName" if "shortName" in df.columns else None

        plot_df = df.copy()

        for col in [x_axis, y_axis, z_axis]:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

        plot_df = plot_df.dropna(subset=[x_axis, y_axis, z_axis])

        fig = px.scatter_3d(
            plot_df,
            x=x_axis,
            y=y_axis,
            z=z_axis,
            color=color_by,
            size=size_by,
            hover_name=hover_name,
            hover_data=hover_data,
            template="plotly_dark",
            height=800,
            opacity=0.75,
        )

        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)


# ----------------- MAIN AI SCOUT UI -----------------
def render_ai_scout_page():
    st.markdown("""
    <h1 style='font-size: 46px; margin-bottom: 0;'>⚽ U Cluj Scout AI</h1>
    <p style='font-size: 19px; color: #94a3b8 !important; margin-top: 8px;'>
    AI-powered scouting using semantic player search, Wyscout-style data, mentality reports and media sentiment.
    </p>
    """, unsafe_allow_html=True)

    top1, top2 = st.columns([3, 1])
    with top1:
        st.success("🚀 Powered by RAG + Pinecone + Vertex AI + News Sentiment")
    with top2:
        if st.button("🧊 3D Stats Dashboard"):
            st.session_state["page"] = "traditional"
            st.rerun()

    search_mode = st.radio(
        "Search mode",
        ["Profile search", "Player name search"],
        horizontal=True
    )

    placeholder = (
        "Ex: young defender with strong mentality and good minutes played"
        if search_mode == "Profile search"
        else "Ex: Francisco Petrasso"
    )

    query = st.text_area(
        "Search query",
        placeholder=placeholder,
        label_visibility="collapsed",
    )

    with st.expander("⚙️ Advanced Filters (Applied to AI Search)"):
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_pos = st.selectbox("Lane / Position", ["Any", "Forward", "Midfielder", "Defender", "Goalkeeper"])
        with col2:
            filter_age = st.number_input("Max Age", min_value=15, max_value=50, value=50)
        with col3:
            filter_foot = st.selectbox("Preferred Foot", ["Any", "left", "right", "both"])

    if st.button("🔍 Search players"):
        with st.spinner("Searching semantic player database..."):
            try:
                backend_query = f"player {query}" if search_mode == "Player name search" else query

                result = rag_search(
                    query=backend_query,
                    position=filter_pos,
                    max_age=filter_age,
                    foot=filter_foot
                )

                if search_mode == "Player name search" and query.strip():
                    exact_matches = [
                        p for p in result["players"]
                        if query.lower().strip() in p["name"].lower()
                    ]
                    if exact_matches:
                        result["players"] = exact_matches

                st.session_state["players"] = result["players"]
                st.session_state["report"] = result["report"]
                st.session_state["selected_player"] = None
                st.session_state["similar_players"] = []
                st.session_state["compare_mode"] = False
                st.session_state["comparison_ready"] = False

            except Exception as e:
                st.error(f"Backend error: {e}")
                st.session_state["players"] = normalize_players(fallback_players)
                st.session_state["report"] = "Backend unavailable. Showing fallback demo data."
                st.session_state["selected_player"] = None
                st.session_state["similar_players"] = []
                st.session_state["compare_mode"] = False
                st.session_state["comparison_ready"] = False

    st.divider()

    left, right = st.columns([2, 3])

    with left:
        st.subheader("Players")

        if "players" not in st.session_state:
            st.info("Run a search to see recommended players.")
        elif len(st.session_state["players"]) == 0:
            st.warning("No matching players found. Try a broader query.")
        else:
            for player in st.session_state["players"]:
                with st.container(border=True):
                    st.markdown(f"### {player['name']}")
                    st.write(
                        f"{safe_value(player['position'])} · "
                        f"{safe_value(player['age'])} years old"
                    )
                    st.caption(f"🏟️ Team: {safe_value(player['team'])}")
                    st.caption(f"Minutes played: {safe_value(player['minutes_played'])}")

                    percent = match_percent(player.get("ai_match_score", 0))
                    st.metric("AI Match", f"{percent}%")
                    st.progress(percent / 100)

                    if st.button("View profile", key=f"profile_{player['id']}"):
                        st.session_state["selected_player"] = player
                        st.session_state["similar_players"] = []
                        st.session_state["compare_mode"] = False
                        st.session_state["comparison_ready"] = False

    with right:
        player = st.session_state.get("selected_player")

        if not player:
            st.info("Select a player to see details →")
        else:
            st.header(f"👤 {player['name']}")

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Age", safe_value(player["age"]))
            p2.metric("Position", safe_value(player["position"]))
            p3.metric("Team", safe_value(player["team"]))
            p4.metric("Minutes", safe_value(player["minutes_played"]))

            # st.subheader("🎯 Why this player matched")
            # percent = match_percent(player.get("ai_match_score", 0))
            #
            # st.write(f"AI semantic match score: **{percent}%**")
            # st.progress(percent / 100)
            #
            # st.write("✔ Matched the natural-language scouting query")
            # st.write("✔ Profile was retrieved from the player knowledge base")
            # st.write("✔ Full stats were hydrated from the lookup dictionary")

            ai_text = player.get("text", "No AI description available.")

            # scoate "Player Profile:"
            if ai_text.startswith("Player Profile:"):
                ai_text = ai_text.split(":", 1)[1].strip()

            # scoate numele duplicat de la început (ex: "B. N'Kololo.")
            name = player["name"]

            if ai_text.startswith(name):
                ai_text = ai_text[len(name):].strip()

            # dacă mai rămâne punct la început, îl curățăm
            ai_text = ai_text.lstrip(". ").strip()

            st.info(ai_text)

            render_news_section(player)

            st.subheader("📊 Player Statistics")

            stats_data = {
                "Metric": [
                    "Minutes Played",
                    "Goals",
                    "Assists",
                    "Yellow Cards",
                    "Position",
                    "Age",
                    "Team",
                    "Foot",
                    "Height",
                    "Weight"
                ],
                "Value": [
                    player.get("minutes_played", "N/A"),
                    player.get("goals", "N/A"),
                    player.get("assists", "N/A"),
                    player.get("yellow_cards", "N/A"),
                    player.get("position", "N/A"),
                    player.get("age", "N/A"),
                    player.get("team", "N/A"),
                    player["raw"].get("foot", "N/A"),
                    player["raw"].get("height", "   N/A"),
                    player["raw"].get("weight", "N/A"),
                ]
            }

            import pandas as pd
            df_stats = pd.DataFrame(stats_data)

            st.table(df_stats)

            st.divider()

            if st.button("⚖️ Compare with another player"):
                st.session_state["compare_mode"] = True
                st.session_state["comparison_ready"] = False

            if st.session_state.get("compare_mode", False):
                available_players = [
                    p for p in st.session_state["players"]
                    if p["id"] != player["id"]
                ]

                if not available_players:
                    st.info("No other players available to compare.")
                else:
                    second_name = st.selectbox(
                        "Choose player to compare with",
                        [p["name"] for p in available_players]
                    )

                    second_player = next(
                        p for p in available_players
                        if p["name"] == second_name
                    )

                    custom_traits = st.text_area(
                        "Custom comparison traits",
                        placeholder="Ex: leadership, calm under pressure, team-first mentality..."
                    )

                    if st.button("Generate comparison"):
                        st.session_state["comparison_ready"] = True

                        if st.session_state.get("comparison_ready", False):
                            st.subheader("Comparison")

                            c1, c2 = st.columns(2)

                            with c1:
                                st.markdown(f"### {player['name']}")
                                st.caption(f"🏟️ {safe_value(player['team'])}")
                                st.metric("AI Match", f"{match_percent(player.get('ai_match_score', 0))}%")
                                st.metric("Age", safe_value(player["age"]))
                                st.metric("Minutes", safe_value(player["minutes_played"]))
                                st.metric("Goals", safe_value(player.get("goals")))
                                st.metric("Assists", safe_value(player.get("assists")))
                                st.metric("Yellow Cards", safe_value(player.get("yellow_cards")))

                                st.markdown("**AI Scouting Profile**")
                                st.info(clean_ai_text(player))

                            with c2:
                                st.markdown(f"### {second_player['name']}")
                                st.caption(f"🏟️ {safe_value(second_player['team'])}")
                                st.metric("AI Match", f"{match_percent(second_player.get('ai_match_score', 0))}%")
                                st.metric("Age", safe_value(second_player["age"]))
                                st.metric("Minutes", safe_value(second_player["minutes_played"]))
                                st.metric("Goals", safe_value(second_player.get("goals")))
                                st.metric("Assists", safe_value(second_player.get("assists")))
                                st.metric("Yellow Cards", safe_value(second_player.get("yellow_cards")))

                                st.markdown("**AI Scouting Profile**")
                                st.info(clean_ai_text(second_player))

                            score_a = match_percent(player.get("ai_match_score", 0))
                            score_b = match_percent(second_player.get("ai_match_score", 0))

                            st.subheader("Verdict")

                            if score_a > score_b:
                                st.success(f"🏆 {player['name']} is the better match for this search.")
                            elif score_b > score_a:
                                st.success(f"🏆 {second_player['name']} is the better match for this search.")
                            else:
                                st.info("Both players are similarly matched.")
                        if custom_traits.strip():
                            st.info(f"Custom traits considered: {custom_traits}")

            st.divider()

            if st.button("🔎 Find similar players"):
                with st.spinner("Finding similar players..."):
                    try:
                        similar = get_similar_players(player["player_id"])
                        st.session_state["similar_players"] = similar
                    except Exception as e:
                        st.error(f"Similar players error: {e}")
                        st.session_state["similar_players"] = []

            if st.session_state.get("similar_players"):
                st.subheader("Similar Players")

                for sp in st.session_state["similar_players"]:
                    with st.container(border=True):
                        st.markdown(f"### {sp['name']}")
                        st.write(
                            f"{safe_value(sp['position'])} · "
                            f"{safe_value(sp['age'])} years old"
                        )
                        st.caption(f"🏟️ Team: {safe_value(sp['team'])}")
                        st.caption(f"Minutes played: {safe_value(sp['minutes_played'])}")

                        sim = match_percent(sp.get("ai_match_score", 0))
                        st.metric("Similarity", f"{sim}%")
                        st.progress(sim / 100)
                        st.markdown("**AI Scouting Profile**")
                        st.info(clean_ai_text(sp))


# ----------------- ROUTING -----------------
if st.session_state["page"] == "traditional":
    render_traditional_dashboard()
else:
    render_ai_scout_page()
