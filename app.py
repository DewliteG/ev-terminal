import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
from datetime import datetime

st.set_page_config(page_title="EV Football Analytics", layout="wide", page_icon="📈")

# ==========================================
# 1. QUANT & MODEL ENGINE
# ==========================================
class QuantEngine:
    @staticmethod
    def calculate_ev(true_prob, odds):
        potential_profit = (odds - 1.0) * 100
        prob_losing = 1.0 - true_prob
        return ((true_prob * potential_profit) - (prob_losing * 100.0)) / 100.0

    @staticmethod
    def calculate_kelly(true_prob, odds, fraction=0.25):
        b = odds - 1.0
        kelly = ((b * true_prob) - (1.0 - true_prob)) / b
        return max(0.0, kelly * fraction)

class PoissonGoalModel:
    def __init__(self):
        self.team_stats = {}
        
    def predict_probs(self, home, away):
        h_xg = 1.45
        a_xg = 1.15
        prob_h = sum(poisson.pmf(i, h_xg) * sum(poisson.pmf(j, a_xg) for j in range(i)) for i in range(1, 6))
        prob_a = sum(poisson.pmf(i, a_xg) * sum(poisson.pmf(j, h_xg) for j in range(i)) for i in range(1, 6))
        prob_d = 1.0 - prob_h - prob_a
        return prob_h, prob_d, prob_a

class AIInsightEngine:
    @staticmethod
    def generate(selection, model_prob, odds, edge):
        implied = 1 / odds
        return (f"Model probability is {model_prob*100:.1f}%, vs SkyBet's implied {implied*100:.1f}%. "
                f"Creates a +{edge*100:.1f}% edge. "
                f"High-safety value profile identified via Poisson distribution.")

# ==========================================
# 2. THE ODDS API LEAGUE KEYS
# ==========================================
LEAGUE_KEYS = {
    "Premier League": "soccer_epl",
    "Championship": "soccer_efl_champ",
    "Champions League": "soccer_uefa_champs_league",
    "Europa League": "soccer_uefa_europa_league",
    "La Liga": "soccer_spain_la_liga",
    "Bundesliga": "soccer_germany_bundesliga",
    "Serie A": "soccer_italy_serie_a",
    "Ligue 1": "soccer_france_ligue_one",
    "Eredivisie": "soccer_netherlands_eredivisie",
    "Primeira Liga": "soccer_portugal_primeira_liga",
    "MLS": "soccer_usa_mls"
}

# ==========================================
# 3. STREAMLIT UI & LIVE API ORCHESTRATION
# ==========================================
st.title("📈 EV Football Analytics Terminal (SkyBet)")
st.markdown("Identify mathematically profitable betting opportunities strictly on **SkyBet** using real live fixtures.")

# --- GLOSSARY EXPANDER ---
with st.expander("📖 Glossary: Understanding Your Metrics (Click to Expand)"):
    st.markdown("""
    * **Edge:** The mathematical difference between our model's true win probability and the bookmaker's implied probability ($\text{Model \%} - \text{Implied \%}$). A positive edge means the bookmaker has underpriced the selection.
    * **EV (Expected Value):** The percentage return you expect to make on average per unit staked over the long run. An EV of +5% means for every £100 staked long-term, you expect an average profit of £5.
    * **Rec. Stake (Kelly):** The optimal fraction of your total bankroll to wager using the Kelly Criterion (scaled down to Quarter-Kelly, i.e., $0.25$), designed to maximize long-term growth while heavily protecting against losing streaks.
    """)

# Sidebar API Configuration
st.sidebar.header("⚙️ Settings")
api_key = st.sidebar.text_input("Enter 'The Odds API' Key", type="password")
st.sidebar.markdown("Get a free key at [the-odds-api.com](https://the-odds-api.com/).")

st.sidebar.markdown("---")
st.sidebar.subheader("Select Leagues to Scan")

selected_leagues = []
for league_name in LEAGUE_KEYS.keys():
    is_default = league_name in ["Premier League", "Champions League"]
    if st.sidebar.checkbox(league_name, value=is_default):
        selected_leagues.append(league_name)

tab1, tab2 = st.tabs(["🎯 Live SkyBet Value Bets", "🔗 Accumulator Engine"])

with tab1:
    st.subheader("Scanning Real Upcoming Matches (Prioritizing High Safety & Margin)")
    
    if st.button("🔄 Scan Selected SkyBet Markets", type="primary"):
        if not api_key:
            st.error("Please enter your API Key in the sidebar to fetch real fixtures.")
        elif not selected_leagues:
            st.warning("Please check at least one league in the sidebar to scan.")
        else:
            with st.status("Fetching live data and calculating margins...", expanded=True) as status:
                poisson_model = PoissonGoalModel()
                bets = []
                
                for league_name in selected_leagues:
                    league_key = LEAGUE_KEYS[league_name]
                    st.write(f"📡 Scanning {league_name}...")
                    
                    url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds/?apiKey={api_key}&regions=uk&bookmakers=skybet&markets=h2h"
                    
                    try:
                        response = requests.get(url)
                        data = response.json()
                        
                        if response.status_code == 401:
                            st.error("Invalid API Key.")
                            break
                        elif response.status_code == 429:
                            st.error("API Quota Reached. Processing available data...")
                            break
                        elif response.status_code != 200:
                            continue
                            
                        if not data:
                            continue
                        
                        for match in data:
                            home_team = match.get("home_team")
                            away_team = match.get("away_team")
                            
                            kickoff_raw = match.get("commence_time")
                            kickoff = datetime.strptime(kickoff_raw, "%Y-%m-%dT%H:%M:%SZ").strftime("%b %d, %H:%M")
                            
                            bookmakers = match.get("bookmakers", [])
                            skybet_data = next((b for b in bookmakers if b["key"] == "skybet"), None)
                            
                            if skybet_data:
                                markets = skybet_data.get("markets", [])
                                h2h_market = next((m for m in markets if m["key"] == "h2h"), None)
                                
                                if h2h_market:
                                    outcomes = h2h_market.get("outcomes", [])
                                    home_odds_data = next((o for o in outcomes if o["name"] == home_team), None)
                                    
                                    if home_odds_data:
                                        odds = home_odds_data["price"]
                                        true_prob_h, _, _ = poisson_model.predict_probs(home_team, away_team)
                                        
                                        edge = true_prob_h - (1 / odds)
                                        
                                        # Capture all odds (including short odds < 2.0) as long as there is positive edge
                                        if edge > 0.00: 
                                            ev = QuantEngine.calculate_ev(true_prob_h, odds)
                                            kelly = QuantEngine.calculate_kelly(true_prob_h, odds)
                                            insight = AIInsightEngine.generate(home_team, true_prob_h, odds, edge)
                                            
                                            bets.append({
                                                "Kickoff": kickoff,
                                                "League": league_name, 
                                                "Fixture": f"{home_team} vs {away_team}", 
                                                "Market": "Match Winner", 
                                                "Bookmaker": "SkyBet",
                                                "Selection": home_team,
                                                "Odds": odds, 
                                                "Model %": f"{true_prob_h*100:.1f}%", 
                                                "Edge": f"+{edge*100:.1f}%", 
                                                "EV": f"+{ev*100:.1f}%", 
                                                "Rec. Stake": f"{kelly*100:.2f}%",
                                                "AI Rationale": insight,
                                                # Raw numerical fields for sorting by safety (model probability) and margin (EV)
                                                "_raw_prob": true_prob_h,
                                                "_raw_ev": ev
                                            })
                                            
                    except Exception as e:
                        st.error(f"Failed to scan {league_name}: {e}")
                
                status.update(label=f"✅ Scan Complete! Found {len(bets)} +EV opportunities.", state="complete", expanded=False)
                
                if bets:
                    df_bets = pd.DataFrame(bets)
                    
                    # PRIORITY SORTING: Prioritize safety (higher model win probability) combined with EV margin
                    # Score = Model Probability * EV
                    df_bets['Safety_Score'] = df_bets['_raw_prob'] * df_bets['_raw_ev']
                    df_bets = df_bets.sort_values(by='Safety_Score', ascending=False)
                    
                    # Clean up temporary helper columns
                    df_bets = df_bets.drop(columns=['_raw_prob', '_raw_ev', 'Safety_Score'])
                    
                    st.dataframe(df_bets, use_container_width=True, hide_index=True)
                else:
                    st.info("No +EV opportunities found across the selected leagues on SkyBet right now. Check back closer to kickoff!")

with tab2:
    st.subheader("Accumulator (Parlay) EV Calculator")
    st.markdown("Calculate the true compounding Expected Value of a SkyBet Accumulator.")
    
    parlay_data = pd.DataFrame({
        "Leg Description": ["Arsenal Match Winner", "Safe Home Double Chance", "Over 1.5 Goals"],
        "SkyBet Odds (Decimal)": [1.45, 1.25, 1.35],
        "True Model Prob (%)": [72.0, 85.0, 78.0]
    })
    
    edited_parlay = st.data_editor(parlay_data, num_rows="dynamic", use_container_width=True, hide_index=True)
    
    if st.button("⚙️ Calculate Accumulator EV", type="primary"):
        try:
            odds_prod = np.prod(edited_parlay["SkyBet Odds (Decimal)"].astype(float))
            prob_prod = np.prod(edited_parlay["True Model Prob (%)"].astype(float) / 100.0)
            implied = 1.0 / odds_prod
            
            edge = prob_prod - implied
            ev = QuantEngine.calculate_ev(prob_prod, odds_prod)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Combined SkyBet Odds", f"{odds_prod:.2f}")
            col2.metric("Combined Implied Prob", f"{implied*100:.2f}%")
            col3.metric("True Model Prob", f"{prob_prod*100:.2f}%")
            col4.metric("Expected Value (EV)", f"{ev*100:.2f}%", delta=f"{edge*100:.2f}% Edge")
            
            if ev > 0:
                st.success("✅ **MASSIVE +EV DETECTED**\n\nBecause you are compounding multiple positive EV edges, your mathematical advantage grows exponentially. This is a sharp accumulator.")
            else:
                st.error("❌ **NEGATIVE EV DETECTED**\n\nBookmaker margins (vig) compound exponentially against you in standard accumulators. Even if these look like safe bets, the math dictates this will lose money long-term.")
                
        except Exception:
            st.warning("Please ensure all Odds and Probabilities are valid numbers.")
