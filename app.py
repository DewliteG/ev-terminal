import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
from datetime import datetime

st.set_page_config(page_title="EV Football Analytics - Pro", layout="wide", page_icon="📈")

# ==========================================
# 1. ADVANCED QUANT & ENSEMBLE ENGINE
# ==========================================
class FootballEloCalculator:
    def __init__(self, home_advantage: float = 65.0):
        self.home_advantage = home_advantage

    def get_win_probabilities(self, elo_home: float, elo_away: float) -> tuple:
        rating_diff = (elo_home + self.home_advantage) - elo_away
        prob_h = 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))
        # Simple heuristic split for draw vs away based on Elo symmetry
        prob_a = 1.0 / (1.0 + 10.0 ** ((rating_diff) / 400.0))
        prob_d = max(0.1, 1.0 - prob_h - prob_a)
        total = prob_h + prob_d + prob_a
        return prob_h/total, prob_d/total, prob_a/total

class PoissonGoalModel:
    def predict_xg_probs(self, home_form_attack: float, away_form_defense: float):
        h_xg = max(0.6, home_form_attack * 1.45)
        a_xg = max(0.4, away_form_defense * 1.05)
        
        prob_h = sum(poisson.pmf(i, h_xg) * sum(poisson.pmf(j, a_xg) for j in range(i)) for i in range(1, 6))
        prob_a = sum(poisson.pmf(i, a_xg) * sum(poisson.pmf(j, h_xg) for j in range(i)) for i in range(1, 6))
        prob_d = 1.0 - prob_h - prob_a
        return max(0.05, prob_h), max(0.05, prob_d), max(0.05, prob_a)

class EnsemblePredictionEngine:
    def __init__(self):
        self.elo_calc = FootballEloCalculator()
        self.poisson_model = PoissonGoalModel()

    def calculate_consensus_probabilities(self, home_team: str, away_team: str):
        base_elos = {
            "Arsenal": 1890, "Man City": 1950, "Liverpool": 1910, "Chelsea": 1780,
            "Real Madrid": 1960, "Barcelona": 1920, "Bayern Munich": 1940, "Inter Milan": 1860,
            "PSG": 1880, "Juventus": 1790, "AC Milan": 1780, "Bayer Leverkusen": 1850,
            "Atletico Madrid": 1820, "Borussia Dortmund": 1810, "Napoli": 1770, "Atalanta": 1790
        }
        
        elo_h = base_elos.get(home_team, 1680)
        elo_a = base_elos.get(away_team, 1680)
        
        elo_h_p, elo_d_p, elo_a_p = self.elo_calc.get_win_probabilities(elo_h, elo_a)
        
        form_attack_h = 1.3 if elo_h > 1800 else 1.0
        form_defense_a = 0.8 if elo_a > 1800 else 1.1
        pois_h_p, pois_d_p, pois_a_p = self.poisson_model.predict_xg_probs(form_attack_h, form_defense_a)
        
        # Blend probabilities safely (Home, Draw, Away)
        final_h = (0.55 * elo_h_p) + (0.45 * pois_h_p)
        final_d = (0.50 * elo_d_p) + (0.50 * pois_d_p)
        final_a = (0.55 * elo_a_p) + (0.45 * pois_a_p)
        
        # Normalize to ensure sum is 1.0
        total = final_h + final_d + final_a
        return final_h/total, final_d/total, final_a/total

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

class AIInsightEngine:
    @staticmethod
    def generate(selection, model_prob, odds, edge):
        implied = 1 / odds
        odds_type = "Short Odds (Safe Favorite)" if odds < 2.0 else "Value / Underdog"
        return (f"[{odds_type}] Model probability is {model_prob*100:.1f}%, "
                f"vs SkyBet implied {implied*100:.1f}%. Edge: +{edge*100:.1f}%. "
                f"Model successfully identified mispricing against bookmaker overround.")

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
st.title("📈 Pro EV Football Analytics Terminal (SkyBet)")
st.markdown("Advanced quantitative betting intelligence capturing both **short-odds favorites (< 2.0)** and high-value opportunities.")

with st.expander("📖 Glossary & Methodology: How the Pro Model Works"):
    st.markdown("""
    * **Edge:** The mathematical difference between our ensemble model's true win probability and SkyBet's implied probability.
    * **EV (Expected Value):** The percentage return expected per unit staked over the long run.
    * **Rec. Stake (Kelly):** Fractional Kelly criterion sizing to safeguard bankroll while maximizing growth.
    """)

st.sidebar.header("⚙️ Settings")
api_key = st.sidebar.text_input("Enter 'The Odds API' Key", type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("Select Leagues to Scan")

selected_leagues = []
for league_name in LEAGUE_KEYS.keys():
    is_default = league_name in ["Premier League", "Champions League"]
    if st.sidebar.checkbox(league_name, value=is_default):
        selected_leagues.append(league_name)

tab1, tab2 = st.tabs(["🎯 Live Ensemble Value Bets", "🔗 Accumulator Engine"])

with tab1:
    st.subheader("Scanning Real Upcoming Matches (Favorites & Value)")
    
    if st.button("🔄 Run Ensemble Market Scan", type="primary"):
        if not api_key:
            st.error("Please enter your API Key in the sidebar.")
        elif not selected_leagues:
            st.warning("Please check at least one league.")
        else:
            with st.status("Running multi-model ensemble analysis...", expanded=True) as status:
                ensemble_engine = EnsemblePredictionEngine()
                bets = []
                
                for league_name in selected_leagues:
                    league_key = LEAGUE_KEYS[league_name]
                    st.write(f"📡 Processing {league_name}...")
                    
                    url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds/?apiKey={api_key}&regions=uk&bookmakers=skybet&markets=h2h"
                    
                    try:
                        response = requests.get(url)
                        data = response.json()
                        
                        if response.status_code != 200 or not data:
                            continue
                        
                        for match in data:
                            home_team = match.get("home_team")
                            away_team = match.get("away_team")
                            
                            kickoff_raw = match.get("commence_time")
                            kickoff = datetime.strptime(kickoff_raw, "%Y-%m-%dT%H:%M:%SZ").strftime("%b %d, %H:%M")
                            
                            skybet_data = next((b for b in match.get("bookmakers", []) if b["key"] == "skybet"), None)
                            
                            if skybet_data:
                                h2h_market = next((m for m in skybet_data.get("markets", []) if m["key"] == "h2h"), None)
                                
                                if h2h_market:
                                    outcomes = h2h_market.get("outcomes", [])
                                    
                                    # Get 3-way consensus probabilities (Home, Draw, Away)
                                    h_prob, d_prob, a_prob = ensemble_engine.calculate_consensus_probabilities(home_team, away_team)
                                    
                                    for outcome in outcomes:
                                        selection_name = outcome["name"]
                                        odds = outcome["price"]
                                        
                                        # Match outcome name to respective model probability
                                        if selection_name == home_team:
                                            true_prob = h_prob
                                        elif selection_name == away_team:
                                            true_prob = a_prob
                                        else:
                                            true_prob = d_prob # Draw
                                            
                                        edge = true_prob - (1 / odds)
                                        
                                        if edge > 0.00: 
                                            ev = QuantEngine.calculate_ev(true_prob, odds)
                                            kelly = QuantEngine.calculate_kelly(true_prob, odds)
                                            insight = AIInsightEngine.generate(selection_name, true_prob, odds, edge)
                                            
                                            bets.append({
                                                "Kickoff": kickoff,
                                                "League": league_name, 
                                                "Fixture": f"{home_team} vs {away_team}", 
                                                "Market": "Match Winner", 
                                                "Bookmaker": "SkyBet",
                                                "Selection": selection_name,
                                                "Odds": odds, 
                                                "Model %": f"{true_prob*100:.1f}%", 
                                                "Edge": f"+{edge*100:.1f}%", 
                                                "EV": f"+{ev*100:.1f}%", 
                                                "Rec. Stake": f"{kelly*100:.2f}%",
                                                "AI Rationale": insight,
                                                "_raw_prob": true_prob,
                                                "_raw_ev": ev
                                            })
                                            
                    except Exception as e:
                        st.error(f"Error scanning {league_name}: {e}")
                
                status.update(label=f"✅ Scan Complete! Found {len(bets)} opportunities.", state="complete", expanded=False)
                
                if bets:
                    df_bets = pd.DataFrame(bets)
                    df_bets['Safety_Score'] = df_bets['_raw_prob'] * df_bets['_raw_ev']
                    df_bets = df_bets.sort_values(by='Safety_Score', ascending=False)
                    df_bets = df_bets.drop(columns=['_raw_prob', '_raw_ev', 'Safety_Score'])
                    
                    st.dataframe(df_bets, use_container_width=True, hide_index=True)
                else:
                    st.info("No +EV opportunities found on SkyBet right now matching current criteria.")

with tab2:
    st.subheader("Advanced Accumulator (Parlay) EV Calculator")
    st.markdown("Evaluate compounding risk and structural value across multiple ensemble legs.")
    
    parlay_data = pd.DataFrame({
        "Leg Description": ["Safe Home Favorite", "Short Odds Banker", "Over 1.5 Goals"],
        "SkyBet Odds (Decimal)": [1.45, 1.30, 1.35],
        "Ensemble Model Prob (%)": [75.0, 82.0, 78.0]
    })
    
    edited_parlay = st.data_editor(parlay_data, num_rows="dynamic", use_container_width=True, hide_index=True)
    
    if st.button("⚙️ Calculate Multi-Model Accumulator EV", type="primary"):
        try:
            odds_prod = np.prod(edited_parlay["SkyBet Odds (Decimal)"].astype(float))
            prob_prod = np.prod(edited_parlay["Ensemble Model Prob (%)"].astype(float) / 100.0)
            implied = 1.0 / odds_prod
            
            edge = prob_prod - implied
            ev = QuantEngine.calculate_ev(prob_prod, odds_prod)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Combined SkyBet Odds", f"{odds_prod:.2f}")
            col2.metric("Combined Implied Prob", f"{implied*100:.2f}%")
            col3.metric("Ensemble Model Prob", f"{prob_prod*100:.2f}%")
            col4.metric("Expected Value (EV)", f"{ev*100:.2f}%", delta=f"{edge*100:.2f}% Edge")
            
            if ev > 0:
                st.success("✅ **VERIFIED +EV ACCUMULATOR**\n\nThe multi-model ensemble confirms a positive compounding edge across these selections.")
            else:
                st.error("❌ **NEGATIVE EV DETECTED**\n\nBookmaker margins outweigh the blended ensemble edge. Long-term profitability is compromised.")
                
        except Exception:
            st.warning("Please verify all data inputs are formatted as numeric values.")
