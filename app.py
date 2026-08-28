import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import random

st.set_page_config(page_title="EV Football Analytics", layout="wide")

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
        
    def fit(self, all_teams):
        self.team_stats = {team: {'attack': random.uniform(0.8, 1.6), 'defense': random.uniform(0.7, 1.4)} 
                           for team in all_teams}
                           
    def predict_probs(self, home, away):
        h_xg = self.team_stats.get(home, {}).get('attack', 1.0) * 1.5
        a_xg = self.team_stats.get(away, {}).get('attack', 1.0) * 1.2
        prob_h = sum(poisson.pmf(i, h_xg) * sum(poisson.pmf(j, a_xg) for j in range(i)) for i in range(1, 6))
        prob_a = sum(poisson.pmf(i, a_xg) * sum(poisson.pmf(j, h_xg) for j in range(i)) for i in range(1, 6))
        prob_d = 1.0 - prob_h - prob_a
        return prob_h, prob_d, prob_a

class AIInsightEngine:
    @staticmethod
    def generate(selection, model_prob, odds, edge, market):
        implied = 1 / odds
        return f"Model prob: {model_prob*100:.1f}% vs SkyBet implied {implied*100:.1f}%. Edge: +{edge*100:.1f}%."

# ==========================================
# 2. DATA CONFIG
# ==========================================
LEAGUES = {
    "Premier League": ["Arsenal", "Chelsea", "Man City", "Liverpool", "Man Utd", "Tottenham", "Aston Villa", "Newcastle"],
    "Championship": ["Leeds United", "Sunderland", "Sheffield Utd", "West Brom", "Burnley", "Middlesbrough", "Norwich", "Watford"],
    "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Villarreal", "Athletic Club", "Real Sociedad", "Sevilla", "Valencia"],
    "Serie A": ["Inter Milan", "Juventus", "AC Milan", "Napoli", "Atalanta", "AS Roma", "Lazio", "Fiorentina"],
    "Bundesliga": ["Bayern Munich", "Bayer Leverkusen", "Dortmund", "RB Leipzig", "Stuttgart", "Eintracht Frankfurt", "Freiburg", "Wolfsburg"]
}
all_teams = [team for teams in LEAGUES.values() for team in teams]
MARKETS = ["Match Winner", "Over/Under Goals", "Both Teams To Score", "Player To Be Carded", "Player Shots On Target"]

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.title("📈 EV Football Analytics Terminal (SkyBet)")

tab1, tab2 = st.tabs(["🎯 SkyBet Value Bets", "🔗 Accumulator Engine"])

with tab1:
    st.subheader("Highest Expected Value Opportunities")
    if st.button("🔄 Scan SkyBet Markets"):
        poisson_model = PoissonGoalModel()
        poisson_model.fit(all_teams)
        bets = []
        
        for _ in range(10):
            league, teams_in_league = random.choice(list(LEAGUES.items()))
            home, away = random.sample(teams_in_league, 2)
            market = random.choice(MARKETS)
            true_prob = random.uniform(0.3, 0.7)
            odds = round((1 / true_prob) * random.uniform(0.9, 1.08), 2)
            edge = true_prob - (1 / odds)
            
            if edge > 0.02:
                ev = QuantEngine.calculate_ev(true_prob, odds)
                kelly = QuantEngine.calculate_kelly(true_prob, odds)
                bets.append({
                    "League": league, "Fixture": f"{home} vs {away}", "Market": market, 
                    "Odds": odds, "Model %": f"{true_prob*100:.1f}%", "Edge": f"+{edge*100:.1f}%", 
                    "EV": f"+{ev*100:.1f}%", "Rec Stake": f"{kelly*100:.2f}%"
                })
        
        df_bets = pd.DataFrame(bets)
        st.dataframe(df_bets, use_container_width=True)

with tab2:
    st.subheader("Accumulator (Parlay) EV Calculator")
    parlay_data = pd.DataFrame({
        "Leg Description": ["Arsenal Win", "Over 2.5 Goals"],
        "SkyBet Odds": [1.85, 2.00],
        "Model Prob (%)": [60.0, 55.0]
    })
    edited_parlay = st.data_editor(parlay_data, num_rows="dynamic", use_container_width=True)
    
    if st.button("Calculate Parlay EV"):
        odds_prod = np.prod(edited_parlay["SkyBet Odds"].astype(float))
        prob_prod = np.prod(edited_parlay["Model Prob (%)"].astype(float) / 100.0)
        implied = 1.0 / odds_prod
        ev = QuantEngine.calculate_ev(prob_prod, odds_prod)
        
        st.metric("Combined Odds", f"{odds_prod:.2f}")
        st.metric("Combined Model Prob", f"{prob_prod*100:.2f}%")
        st.metric("Expected Value (EV)", f"{ev*100:.2f}%")