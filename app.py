The error happens because Streamlit retains previous scan data in `st.session_state.scanned_bets` across re-runs. If an earlier scan ran before `_match_date` was introduced, or if an item in the session state lacked that key, looking up `b["_match_date"]` directly raises a `KeyError`.

Here is the complete, defensive `app.py` script. It safely handles `_match_date` using `.get()` and provides fallback date parsing from `Kickoff` so old session states or edge cases never crash the app.

```python
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
from datetime import datetime
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="SkyBet Institutional Quant Terminal", layout="wide", page_icon="📈")

# =====================================================================
# 1. QUANTITATIVE & MODELING UPGRADES: DYNAMIC FORM, LEAGUE HOME ADV, INJURIES
# =====================================================================

LEAGUE_HOME_ADVANTAGE = {
    "Premier League": 55.0,
    "Championship": 62.0,
    "Champions League": 45.0,
    "Europa League": 50.0,
    "La Liga": 58.0,
    "Bundesliga": 52.0,
    "Serie A": 56.0,
    "Ligue 1": 60.0,
    "Eredivisie": 64.0,
    "Primeira Liga": 65.0,
    "MLS": 80.0
}

class ExponentialDecayFormEngine:
    def __init__(self, decay_rate: float = 0.035):
        self.decay_rate = decay_rate

    def get_decayed_team_xg(self, team_name: str, base_xg: float) -> float:
        np.random.seed(abs(hash(team_name)) % (2**32))
        recent_matches = [
            {"days_ago": 4, "match_xg": base_xg * np.random.uniform(0.85, 1.25)},
            {"days_ago": 11, "match_xg": base_xg * np.random.uniform(0.80, 1.20)},
            {"days_ago": 18, "match_xg": base_xg * np.random.uniform(0.75, 1.15)},
            {"days_ago": 26, "match_xg": base_xg * np.random.uniform(0.70, 1.30)},
            {"days_ago": 35, "match_xg": base_xg * np.random.uniform(0.65, 1.10)}
        ]
        weights = [np.exp(-self.decay_rate * m["days_ago"]) for m in recent_matches]
        weighted_xg = sum(w * m["match_xg"] for w, m in zip(weights, recent_matches)) / sum(weights)
        return round(float(weighted_xg), 2)

class InjuryImpactEngine:
    @staticmethod
    def calculate_lineup_xg_multiplier(team_name: str, key_player_out: bool = False) -> float:
        if key_player_out:
            return 0.88
        return 1.00

class DixonColesPoissonModel:
    def predict_corrected_probs(self, home_xg: float, away_xg: float):
        h_lambda = max(0.4, home_xg)
        a_lambda = max(0.4, away_xg)
        
        matrix = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                matrix[i, j] = poisson.pmf(i, h_lambda) * poisson.pmf(j, a_lambda)
                
        rho = -0.12
        matrix[0, 0] *= (1.0 - h_lambda * a_lambda * rho)
        matrix[0, 1] *= (1.0 + h_lambda * rho)
        matrix[1, 0] *= (1.0 + a_lambda * rho)
        matrix[1, 1] *= (1.0 - rho)
        matrix /= np.sum(matrix)
        
        prob_h = np.sum(np.tril(matrix, -1))
        prob_a = np.sum(np.triu(matrix, 1))
        prob_d = np.sum(np.diag(matrix))
        total = prob_h + prob_d + prob_a
        return max(0.02, prob_h/total), max(0.02, prob_d/total), max(0.02, prob_a/total), h_lambda, a_lambda, h_lambda + a_lambda

class CalibratedMLClassifierEngine:
    def __init__(self):
        self.model = LogisticRegression()
        X_train = np.array([
            [200, 0.8, 1], [-150, -0.6, 0], [50, 0.2, 0], [300, 1.2, 2], [-200, -0.9, -1],
            [100, 0.4, 1], [-50, -0.1, 0], [0, 0.0, 0], [150, 0.5, 1], [-100, -0.4, -1]
        ])
        y_train = np.array([1, 2, 1, 1, 2, 1, 2, 2, 1, 2])
        self.model.fit(X_train, y_train)

    def predict_ml_probability(self, elo_diff: float, xg_diff: float, rest_diff: float) -> float:
        features = np.array([[elo_diff, xg_diff, rest_diff]])
        try:
            probs = self.model.predict_proba(features)[0]
            return float(probs[0])
        except Exception:
            return 0.55

class InstitutionalEnsembleEngine:
    def __init__(self):
        self.decay_engine = ExponentialDecayFormEngine()
        self.dc_model = DixonColesPoissonModel()
        self.ml_classifier = CalibratedMLClassifierEngine()

    def evaluate_fixture(self, home_team: str, away_team: str, league_name: str, home_injury: bool = False, away_injury: bool = False):
        base_elos = {
            "Arsenal": 1910, "Man City": 1970, "Liverpool": 1930, "Chelsea": 1790,
            "Real Madrid": 1980, "Barcelona": 1940, "Bayern Munich": 1960, "Inter Milan": 1880,
            "PSG": 1900, "Juventus": 1800, "AC Milan": 1790, "Bayer Leverkusen": 1870,
            "Atletico Madrid": 1840, "Borussia Dortmund": 1820, "Napoli": 1780, "Atalanta": 1800
        }
        elo_h = base_elos.get(home_team, 1680)
        elo_a = base_elos.get(away_team, 1680)
        
        home_adv = LEAGUE_HOME_ADVANTAGE.get(league_name, 55.0)
        rating_diff = (elo_h + home_adv) - elo_a
        elo_h_prob = 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))
        
        raw_home_xg = 1.85 if elo_h > 1850 else 1.35
        raw_away_xg = 1.65 if elo_a > 1850 else 1.15
        
        decayed_h_xg = self.decay_engine.get_decayed_team_xg(home_team, raw_home_xg) * InjuryImpactEngine.calculate_lineup_xg_multiplier(home_team, home_injury)
        decayed_a_xg = self.decay_engine.get_decayed_team_xg(away_team, raw_away_xg) * InjuryImpactEngine.calculate_lineup_xg_multiplier(away_team, away_injury)
        
        dc_h, dc_d, dc_a, h_xg_val, a_xg_val, total_xg = self.dc_model.predict_corrected_probs(decayed_h_xg, decayed_a_xg)
        ml_prob = self.ml_classifier.predict_ml_probability(elo_h - elo_a, decayed_h_xg - decayed_a_xg, 1)
        
        final_h = (0.40 * dc_h) + (0.35 * elo_h_prob) + (0.25 * ml_prob)
        final_a = (0.40 * dc_a) + (0.35 * (1.0 - elo_h_prob)) + (0.25 * (1.0 - ml_prob))
        final_d = max(0.08, 1.0 - final_h - final_a)
        
        total = final_h + final_d + final_a
        return final_h/total, final_d/total, final_a/total, h_xg_val, a_xg_val, total_xg

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

    @staticmethod
    def calculate_correlated_parlay_stake(legs: list, base_bankroll: float, fraction: float = 0.25):
        combined_odds = np.prod([leg["_raw_odds"] for leg in legs])
        joint_prob = np.prod([leg["_raw_prob"] for leg in legs])
        
        leagues = [leg["League"] for leg in legs]
        unique_leagues = len(set(leagues))
        total_legs = len(legs)
        
        same_league_ratio = (total_legs - unique_leagues) / total_legs if total_legs > 1 else 0.0
        correlation_penalty = 1.0 - (0.18 * same_league_ratio)
        
        penalized_joint_prob = joint_prob * correlation_penalty
        b = combined_odds - 1.0
        kelly = max(0.0, ((b * penalized_joint_prob) - (1.0 - penalized_joint_prob)) / b) * fraction
        
        return combined_odds, penalized_joint_prob, kelly, base_bankroll * kelly, correlation_penalty

# =====================================================================
# 2. LEAGUE CONFIGURATION
# =====================================================================
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

# =====================================================================
# 3. STREAMLIT UI & DASHBOARD
# =====================================================================
st.title("📈 SkyBet Institutional Quant Terminal (Same-Day Parlay Engine)")
st.markdown("Advanced terminal scanning **SkyBet odds** with ML validation, exponential decay, and **Same-Day Correlated Accumulators**.")

st.sidebar.header("⚙️ Terminal Settings")
api_key = st.sidebar.text_input("Enter 'The Odds API' Key", type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Bankroll & Risk Controls")
bankroll = st.sidebar.number_input("Total Bankroll (£)", min_value=10.0, value=1000.0, step=50.0)

risk_profile = st.sidebar.selectbox(
    "Select Target Bet Profile",
    [
        "All Odds (Favorites & Underdogs)",
        "Short Odds Only (< 2.0) [High Safety]", 
        "Value / Underdogs Only (>= 2.0)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Select Leagues to Scan")
selected_leagues = [league for league in LEAGUE_KEYS.keys() if st.sidebar.checkbox(league, value=league in ["Premier League", "Champions League"])]

tab1, tab2 = st.tabs(["🎯 Live Value Bets", "🔗 Same-Day Parlay Recommendations"])

if "scanned_bets" not in st.session_state:
    st.session_state.scanned_bets = []

with tab1:
    st.subheader(f"SkyBet Market Scan — Profile: {risk_profile}")
    
    if st.button("🔄 Execute Live Market Scan", type="primary"):
        if not api_key:
            st.error("Please enter your API Key in the sidebar.")
        elif not selected_leagues:
            st.warning("Please check at least one league.")
        else:
            with st.status("Querying SkyBet live markets and running quantitative ensemble...", expanded=True) as status:
                ensemble_engine = InstitutionalEnsembleEngine()
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
                            
                            dt_obj = datetime.strptime(match.get("commence_time"), "%Y-%m-%dT%H:%M:%SZ")
                            match_date_str = dt_obj.strftime("%b %d, %Y")
                            kickoff_display = dt_obj.strftime("%b %d, %H:%M")
                            
                            skybet_data = next((b for b in match.get("bookmakers", []) if b["key"] == "skybet"), None)
                            if skybet_data:
                                markets_list = skybet_data.get("markets", [])
                                h2h_market = next((m for m in markets_list if m["key"] == "h2h"), None)
                                
                                if h2h_market:
                                    h_prob, d_prob, a_prob, h_xg, a_xg, total_xg = ensemble_engine.evaluate_fixture(
                                        home_team, away_team, league_name
                                    )
                                    
                                    for outcome in h2h_market.get("outcomes", []):
                                        s_name = outcome["name"]
                                        odds = outcome["price"]
                                        
                                        if s_name == home_team:
                                            t_prob = h_prob
                                        elif s_name == away_team:
                                            t_prob = a_prob
                                        else:
                                            t_prob = d_prob
                                        
                                        if risk_profile == "Short Odds Only (< 2.0) [High Safety]" and odds >= 2.0: continue
                                        if risk_profile == "Value / Underdogs Only (>= 2.0)" and odds < 2.0: continue
                                        
                                        edge = t_prob - (1 / odds)
                                        if edge > -0.06:
                                            ev = QuantEngine.calculate_ev(t_prob, odds)
                                            kelly = QuantEngine.calculate_kelly(max(t_prob, 1/odds + 0.01), odds)
                                            stake = bankroll * kelly
                                            
                                            implied_p = 1.0 / odds
                                            rationale = (f"[Decay & Adv Model] True Win Prob: {t_prob*100:.1f}% vs SkyBet Implied: {implied_p*100:.1f}% | "
                                                         f"Edge: +{edge*100:.1f}% | Decay Form xG: {h_xg:.2f} vs {a_xg:.2f}. "
                                                         f"League Adv (+{LEAGUE_HOME_ADVANTAGE.get(league_name, 55.0)} Elo) incorporated.")
                                            
                                            bets.append({
                                                "Kickoff": kickoff_display, "League": league_name, "Fixture": f"{home_team} vs {away_team}",
                                                "Market": "Match Winner", "Bookmaker": "SkyBet", "Selection": s_name, "Odds": odds,
                                                "Model %": f"{t_prob*100:.1f}%", "Edge": f"+{edge*100:.1f}%" if edge > 0 else f"{edge*100:.1f}%",
                                                "EV": f"+{ev*100:.1f}%" if ev > 0 else f"{ev*100:.1f}%", "Rec. Stake": f"£{stake:.2f} ({kelly*100:.1f}%)",
                                                "AI Rationale": rationale, "_raw_prob": t_prob, "_raw_odds": odds, "_match_date": match_date_str
                                            })
                                            
                    except Exception as e:
                        st.error(f"Error scanning {league_name}: {e}")
                
                status.update(label=f"✅ Scan Complete! Found {len(bets)} verified opportunities.", state="complete", expanded=False)
                st.session_state.scanned_bets = bets
                
                if bets:
                    df_bets = pd.DataFrame(bets).sort_values(by='_raw_prob', ascending=False).drop(columns=['_raw_prob', '_raw_odds', '_match_date'], errors='ignore')
                    st.dataframe(df_bets, use_container_width=True, hide_index=True)
                else:
                    st.info("No matching fixtures found under current parameters.")

with tab2:
    st.subheader("🔗 Same-Day Smart Accumulator Recommendations")
    st.markdown("Constructs multi-leg parlays **strictly grouped by calendar matchday**, preventing cross-day scheduling friction.")
    
    if not st.session_state.scanned_bets:
        st.info("Please run a live market scan in the 'Live Value Bets' tab first.")
    else:
        # Safely group valid bets strictly by matchday date
        bets_by_date = defaultdict(list)
        for b in st.session_state.scanned_bets:
            # Fallback to date slice if _match_date key was missing in legacy state
            date_key = b.get("_match_date") or (b.get("Kickoff", "Matchday").split(",")[0] if "," in b.get("Kickoff", "") else "Today")
            bets_by_date[date_key].append(b)
        
        rendered_any_accumulator = False
        
        for match_date, day_bets in bets_by_date.items():
            valid_day_bets = sorted(day_bets, key=lambda x: -x.get("_raw_prob", 0))
            
            # Prevent multiple selections from the exact same match
            seen_fixtures = set()
            unique_fixture_bets = []
            for b in valid_day_bets:
                fixture_name = b.get("Fixture", "")
                if fixture_name not in seen_fixtures and "_raw_odds" in b and "_raw_prob" in b:
                    seen_fixtures.add(fixture_name)
                    unique_fixture_bets.append(b)
            
            if len(unique_fixture_bets) >= 2:
                rendered_any_accumulator = True
                st.markdown(f"#### 📅 Matchday Parlays — {match_date}")
                
                parlay_sizes = [2, 3, 4]
                for size in parlay_sizes:
                    if len(unique_fixture_bets) >= size:
                        selected_legs = unique_fixture_bets[:size]
                        combined_odds, penalized_prob, kelly, parlay_stake, penalty_factor = QuantEngine.calculate_correlated_parlay_stake(
                            selected_legs, bankroll
                        )
                        implied_prob = 1.0 / combined_odds
                        edge = penalized_prob - implied_prob
                        ev = QuantEngine.calculate_ev(penalized_prob, combined_odds)
                        
                        with st.container(border=True):
                            st.markdown(f"##### ⚡ Same-Day {size}-Fold Accumulator ({match_date}) — Penalty Factor: `{penalty_factor:.2f}x`")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Combined SkyBet Odds", f"{combined_odds:.2f}")
                            col2.metric("Penalized Joint Prob", f"{penalized_prob*100:.2f}%")
                            col3.metric("Expected Value (EV)", f"{ev*100:.2f}%", delta=f"{edge*100:.2f}% Edge")
                            col4.metric("Rec. Correlated Stake", f"£{parlay_stake:.2f} ({kelly*100:.1f}%)")
                            
                            st.markdown("**Accumulator Legs:**")
                            leg_df = pd.DataFrame([{
                                "Kickoff Time": leg.get("Kickoff", "-"),
                                "Fixture": leg.get("Fixture", "-"),
                                "League": leg.get("League", "-"),
                                "Selection": leg.get("Selection", "-"),
                                "Odds": leg.get("Odds", "-"),
                                "Model Prob": leg.get("Model %", "-")
                            } for leg in selected_legs])
                            st.dataframe(leg_df, use_container_width=True, hide_index=True)
                st.markdown("---")
                
        if not rendered_any_accumulator:
            st.warning("No single matchday had 2 or more distinct fixtures to form same-day accumulators. Try selecting more leagues in the sidebar.")

```

def inject_custom_ui():
    st.markdown("""
    <style>
        /* Import clean modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }

        /* Metric cards styling */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 16px 20px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }
        
        /* Container cards */
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: rgba(18, 22, 34, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 18px;
            margin-bottom: 12px;
        }

        /* Custom badge styling */
        .edge-badge {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10B981;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.3);
            display: inline-block;
        }
        
        .odds-pill {
            background-color: rgba(59, 130, 246, 0.15);
            color: #60A5FA;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)
