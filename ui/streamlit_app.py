"""
AI Football Picks Dashboard - Milestone 14 Redesign

A customer-friendly Streamlit UI for betting picks.

Run:
    streamlit run ui/streamlit_app.py
"""

import os
import sys
import glob
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import config
except ImportError:
    config = None

from ui.components import (
    inject_dark_theme,
    hero_pick_card,
    pick_card,
    track_record_widget,
    confidence_stars,
    probability_bar,
    format_op_pick,
    value_badge,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions() -> pd.DataFrame:
    """Load the latest predictions CSV."""
    reports_dir = _ROOT / "reports"
    current = reports_dir / "cgm_upcoming_predictions.csv"
    if current.exists():
        try:
            df = pd.read_csv(current)
            if len(df) > 0:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def load_picks() -> pd.DataFrame:
    """Load picks from reports/."""
    reports_dir = _ROOT / "reports"
    
    # Try picks.csv first (explained picks)
    for name in ["picks_explained.csv", "picks.csv", "picks_goals.csv"]:
        path = reports_dir / name
        if path.exists():
            df = pd.read_csv(path)
            if len(df) > 0:
                # Normalize column names
                if 'fixture_datetime' in df.columns:
                    df = df.rename(columns={'fixture_datetime': 'date'})
                return df
    
    return pd.DataFrame()


def load_backtest() -> pd.DataFrame:
    """Load backtest results."""
    reports_dir = _ROOT / "reports"
    # Prefer the full backtest we just ran
    for name in ["full_backtest_2025.csv", "backtest_epl_2025.csv"]:
        path = reports_dir / name
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def load_history() -> pd.DataFrame:
    """Load match history for results."""
    path = _ROOT / "data" / "enhanced" / "cgm_match_history_with_elo_stats_xg.csv"
    if path.exists():
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    return pd.DataFrame()


def get_track_record(backtest: pd.DataFrame) -> dict:
    """Calculate win rates from backtest data."""
    if backtest.empty:
        return {
            'week': {'wins': 0, 'total': 0, 'rate': 0},
            'month': {'wins': 0, 'total': 0, 'rate': 0},
            'season': {'wins': 0, 'total': 0, 'rate': 0},
        }
    
    backtest['date'] = pd.to_datetime(backtest['date'], errors='coerce')
    
    # Filter out 0-0 results (likely unplayed)
    played = backtest[~((backtest['ft_home'] == 0) & (backtest['ft_away'] == 0))]
    
    if played.empty:
        return {
            'week': {'wins': 0, 'total': 0, 'rate': 0},
            'month': {'wins': 0, 'total': 0, 'rate': 0},
            'season': {'wins': 0, 'total': 0, 'rate': 0},
        }
    
    # Calculate O/U 2.5 accuracy
    played = played.copy()
    played['actual_over'] = (played['ft_home'] + played['ft_away']) > 2.5
    played['pred_over'] = played['p_over25'] > 0.35  # Calibrated threshold
    played['correct'] = played['actual_over'] == played['pred_over']
    
    now = pd.Timestamp.now()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)
    
    # Calculate stats for each period
    def calc_stats(df):
        if len(df) == 0:
            return {'wins': 0, 'total': 0, 'rate': 0}
        wins = df['correct'].sum()
        total = len(df)
        return {'wins': int(wins), 'total': int(total), 'rate': wins / total}
    
    return {
        'week': calc_stats(played[played['date'] >= week_ago]),
        'month': calc_stats(played[played['date'] >= month_ago]),
        'season': calc_stats(played),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="AI Football Picks",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    # Inject dark theme
    inject_dark_theme()
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <h1 style="color: #f1f5f9; margin-bottom: 0;">🎯 AI Football Picks</h1>
        <p style="color: #64748b; margin-top: 0;">Smart betting predictions powered by machine learning</p>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="text-align: right; color: #64748b; padding-top: 16px;">
            Last update: {datetime.now().strftime('%H:%M')}
        </div>
        """, unsafe_allow_html=True)
    
    # Load data
    picks = load_picks()
    predictions = load_predictions()
    backtest = load_backtest()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Predictions Table",
        "🎯 Today's Picks",
        "📈 Track Record",
        "🔍 Match Explorer",
        "⚙️ Settings"
    ])
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: PREDICTIONS TABLE (NEW - like generate_predictions_report.py)
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### 📊 Multi-League Predictions")
        
        if predictions.empty:
            st.info("🔄 No predictions available. Run the pipeline to generate predictions.")
        else:
            # League picker
            leagues = ["All Leagues"] + sorted(predictions['league'].dropna().unique().tolist())
            selected_league = st.selectbox("🌍 Select League", leagues, key="league_picker")
            
            # Filter by league
            if selected_league != "All Leagues":
                df = predictions[predictions['league'] == selected_league].copy()
            else:
                df = predictions.copy()
            
            if df.empty:
                st.warning(f"No predictions for {selected_league}")
            else:
                # Calculate best value for each match
                def get_best_value(row):
                    ev_cols = {
                        'Over 2.5': row.get('EV_over25', float('-inf')),
                        'Under 2.5': row.get('EV_under25', float('-inf')),
                        'BTTS Yes': row.get('EV_btts_yes', float('-inf')),
                        'BTTS No': row.get('EV_btts_no', float('-inf')),
                    }
                    valid_evs = {k: v for k, v in ev_cols.items() if pd.notna(v) and v > -100}
                    if not valid_evs:
                        return "—", 0
                    best = max(valid_evs, key=valid_evs.get)
                    return best, valid_evs[best]
                
                df['best_pick'], df['best_ev'] = zip(*df.apply(get_best_value, axis=1))
                
                # Date filter
                df['date'] = pd.to_datetime(df.get('fixture_datetime'), errors='coerce')
                dates = ["All Dates"] + sorted(df['date'].dt.date.dropna().unique().astype(str).tolist())
                selected_date = st.selectbox("📅 Filter by Date", dates, key="date_picker")
                
                if selected_date != "All Dates":
                    df = df[df['date'].dt.date.astype(str) == selected_date]
                
                # Summary stats
                st.markdown(f"**{len(df)} matches** | League: {selected_league}")
                
                st.markdown("---")
                
                # Top 10 by EV Section
                st.markdown("### 🏆 Top 10 Value Bets (by EV)")
                top10 = df.nlargest(10, 'best_ev')
                
                for idx, (_, row) in enumerate(top10.iterrows()):
                    medal = ["🥇", "🥈", "🥉"][idx] if idx < 3 else f"#{idx+1}"
                    home = row.get('home', '?')
                    away = row.get('away', '?')
                    league = row.get('league', '?')
                    best_pick = row['best_pick']
                    best_ev = row['best_ev']
                    
                    ev_color = "#22c55e" if best_ev > 0.05 else "#fbbf24" if best_ev > 0 else "#ef4444"
                    
                    st.markdown(f"""
                    <div style="background: #0f172a; border-radius: 8px; padding: 12px 16px; margin: 4px 0; border-left: 4px solid {ev_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-size: 18px; margin-right: 8px;">{medal}</span>
                                <span style="color: #f1f5f9; font-weight: 600;">{home} vs {away}</span>
                                <span style="color: #64748b; margin-left: 8px;">({league})</span>
                            </div>
                            <div>
                                <span style="color: #60a5fa; font-weight: 600; margin-right: 12px;">{best_pick}</span>
                                <span style="color: {ev_color}; font-weight: 700;">+{best_ev*100:.1f}% EV</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Full Predictions Table
                st.markdown("### 📋 All Predictions")
                
                # Create display dataframe
                display_cols = []
                for _, row in df.iterrows():
                    home = row.get('home', '?')
                    away = row.get('away', '?')
                    league = row.get('league', '?')
                    match_date = row['date'].strftime('%m/%d') if pd.notna(row['date']) else '?'
                    
                    # O/U and BTTS
                    p_over = row.get('p_over25', 0) * 100 if pd.notna(row.get('p_over25')) else 0
                    p_under = row.get('p_under25', 0) * 100 if pd.notna(row.get('p_under25')) else 0
                    p_btts_yes = row.get('p_btts_yes', 0) * 100 if pd.notna(row.get('p_btts_yes')) else 0
                    p_btts_no = row.get('p_btts_no', 0) * 100 if pd.notna(row.get('p_btts_no')) else 0
                    
                    # EVs
                    ev_over = row.get('EV_over25', 0) * 100 if pd.notna(row.get('EV_over25')) else 0
                    ev_btts = row.get('EV_btts_yes', 0) * 100 if pd.notna(row.get('EV_btts_yes')) else 0
                    
                    display_cols.append({
                        'Date': match_date,
                        'League': league,
                        'Match': f"{home} vs {away}",
                        'O2.5': f"{p_over:.0f}%",
                        'U2.5': f"{p_under:.0f}%",
                        'BTTS Y': f"{p_btts_yes:.0f}%",
                        'BTTS N': f"{p_btts_no:.0f}%",
                        'Best Pick': row['best_pick'],
                        'EV': f"+{row['best_ev']*100:.1f}%" if row['best_ev'] > 0 else f"{row['best_ev']*100:.1f}%",
                    })
                
                display_df = pd.DataFrame(display_cols)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: TODAY'S PICKS
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        if picks.empty:
            st.info("🔄 No picks available yet. Run the pipeline to generate predictions.")
        else:
            # Sort by confidence/score
            if 'score' in picks.columns:
                picks = picks.sort_values('score', ascending=False)
            elif 'model_prob' in picks.columns:
                picks = picks.sort_values('model_prob', ascending=False)
            
            # Hero pick (best pick)
            if len(picks) > 0:
                hero_pick_card(picks.iloc[0])
            
            # Track record mini-widget
            track_record = get_track_record(backtest)
            if track_record['season']['total'] > 0:
                st.markdown("### 📊 Quick Stats")
                cols = st.columns(3)
                with cols[0]:
                    rate = track_record['week']['rate'] * 100
                    st.metric("Last 7 Days", f"{rate:.0f}%" if rate > 0 else "—")
                with cols[1]:
                    rate = track_record['month']['rate'] * 100
                    st.metric("This Month", f"{rate:.0f}%" if rate > 0 else "—")
                with cols[2]:
                    rate = track_record['season']['rate'] * 100
                    st.metric("Season", f"{rate:.0f}%")
            
            # All picks
            if len(picks) > 1:
                st.markdown("### 📋 All Picks")
                for idx, row in picks.iloc[1:].iterrows():
                    pick_card(row)
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3: TRACK RECORD
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 📊 Performance History")
        
        if backtest.empty:
            st.info("📈 Run a backtest to see historical performance.")
        else:
            # Track record widget
            track_record = get_track_record(backtest)
            track_record_widget(track_record)
            
            st.markdown("---")
            
            # Recent results
            st.markdown("### 📅 Recent Results")
            
            backtest['date'] = pd.to_datetime(backtest['date'], errors='coerce')
            recent = backtest.sort_values('date', ascending=False).head(20)
            
            # Filter out unplayed
            played = recent[~((recent['ft_home'] == 0) & (recent['ft_away'] == 0))]
            
            for _, row in played.iterrows():
                home = row.get('home', '?')
                away = row.get('away', '?')
                ft_h = int(row.get('ft_home', 0))
                ft_a = int(row.get('ft_away', 0))
                
                actual_over = (ft_h + ft_a) > 2.5
                pred_over = row.get('p_over25', 0) > 0.35
                won = actual_over == pred_over
                
                icon = "✅" if won else "❌"
                status_color = "#4ade80" if won else "#ef4444"
                pick_label = "OVER 2.5" if pred_over else "UNDER 2.5"
                
                st.markdown(f"""
                <div style="
                    background: #0f172a;
                    border-radius: 8px;
                    padding: 12px 16px;
                    margin: 4px 0;
                    border-left: 4px solid {status_color};
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div>
                        <span style="color: #e2e8f0;">{icon} {home} {ft_h}-{ft_a} {away}</span>
                        <span style="color: #64748b; margin-left: 12px;">({pick_label})</span>
                    </div>
                    <span style="color: {status_color}; font-weight: 600;">{"WON" if won else "LOST"}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4: MATCH EXPLORER
    # ─────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 🔍 Match Explorer")
        
        if predictions.empty:
            st.info("No predictions available.")
        else:
            # Create match selector
            predictions['match_label'] = predictions['home'] + " vs " + predictions['away']
            match_options = predictions['match_label'].tolist()
            
            selected_match = st.selectbox("Select Match", match_options)
            
            if selected_match:
                match_row = predictions[predictions['match_label'] == selected_match].iloc[0]
                
                # Match header
                home = match_row['home']
                away = match_row['away']
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                    border-radius: 16px;
                    padding: 24px;
                    margin: 16px 0;
                    text-align: center;
                    border: 1px solid #3d7ab8;
                ">
                    <div style="color: #f1f5f9; font-size: 28px; font-weight: 700;">
                        {home} vs {away}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Expected goals
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    mu_h = match_row.get('mu_home', 0)
                    mu_a = match_row.get('mu_away', 0)
                    st.markdown(f"""
                    <div style="text-align: center; margin: 16px 0;">
                        <div style="color: #64748b; font-size: 14px;">Expected Goals</div>
                        <div style="color: #f1f5f9; font-size: 36px; font-weight: 700;">
                            {mu_h:.2f} - {mu_a:.2f}
                        </div>
                        <div style="color: #94a3b8; font-size: 14px;">Total: {mu_h + mu_a:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Value Bets Table
                st.markdown("### 💰 Value Analysis")
                
                def format_ev(ev):
                    if ev > 0:
                        return f"+{ev*100:.1f}%"
                    return f"{ev*100:.1f}%"
                
                # Over/Under Section
                st.markdown("#### ⚽ Goals (Over/Under 2.5)")
                markets_ou = [
                    ("Over 2.5", match_row.get('p_over25', 0), match_row.get('odds_over25', 0), match_row.get('EV_over25', 0)),
                    ("Under 2.5", match_row.get('p_under25', 0), match_row.get('odds_under25', 0), match_row.get('EV_under25', 0)),
                ]
                
                for name, prob, odds, ev in markets_ou:
                    is_pick = ev > 0.05
                    bg = "#1e3a2e" if is_pick else "#1e293b"
                    border = "#22c55e" if is_pick else "#334155"
                    st.markdown(f"""
                    <div style="background: {bg}; border: 1px solid {border}; border-radius: 8px; padding: 12px; margin: 6px 0; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="color: #f1f5f9; font-weight: 600;">{name}</span>
                            <span style="color: #94a3b8; margin-left: 12px;">{prob*100:.1f}%</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 16px;">
                            <span style="color: #fbbf24;">@ {odds:.2f}</span>
                            <span style="color: {'#22c55e' if ev > 0 else '#ef4444'}; font-weight: 600;">{format_ev(ev)}</span>
                            {value_badge(ev)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # BTTS Section
                st.markdown("#### 🎯 Both Teams to Score")
                btts_yes_odds = match_row.get('odds_btts_yes', 0)
                btts_no_odds = match_row.get('odds_btts_no', 0)
                
                if btts_yes_odds and btts_no_odds:
                    markets_btts = [
                        ("BTTS Yes", match_row.get('p_btts_yes', 0), btts_yes_odds, match_row.get('EV_btts_yes', 0)),
                        ("BTTS No", match_row.get('p_btts_no', 0), btts_no_odds, match_row.get('EV_btts_no', 0)),
                    ]
                    
                    for name, prob, odds, ev in markets_btts:
                        is_pick = ev > 0.05
                        bg = "#1e3a2e" if is_pick else "#1e293b"
                        border = "#22c55e" if is_pick else "#334155"
                        st.markdown(f"""
                        <div style="background: {bg}; border: 1px solid {border}; border-radius: 8px; padding: 12px; margin: 6px 0; display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="color: #f1f5f9; font-weight: 600;">{name}</span>
                                <span style="color: #94a3b8; margin-left: 12px;">{prob*100:.1f}%</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 16px;">
                                <span style="color: #fbbf24;">@ {odds:.2f}</span>
                                <span style="color: {'#22c55e' if ev > 0 else '#ef4444'}; font-weight: 600;">{format_ev(ev)}</span>
                                {value_badge(ev)}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("BTTS odds not available for this match")
                
                # Recommended Picks Summary
                st.markdown("### ✅ Recommended Picks")
                
                all_picks = [
                    (format_op_pick("UNDER 2.5"), match_row.get('EV_under25', 0), match_row.get('odds_under25', 0)),
                    (format_op_pick("OVER 2.5"), match_row.get('EV_over25', 0), match_row.get('odds_over25', 0)),
                    (format_op_pick("BTTS NO"), match_row.get('EV_btts_no', 0), match_row.get('odds_btts_no', 0)),
                    (format_op_pick("BTTS YES"), match_row.get('EV_btts_yes', 0), match_row.get('odds_btts_yes', 0)),
                ]
                
                # Sort by EV and filter positive
                value_picks = [(n, ev, o) for n, ev, o in all_picks if ev > 0.02] 
                value_picks.sort(key=lambda x: x[1], reverse=True)
                
                if value_picks:
                    for i, (name, ev, odds) in enumerate(value_picks[:3]):
                        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "▪️"
                        st.markdown(f"""
                        <div style="background: #1e3a2e; border: 1px solid #22c55e; border-radius: 8px; padding: 16px; margin: 8px 0;">
                            <span style="font-size: 20px;">{medal}</span>
                            <span style="color: #f1f5f9; font-weight: 700; font-size: 18px; margin-left: 8px;">{name}</span>
                            <span style="color: #fbbf24; margin-left: 12px;">@ {odds:.2f}</span>
                            <span style="color: #22c55e; margin-left: 12px; font-weight: 600;">{format_ev(ev)} EV</span>
                            <span style="margin-left: 12px;">{value_badge(ev)}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No value bets identified for this match")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5: SETTINGS
    # ─────────────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown("### ⚙️ Settings")
        st.caption("Advanced configuration for power users")
        
        with st.expander("🎚️ Threshold Tuning"):
            ou_threshold = st.slider(
                "Over 2.5 Threshold",
                min_value=0.3,
                max_value=0.7,
                value=0.35,
                step=0.05,
                help="Lower = more aggressive (pick more overs)"
            )
            
            btts_threshold = st.slider(
                "BTTS Threshold",
                min_value=0.3,
                max_value=0.7,
                value=0.35,
                step=0.05,
                help="Lower = more aggressive (pick more BTTS yes)"
            )
            
            st.info(f"Current thresholds: O/U 2.5 = {ou_threshold*100:.0f}%, BTTS = {btts_threshold*100:.0f}%")
        
        with st.expander("📊 Raw Data"):
            if not picks.empty:
                st.dataframe(picks, use_container_width=True)
            else:
                st.info("No picks data available")


if __name__ == "__main__":
    main()
