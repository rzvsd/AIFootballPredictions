import streamlit as st
import pandas as pd

def inject_dark_theme():
    """Injects custom CSS for a premium dark theme."""
    st.markdown("""
        <style>
        /* Global Reset */
        .stApp {
            background-color: #0f172a;
            color: #f1f5f9;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #f8fafc !important;
            font-family: 'Inter', sans-serif;
        }
        
        /* Cards */
        .stMetric {
            background-color: #1e293b;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #334155;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px;
            color: #94a3b8;
            font-size: 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1e293b;
            color: #3b82f6;
            font-weight: 600;
        }
        
        /* Dataframes */
        [data-testid="stDataFrame"] {
            background-color: #1e293b;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

def value_badge(ev: float) -> str:
    """Returns HTML for a value badge based on EV."""
    if ev >= 0.20:
        return '<span style="background: #22c55e; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">💎 BIG VALUE</span>'
    if ev >= 0.05:
        return '<span style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">✅ VALUE</span>'
    if ev > 0:
        return '<span style="background: #6b7280; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">⚖️ EDGE</span>'
    return '<span style="color: #ef4444; font-size: 12px;">❌ NO VALUE</span>'

def format_op_pick(pick_str: str) -> str:
    """Formats raw pick strings (e.g. 'home' -> 'HOME WIN')."""
    if not pick_str: return ""
    pick_upper = pick_str.upper()
    if "HOME" in pick_upper: return "HOME WIN"
    if "AWAY" in pick_upper: return "AWAY WIN"
    if "DRAW" in pick_upper: return "DRAW"
    if "OVER" in pick_upper: return "OVER 2.5"
    if "UNDER" in pick_upper: return "UNDER 2.5"
    if "BTTS_YES" in pick_upper or "YES" in pick_upper: return "BTTS YES"
    if "BTTS_NO" in pick_upper or "NO" in pick_upper: return "BTTS NO"
    return pick_upper

def confidence_stars(score: float) -> str:
    """Returns star string based on confidence score (0-100)."""
    if score >= 80: return "⭐⭐⭐⭐⭐"
    if score >= 60: return "⭐⭐⭐⭐"
    if score >= 40: return "⭐⭐⭐"
    if score >= 20: return "⭐⭐"
    return "⭐"

def probability_bar(prob: float, color="#3b82f6") -> str:
    """Returns a single-line HTML progress bar string."""
    pct = int(prob * 100)
    return f'<div style="background-color: #334155; width: 100%; height: 8px; border-radius: 4px; margin-top: 6px;"><div style="background-color: {color}; width: {pct}%; height: 100%; border-radius: 4px;"></div></div>'

def hero_pick_card(row: pd.Series):
    """Displays the 'Hero' (best) pick in a prominent card."""
    match_name = f"{row.get('home')} vs {row.get('away')}"
    pick_type = format_op_pick(row.get('pick', 'Unknown'))
    odds = row.get('odds', 0.0)
    ev = row.get('ev', 0.0)
    score = row.get('score', 0)
    # If using newer report format, might be 'model_prob' etc.
    prob = row.get('model_prob', 0.0)
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    ">
        <div style="color: #93c5fd; font-size: 14px; font-weight: 600; text-transform: uppercase; margin-bottom: 8px;">
            ✨ STARTING PICK OF THE DAY
        </div>
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <h2 style="color: white !important; margin: 0; font-size: 24px;">{match_name}</h2>
                <div style="color: #bfdbfe; font-size: 16px; margin-top: 4px;">{row.get('date')} • {row.get('league')}</div>
            </div>
            <div style="text-align: right;">
                <div style="background: rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 20px; color: #fff; font-weight: 600;">
                    {confidence_stars(score)}
                </div>
            </div>
        </div>
        
        <div style="margin-top: 24px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;">
            <div style="background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px;">
                <div style="color: #93c5fd; font-size: 12px;">RECOMMENDED BET</div>
                <div style="color: white; font-weight: 700; font-size: 18px;">{pick_type}</div>
            </div>
            <div style="background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px;">
                <div style="color: #93c5fd; font-size: 12px;">ODDS & VALUE</div>
                <div style="color: #fbbf24; font-weight: 700; font-size: 18px;">@{odds:.2f} <span style="font-size:14px; color:#22c55e;">({ev*100:+.1f}%)</span></div>
            </div>
            <div style="background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px;">
                <div style="color: #93c5fd; font-size: 12px;">PROBABILITY</div>
                <div style="color: white; font-weight: 700; font-size: 18px;">{prob*100:.1f}%</div>
                {probability_bar(prob, color="#22c55e")}
            </div>
        </div>
        
        <div style="margin-top: 16px; color: #dbeafe; font-style: italic; font-size: 14px;">
            "Analysis: {row.get('reason_text', 'Strong statistical edge identified.')}"
        </div>
    </div>
    """, unsafe_allow_html=True)

def pick_card(row: pd.Series):
    """Displays a standard pick card."""
    match_name = f"{row.get('home')} vs {row.get('away')}"
    pick_type = format_op_pick(row.get('pick', 'Unknown'))
    odds = row.get('odds', 0.0)
    ev = row.get('ev', 0.0)
    prob = row.get('model_prob', 0.0)
    
    st.markdown(f"""
    <div style="
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        transition: transform 0.2s;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="flex: 2;">
                <div style="color: #94a3b8; font-size: 12px;">{row.get('date')} • {row.get('league')}</div>
                <div style="color: #f8fafc; font-weight: 600; font-size: 16px;">{match_name}</div>
            </div>
            <div style="flex: 1; text-align: center;">
                <div style="color: #3b82f6; font-weight: 700; font-size: 15px;">{pick_type}</div>
            </div>
            <div style="flex: 1; text-align: right;">
                <span style="color: #fbbf24; font-weight: 600;">@{odds:.2f}</span>
                <span style="margin-left: 8px; color: {'#22c55e' if ev > 0.05 else '#94a3b8'}; font-weight: 600;">{ev*100:+.1f}%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def track_record_widget(stats: dict):
    """Displays win rates and profit summary."""
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px;">
    """, unsafe_allow_html=True)
    
    for period, data in stats.items():
        rate = data.get('rate', 0.0) * 100
        total = data.get('total', 0)
        wins = data.get('wins', 0)
        
        color = "#22c55e" if rate > 50 else "#94a3b8"
        
        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">{period}</div>
            <div style="color: {color}; font-size: 32px; font-weight: 700; margin: 8px 0;">{rate:.0f}%</div>
            <div style="color: #64748b; font-size: 13px;">{wins} wins / {total} matches</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

def result_card(match: dict, result_status: str):
    """(Optional) Displays a historic result."""
    pass # Implemented inline in main app for now, or can be moved here.
