"""
Streamlit UI Components - Milestone 14

Reusable components for customer-friendly betting picks display.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta


def confidence_stars(confidence: float, max_stars: int = 5) -> str:
    """
    Convert a confidence score (0-1) to star rating.
    
    Args:
        confidence: Float between 0 and 1
        max_stars: Maximum stars to show
    
    Returns:
        Star string like "⭐⭐⭐⭐"
    """
    if pd.isna(confidence):
        return "⭐"
    
    # Scale to 1-5 stars
    stars = max(1, min(max_stars, int(confidence * max_stars) + 1))
    return "⭐" * stars


def confidence_label(confidence: float) -> str:
    """Convert confidence score to human-readable label."""
    if pd.isna(confidence) or confidence < 0.5:
        return "Low"
    elif confidence < 0.6:
        return "Moderate"
    elif confidence < 0.7:
        return "Good"
    elif confidence < 0.8:
        return "High"
    else:
        return "Very High"


def format_pick_type(pick_type: str) -> tuple:
    """
    Format pick type for display.
    
    Returns:
        Tuple of (display_name, emoji)
    """
    types = {
        'OU25_OVER': ('Over 2.5 Goals', '⚽'),
        'OU25_UNDER': ('Under 2.5 Goals', '🛡️'),
        'BTTS_YES': ('Both Teams to Score', '🎯'),
        'BTTS_NO': ('Clean Sheet Expected', '🧤'),
        '1X2_HOME': ('Home Win', '🏠'),
        '1X2_AWAY': ('Away Win', '✈️'),
        '1X2_DRAW': ('Draw', '🤝'),
    }
    return types.get(pick_type, (pick_type, '📊'))


def hero_pick_card(pick: pd.Series):
    """
    Display the featured pick of the day.
    
    Args:
        pick: Series with pick data (home, away, pick_type, confidence, narrative)
    """
    pick_name, emoji = format_pick_type(pick.get('pick_type', ''))
    confidence = pick.get('model_prob', pick.get('confidence', 0.6))
    stars = confidence_stars(confidence)
    label = confidence_label(confidence)
    
    st.markdown("""
    <style>
    .hero-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        border: 1px solid #3d7ab8;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero-title {
        color: #ffd700;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }
    .hero-match {
        color: #ffffff;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .hero-pick {
        color: #4ade80;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .hero-confidence {
        color: #fbbf24;
        font-size: 16px;
        margin-bottom: 12px;
    }
    .hero-narrative {
        color: #94a3b8;
        font-size: 14px;
        font-style: italic;
        border-left: 3px solid #4ade80;
        padding-left: 12px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    home = pick.get('home', 'Team A')
    away = pick.get('away', 'Team B')
    narrative = pick.get('narrative', pick.get('title', 'Strong statistical edge'))
    
    st.markdown(f"""
    <div class="hero-card">
        <div class="hero-title">🔥 PICK OF THE DAY</div>
        <div class="hero-match">{home} vs {away}</div>
        <div class="hero-pick">{emoji} {pick_name}</div>
        <div class="hero-confidence">{stars} {label} Confidence</div>
        <div class="hero-narrative">💡 "{narrative}"</div>
    </div>
    """, unsafe_allow_html=True)


def pick_card(pick: pd.Series, show_time: bool = True):
    """
    Display a standard pick card.
    
    Args:
        pick: Series with pick data
        show_time: Whether to show match time
    """
    pick_name, emoji = format_pick_type(pick.get('pick_type', ''))
    confidence = pick.get('model_prob', pick.get('confidence', 0.5))
    stars = confidence_stars(confidence, max_stars=4)
    
    home = pick.get('home', 'Team A')
    away = pick.get('away', 'Team B')
    narrative = pick.get('narrative', pick.get('title', ''))
    
    # Time formatting
    time_str = ""
    if show_time and 'fixture_datetime' in pick:
        try:
            dt = pd.to_datetime(pick['fixture_datetime'])
            time_str = dt.strftime('%H:%M')
        except:
            time_str = ""
    
    st.markdown(f"""
    <div style="
        background: #1e293b;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border: 1px solid #334155;
    ">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #f1f5f9; font-weight: 600;">{home} vs {away}</span>
            <span style="color: #64748b;">{time_str}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #4ade80; font-weight: 600;">{emoji} {pick_name}</span>
            <span style="color: #fbbf24;">{stars}</span>
        </div>
        <div style="color: #94a3b8; font-size: 13px; margin-top: 8px; font-style: italic;">
            "{narrative[:80]}{'...' if len(str(narrative)) > 80 else ''}"
        </div>
    </div>
    """, unsafe_allow_html=True)


def result_card(result: pd.Series):
    """
    Display a result card for yesterday's picks.
    
    Args:
        result: Series with match result data
    """
    won = result.get('won', False)
    icon = "✅" if won else "❌"
    status = "WON" if won else "LOST"
    status_color = "#4ade80" if won else "#ef4444"
    
    home = result.get('home', 'Team A')
    away = result.get('away', 'Team B')
    score = f"{int(result.get('ft_home', 0))}-{int(result.get('ft_away', 0))}"
    pick_name, _ = format_pick_type(result.get('pick_type', ''))
    pick_value = result.get('pick_value', '')
    
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
            <span style="color: #e2e8f0;">{icon} {home} {score} {away}</span>
            <span style="color: #64748b; margin-left: 12px;">({pick_name})</span>
        </div>
        <span style="color: {status_color}; font-weight: 600;">{status}</span>
    </div>
    """, unsafe_allow_html=True)


def track_record_widget(stats: dict):
    """
    Display track record statistics.
    
    Args:
        stats: Dict with keys like 'week', 'month', 'season'
               Each value is a dict with 'wins', 'total', 'rate'
    """
    st.markdown("""
    <style>
    .stat-box {
        background: #1e293b;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #334155;
    }
    .stat-label {
        color: #64748b;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-rate {
        color: #4ade80;
        font-size: 28px;
        font-weight: 700;
        margin: 8px 0;
    }
    .stat-detail {
        color: #94a3b8;
        font-size: 13px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    
    periods = [
        ('week', 'Last 7 Days'),
        ('month', 'This Month'),
        ('season', 'Season')
    ]
    
    for col, (key, label) in zip(cols, periods):
        data = stats.get(key, {'wins': 0, 'total': 0, 'rate': 0})
        rate = data.get('rate', 0) * 100
        wins = data.get('wins', 0)
        total = data.get('total', 0)
        
        with col:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">{label}</div>
                <div class="stat-rate">{rate:.0f}%</div>
                <div class="stat-detail">{wins}/{total} picks</div>
            </div>
            """, unsafe_allow_html=True)


def probability_bar(label: str, probability: float, threshold: float = 0.5):
    """
    Display a probability bar with visual indicator.
    
    Args:
        label: Market label (e.g., "Over 2.5")
        probability: Model probability (0-1)
        threshold: Decision threshold
    """
    pct = probability * 100
    color = "#4ade80" if probability > threshold else "#64748b"
    
    st.markdown(f"""
    <div style="margin: 8px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #e2e8f0;">{label}</span>
            <span style="color: {color}; font-weight: 600;">{pct:.0f}%</span>
        </div>
        <div style="background: #334155; border-radius: 4px; height: 8px; overflow: hidden;">
            <div style="background: {color}; width: {pct}%; height: 100%; border-radius: 4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def inject_dark_theme():
    """Inject custom dark theme CSS."""
    st.markdown("""
    <style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0f172a;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        border-radius: 12px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94a3b8;
        border-radius: 8px;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    
    h1, h2, h3 {
        color: #f1f5f9;
    }
    
    .stMetric {
        background-color: #1e293b;
        padding: 16px;
        border-radius: 12px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
