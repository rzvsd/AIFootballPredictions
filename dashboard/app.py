"""
Streamlit Dashboard — Monitoring (Phase 4)

Features:
- Bankroll and P/L with filters (league/date/market)
- Open vs Settled bets from data/bets_log.csv (+ results joined from data/processed/raw)
- Upcoming predictions (top picks) using current engine
- Export CSV and figure PNG

Run:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import io
import os
import sys
import json
import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root (parent of this file) is importable: config.py, bet_fusion.py
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
import bet_fusion as fusion


def load_bankroll() -> float:
    path = os.path.join('data','bankroll.json')
    try:
        with open(path,'r',encoding='utf-8') as f:
            return float((json.load(f) or {}).get('bankroll', 1000.0))
    except Exception:
        return 1000.0


def load_bets() -> pd.DataFrame:
    path = os.path.join('data','bets_log.csv')
    if not os.path.exists(path):
        return pd.DataFrame(columns=['date','league','home_team','away_team','market','odds','stake','model_prob','expected_value'])
    df = pd.read_csv(path)
    # Parse date
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    except Exception:
        pass
    # Split market/outcome
    def split_market(m: str) -> Tuple[str,str,str]:
        m = str(m)
        # Examples: "1X2 H", "OU 2.5 Over", "TG Interval 0-3", "DC 1X"
        parts = m.split()
        if not parts:
            return m, m, ''
        if parts[0] == 'OU' and len(parts) >= 3:
            return 'OU', f"OU {parts[1]}", parts[2]
        if parts[0] == 'TG' and len(parts) >= 3:
            return 'TG Interval', 'TG Interval', parts[2]
        if parts[0] == '1X2' and len(parts) >= 2:
            return '1X2', '1X2', parts[1]
        if parts[0] == 'DC' and len(parts) >= 2:
            return 'DC', 'DC', parts[1]
        return parts[0], parts[0], parts[-1]
    df[['market_base','market_norm','outcome']] = df.apply(lambda r: pd.Series(split_market(r.get('market',''))), axis=1)
    df.rename(columns={'home_team':'home','away_team':'away'}, inplace=True)
    # Ensure expected columns exist (older logs may miss some)
    required_cols = ['date','league','home','away','market','odds','stake','model_prob','expected_value','market_base','market_norm','outcome']
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan
    # Types for numeric fields
    for c in ('odds','stake','model_prob','expected_value'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def load_results_for_league(league: str) -> pd.DataFrame:
    # prefer processed merged_preprocessed
    proc = os.path.join('data','processed', f'{league}_merged_preprocessed.csv')
    if os.path.exists(proc):
        df = pd.read_csv(proc)
    else:
        raw = os.path.join('data','raw', f'{league}_merged.csv')
        if os.path.exists(raw):
            df = pd.read_csv(raw)
        else:
            return pd.DataFrame(columns=['Date','HomeTeam','AwayTeam','FTHG','FTAG'])
    # normalize
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        except Exception:
            pass
    # outcome shortcuts
    def res_row(r) -> str:
        try:
            h = int(r.get('FTHG')); a = int(r.get('FTAG'))
            return 'H' if h>a else ('A' if a>h else 'D')
        except Exception:
            return None
    df['_res'] = df.apply(res_row, axis=1)
    df['_tot'] = pd.to_numeric(df.get('FTHG'), errors='coerce').fillna(0) + pd.to_numeric(df.get('FTAG'), errors='coerce').fillna(0)
    # standardize team names to model names
    df['HomeTeamStd'] = df['HomeTeam'].astype(str).map(config.normalize_team_name)
    df['AwayTeamStd'] = df['AwayTeam'].astype(str).map(config.normalize_team_name)
    return df[['Date','HomeTeamStd','AwayTeamStd','FTHG','FTAG','_res','_tot']].rename(columns={'HomeTeamStd':'home','AwayTeamStd':'away'})


def settle_bet(row: pd.Series, res: str, tot: float) -> Tuple[bool,bool]:
    """Return (is_settled, won)."""
    mb = str(row.get('market_base'))
    out = str(row.get('outcome'))
    if res is None or (pd.isna(res) if isinstance(res,float) else False):
        return (False, False)
    if mb == '1X2':
        return (True, out == res)
    if mb == 'OU':
        try:
            line = float(str(row.get('market_norm','OU 2.5')).split()[1])
            if out.lower()=='over':
                return (True, float(tot) > line)
            if out.lower()=='under':
                return (True, float(tot) < line)
        except Exception:
            return (False, False)
    if mb == 'TG Interval':
        try:
            a_str, b_str = out.split('-')
            a = int(a_str); b = int(b_str)
            return (True, a <= float(tot) <= b)
        except Exception:
            return (False, False)
    # DC, others: treat as unsettled for now
    return (False, False)


def compute_pnl(df_bets: pd.DataFrame, leagues: List[str]) -> pd.DataFrame:
    # Join results per league
    res_all = []
    for lg in sorted(set([l for l in leagues if isinstance(l,str) and l])):
        res_all.append(load_results_for_league(lg))
    res = pd.concat(res_all, ignore_index=True) if res_all else pd.DataFrame(columns=['date','home','away','_res','_tot'])
    # Create match key by teams; date match is tricky due to kickoff time — we’ll match by teams first per day window.
    bets = df_bets.copy()
    bets['date_d'] = pd.to_datetime(bets['date'], errors='coerce').dt.date
    res['date_d'] = pd.to_datetime(res['Date'], errors='coerce').dt.date
    # naive join by closest date (same day) and teams
    merged = pd.merge(bets, res, left_on=['home','away','date_d'], right_on=['home','away','date_d'], how='left', suffixes=('','_res'))
    # Settle
    settled=[]; won=[]
    for _, r in merged.iterrows():
        is_set, is_won = settle_bet(r, r.get('_res'), r.get('_tot'))
        settled.append(is_set)
        won.append(is_won)
    merged['settled'] = settled
    merged['won'] = won
    merged['pnl'] = np.where(merged['settled'] & merged['won'], (merged['odds'] - 1.0) * merged['stake'], np.where(merged['settled'] & ~merged['won'], -merged['stake'], 0.0))
    return merged


def upcoming_predictions(league: str, top_n: int = 20) -> pd.DataFrame:
    cfg = fusion.load_config()
    cfg['league'] = league
    mb = fusion.generate_market_book(cfg)
    if mb.empty:
        return mb
    # Use real odds when available (BOT_ODDS_MODE=local loads data/odds/{LEAGUE}.json)
    df_odds = fusion.attach_value_metrics(
        fusion._fill_odds_for_df(mb, league, with_odds=True),
        use_placeholders=False,
    )
    # prefer best per market per match
    rows=[]
    for (d,h,a), g in df_odds.groupby(['date','home','away']):
        # pick highest EV among each market base
        def pick(m):
            gg = g[g['market'].astype(str).str.startswith(m)]
            if gg.empty: return None
            return gg.sort_values(['EV','prob'], ascending=[False,False]).iloc[0]
        p1=pick('1X2'); p2=pick('OU '); p3=pick('TG Interval')
        for p in (p1,p2,p3):
            if p is None: continue
            rows.append({'date': d, 'home': h, 'away': a, 'market': p['market'], 'outcome': p['outcome'], 'prob': float(p['prob']), 'odds': float(p['odds']), 'edge': float(p['edge']), 'EV': float(p['EV'])})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(['EV','prob'], ascending=[False, False]).head(top_n)


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf.read()


def main():
    st.set_page_config(page_title='Football Predictions — Monitoring', layout='wide')
    st.title('Monitoring Dashboard')

    # Sidebar filters
    st.sidebar.header('Filters')
    leagues_default = ['E0','D1','F1','I1','SP1']
    leagues = st.sidebar.multiselect('Leagues', options=leagues_default, default=leagues_default)
    date_from = st.sidebar.date_input('From', value=dt.date.today() - dt.timedelta(days=60))
    date_to = st.sidebar.date_input('To', value=dt.date.today() + dt.timedelta(days=1))
    market_opts = ['1X2','OU','TG Interval','DC']
    markets = st.sidebar.multiselect('Markets', options=market_opts, default=market_opts)

    # Bankroll and P/L
    bk = load_bankroll()
    bets = load_bets()
    if not bets.empty:
        bets = bets[bets['league'].isin(leagues)]
        bets = bets[(bets['date']>=pd.to_datetime(date_from)) & (bets['date']<=pd.to_datetime(date_to)+pd.Timedelta(days=1))]
        bets = bets[bets['market_base'].isin(markets)]
    settled = compute_pnl(bets, leagues) if not bets.empty else bets.copy()

    col1,col2,col3,col4 = st.columns(4)
    staked = float(settled['stake'].sum()) if not settled.empty else 0.0
    pnl = float(settled['pnl'].sum()) if not settled.empty else 0.0
    roi = (pnl / staked) if staked>0 else 0.0
    col1.metric('Bankroll', f"{bk:,.2f}")
    col2.metric('Total Staked', f"{staked:,.2f}")
    col3.metric('P/L', f"{pnl:,.2f}", delta=f"{roi*100:.1f}% ROI")
    col4.metric('Bets', f"{len(settled):,}")

    # P/L charts
    st.subheader('P/L by League and Market')
    if not settled.empty:
        g1 = settled.groupby('league', as_index=False)['pnl'].sum()
        g2 = settled.groupby('market_base', as_index=False)['pnl'].sum()
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        ax[0].bar(g1['league'], g1['pnl']); ax[0].set_title('By League'); ax[0].set_ylabel('P/L')
        ax[1].bar(g2['market_base'], g2['pnl']); ax[1].set_title('By Market')
        st.pyplot(fig, use_container_width=True)
        st.download_button('Download P/L Figure (PNG)', data=fig_to_png_bytes(fig), file_name='pl_summary.png', mime='image/png')
    else:
        st.info('No bets to summarize for current filters.')

    # Open vs Settled table
    st.subheader('Bets (Open / Settled)')
    st.dataframe(settled.sort_values('date', ascending=False), use_container_width=True)
    st.download_button('Download Bets (CSV)', data=settled.to_csv(index=False).encode('utf-8'), file_name='bets_filtered.csv', mime='text/csv')

    # Upcoming predictions (top EV)
    st.subheader('Upcoming Predictions (Top EV)')
    tabs = st.tabs(leagues if leagues else ['E0'])
    for lg, tab in zip(leagues if leagues else ['E0'], tabs):
        with tab:
            preds = upcoming_predictions(lg, top_n=25)
            if preds.empty:
                st.write('No upcoming predictions.')
            else:
                st.dataframe(preds, use_container_width=True)
                st.download_button(f'Download {lg} Predictions (CSV)', data=preds.to_csv(index=False).encode('utf-8'), file_name=f'{lg}_predictions.csv', mime='text/csv')


if __name__ == '__main__':
    main()
