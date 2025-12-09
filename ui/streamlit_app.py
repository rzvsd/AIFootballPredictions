"""
Streamlit Dashboard â€” Monitoring (Phase 4)

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

import glob

import config


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
    # Create match key by teams; date match is tricky due to kickoff time â€” weâ€™ll match by teams first per day window.
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
    """Build shopping-list style view: best 1X2, OU, TG per match using current engine data."""
    cols = ['date','home','away','market','outcome','prob','fair_odds','book_odds','price_source','odds','edge','EV']

    # Prefer live computation (ensures all matches/markets shown), fallback to latest report if compute fails
    def _from_engine() -> pd.DataFrame:
        try:
            cfg = fusion.load_config()
            cfg['league'] = league
            # respect BOT_FIXTURES_DAYS if set; otherwise default inside engine
            days_env = os.getenv('BOT_FIXTURES_DAYS')
            if days_env:
                cfg['fixtures_days'] = int(days_env)
            df = fusion.generate_market_book(cfg)
            if df is None or df.empty:
                return pd.DataFrame(columns=cols)
            # attach real odds from lookup first
            df_odds = fusion._fill_odds_for_df(df, league, with_odds=True)
            df_val = fusion.attach_value_metrics(df_odds, use_placeholders=True, league_code=league)
            missing = [c for c in cols if c not in df_val.columns]
            for c in missing:
                df_val[c] = np.nan
            return df_val[cols].copy()
        except Exception:
            return pd.DataFrame(columns=cols)

    df = _from_engine()
    if df.empty:
        paths = sorted(glob.glob(os.path.join('reports', f'{league}_*_picks.csv')))
        if not paths:
            return pd.DataFrame(columns=cols)
        latest = paths[-1]
        try:
            df = pd.read_csv(latest)
        except Exception:
            return pd.DataFrame(columns=cols)
        missing = [c for c in cols if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        df = df[cols].copy()

    for c in ('prob','odds','edge','EV','fair_odds','book_odds'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # pick best market per match (1X2, OU, TG interval) by highest probability (shopping list)
    rows = []
    for (date, home, away), group in df.groupby(['date','home','away']):
        def pick(prefix: str, exact: bool = False):
            mask = group['market'].astype(str) == prefix if exact else group['market'].astype(str).str.startswith(prefix)
            sub = group[mask]
            if sub.empty:
                return None
            return sub.sort_values(['prob','EV'], ascending=[False, False]).iloc[0]
        selections = [
            pick('1X2', exact=True),
            pick('OU '),
            pick('TG Interval'),
        ]
        for sel in selections:
            if sel is None:
                continue
            rows.append(sel.to_dict())
    out = pd.DataFrame(rows, columns=cols).dropna(subset=['date','home','away','market'])
    if out.empty:
        return out
    return out.sort_values(['date','home','away','market','prob'], ascending=[True, True, True, True, False])


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf.read()


def main():
    st.set_page_config(page_title='Football Predictions â€” Monitoring', layout='wide')
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
                display_df = preds.copy()
                for col in ('prob','edge','odds','EV'):
                    if col in display_df.columns:
                        display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                if 'edge' in display_df.columns:
                    display_df['edge'] = display_df['edge'] * 100.0

                def adjust_row(row):
                    fair = float(row.get('fair_odds', np.nan)) if not pd.isna(row.get('fair_odds')) else np.nan
                    book = row.get('book_odds', np.nan)
                    book = float(book) if not pd.isna(book) else np.nan
                    tag = str(row.get('price_source', '') or '').lower()
                    if tag not in ('real', 'synth'):
                        tag = 'synth' if pd.isna(book) else 'real'
                    ev_val = float(row.get('EV', np.nan)) if (not pd.isna(row.get('EV')) and tag == 'real') else np.nan
                    return pd.Series({'fair_odds': fair, 'book_odds': book, 'disp_ev': ev_val, 'price_tag': tag})

                display_df = pd.concat([display_df, display_df.apply(adjust_row, axis=1)], axis=1)
                # Ensure unique columns for Styler and reset index
                display_df = display_df.loc[:, ~display_df.columns.duplicated()].reset_index(drop=True)

                # Compact per-match view: TG Interval, 1X2, Over/Under side by side
                def _best_pick(group: pd.DataFrame, predicate) -> pd.Series | None:
                    sub = group[predicate(group['market'].astype(str))]
                    if sub.empty:
                        return None
                    sort_cols = ['disp_ev' if 'disp_ev' in sub.columns else 'EV', 'prob']
                    return sub.sort_values(sort_cols, ascending=[False, False]).iloc[0]

                rows = []
                for (date, home, away), g in preds.groupby(['date','home','away']):
                    tg = _best_pick(g, lambda s: s.str.startswith('TG Interval'))
                    one = _best_pick(g, lambda s: s == '1X2')
                    ou = _best_pick(g, lambda s: s.str.startswith('OU '))
                    def fmt(row):
                        if row is None:
                            return '-'
                        prob = float(row.get('prob', np.nan))
                        fair = float(row.get('fair_odds', np.nan))
                        odds = row.get('book_odds', np.nan)
                        tag = str(row.get('price_tag','')).upper()
                        tag_str = f" [{tag}]" if tag else ""
                        odds_str = f"{float(odds):.2f}" if pd.notna(odds) else "n/a"
                        fair_str = f"{fair:.2f}+" if pd.notna(fair) else "n/a"
                        return f"{row['market']} {row['outcome']} | p={prob*100:.1f}% | target>={fair_str} | book={odds_str}{tag_str}"
                    rows.append({
                        'Kickoff': pd.to_datetime(date, errors='coerce'),
                        'Match': f"{home} vs {away}",
                        'TG Interval': fmt(tg),
                        '1X2': fmt(one),
                        'Over/Under': fmt(ou),
                    })
                if rows:
                    match_df = pd.DataFrame(rows).sort_values('Kickoff')
                    st.markdown('**Per-match best picks (by EV per market)**')
                    st.dataframe(
                        match_df,
                        column_config={
                            'Kickoff': st.column_config.DatetimeColumn('Kickoff', format='D MMM HH:mm'),
                            'Match': st.column_config.TextColumn('Match'),
                            'TG Interval': st.column_config.TextColumn('TG Interval'),
                            '1X2': st.column_config.TextColumn('1X2'),
                            'Over/Under': st.column_config.TextColumn('Over/Under'),
                        },
                        width='stretch',
                        hide_index=True,
                    )
                # Full list in an expander, grouped view to avoid duplicate rows at top-level
                with st.expander('Full list (per bet)', expanded=False):
                    column_cfg = {
                        'date': st.column_config.DatetimeColumn('Kickoff', format='D MMM HH:mm'),
                        'home': 'Home Team',
                        'away': 'Away Team',
                        'market': st.column_config.TextColumn('Market', help='Type of bet'),
                        'outcome': st.column_config.TextColumn('Pick', help='Target result'),
                        'prob': st.column_config.ProgressColumn(
                            'Probability',
                            format='%.1f%%',
                            min_value=0.0,
                            max_value=1.0,
                            help='Model confidence',
                        ),
                        'fair_odds': st.column_config.NumberColumn('Target/Fair Odds', format='%.2f'),
                        'book_odds': st.column_config.NumberColumn('Book Odds', format='%.2f'),
                        'edge': st.column_config.NumberColumn(
                            'Edge',
                            format='%.1f%%',
                            help='Advantage over bookie'
                        ),
                        'EV': st.column_config.NumberColumn('EV (raw)', format='%.2f'),
                        'disp_ev': st.column_config.NumberColumn('EV (fair)', format='%.2f'),
                        'price_source': st.column_config.TextColumn('Price Source', help='real = feed price, synth = derived fair'),
                        'price_tag': st.column_config.TextColumn('Price Tag', help='real = feed price, synth = derived fair'),
                    }

                    def highlight_high_ev(row):
                        try:
                            ev_val = float(row.get('disp_ev', 0) or 0)
                            if ev_val > 1.10:
                                return ['background-color: #d4edda; color: black'] * len(row)
                            if ev_val > 0.5:
                                return ['background-color: #f0f7fb; color: black'] * len(row)
                        except Exception:
                            pass
                        return [''] * len(row)

                    styled = display_df.style.apply(highlight_high_ev, axis=1)
                    st.dataframe(
                        styled,
                        column_config=column_cfg,
                        width='stretch',
                        hide_index=True,
                        height=600,
                    )
                    st.download_button(
                        f'Download {lg} Predictions (CSV)',
                        data=preds.to_csv(index=False).encode('utf-8'),
                        file_name=f'{lg}_predictions.csv',
                        mime='text/csv',
                    )

                st.markdown("""
**Legend**
- Probability: model-derived win/probability for the market (xG → Poisson/NegBin score matrix → market prob; calibrated where configured).
- Odds (fair/synth): if the feed price is missing or placeholder, we derive a synthetic bookmaker-like price from the model prob plus a margin (`BOT_SYNTH_MARGIN`, default 6%).
- Edge: model probability minus implied probability from the displayed odds.
- EV: expected value using the displayed (real or synthetic) odds; EV>0 implies positive theoretical return.
- Price Type: REAL = from odds feed; SYNTH = derived fair price (no bookmaker quote available).
""")


if __name__ == '__main__':
    main()


