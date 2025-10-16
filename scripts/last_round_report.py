"""
Last Round Report (API-Football rounds)

Pulls the latest completed round for a league/season using API-Football v3
and prints a compact, human-friendly per-match summary with model picks and
final scores + correctness flags (1X2/OU 2.5/TG interval).

Usage:
  python -m scripts.last_round_report --league SP1 [--season 2025] [--expect-n 10]

Env:
  API_FOOTBALL_KEY (or API_FOOTBALL_ODDS_KEY)

Notes:
  - Does NOT retrain. It builds a frozen snapshot at the earliest fixture time
    to avoid post-round leakage, then uses the existing engine to produce probabilities.
  - If the current round is incomplete, it automatically falls back to the previous round.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import pandas as pd

import config
import bet_fusion as fusion

API_BASE = "https://v3.football.api-sports.io"
LEAGUE_IDS: Dict[str, int] = {'E0':39,'D1':78,'F1':61,'I1':135,'SP1':140}

def _headers() -> Dict[str,str]:
    key = (os.getenv('API_FOOTBALL_ODDS_KEY') or os.getenv('API_FOOTBALL_KEY'))
    if not key:
        raise SystemExit('Missing API key: set API_FOOTBALL_KEY or API_FOOTBALL_ODDS_KEY')
    return {'x-apisports-key': key}

def _season_for_today() -> int:
    d = pd.Timestamp.today().date()
    return d.year if d.month >= 8 else d.year - 1

def _get_current_round(league_id:int, season:int) -> str|None:
    r = requests.get(f"{API_BASE}/fixtures/rounds", params={'league':league_id,'season':season,'current':'true'}, headers=_headers(), timeout=20)
    if r.status_code!=200: return None
    arr = r.json().get('response') or []
    return str(arr[0]) if arr else None

def _get_all_rounds(league_id:int, season:int) -> List[str]:
    r = requests.get(f"{API_BASE}/fixtures/rounds", params={'league':league_id,'season':season}, headers=_headers(), timeout=20)
    if r.status_code!=200: return []
    return [str(x) for x in (r.json().get('response') or [])]

def _fetch_round_fixtures(league_id:int, season:int, round_label:str) -> List[Dict]:
    r = requests.get(f"{API_BASE}/fixtures", params={'league':league_id,'season':season,'round':round_label,'status':'FT-AET-PEN'}, headers=_headers(), timeout=20)
    if r.status_code!=200: return []
    return r.json().get('response') or []

def _choose_last_completed_round(league_id:int, season:int, expect_n:int) -> tuple[str|None, List[Dict]]:
    cur = _get_current_round(league_id, season)
    rounds = _get_all_rounds(league_id, season)
    if cur:
        fx = _fetch_round_fixtures(league_id, season, cur)
        if len(fx) >= max(1, expect_n//2):
            return cur, fx
    if cur and rounds and cur in rounds:
        try:
            idx = rounds.index(cur); prev_lbl = rounds[max(0, idx-1)]
        except Exception:
            prev_lbl = rounds[-1] if rounds else None
    else:
        prev_lbl = rounds[-1] if rounds else None
    if prev_lbl:
        fx = _fetch_round_fixtures(league_id, season, prev_lbl)
        return prev_lbl, fx
    return None, []

def _write_fixtures_csv(league:str, fixtures:List[Dict]) -> Path:
    out_dir = Path('data')/'fixtures'; out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{league}_last_round.csv"
    rows = []
    for it in fixtures:
        fx = it.get('fixture') or {}; teams = it.get('teams') or {}
        d = str((fx.get('date') or '').replace('T',' ').replace('Z','')).strip()
        h = config.normalize_team_name((teams.get('home') or {}).get('name') or '')
        a = config.normalize_team_name((teams.get('away') or {}).get('name') or '')
        if h and a and d:
            rows.append({'date':d,'home':h,'away':a})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

def _compact_line(idx:int, d:str, h:str, a:str, k1:str, p1:float, ou:str, po:float, tg:str, pt:float, fhg:int|None, fag:int|None) -> str:
    def pct(x):
        try: return f"{float(x)*100:.0f}%"
        except Exception: return ''
    tail = ''
    if fhg is not None and fag is not None:
        act = '1' if fhg>fag else ('X' if fhg==fag else '2')
        ok1 = 'OK' if k1==act else 'KO'
        try:
            tot = fhg+fag
            if str(ou).startswith('over'):
                th = float(ou.split()[1].replace('g','')); okou = 'OK' if tot>th else 'KO'
            elif str(ou).startswith('under'):
                th = float(ou.split()[1].replace('g','')); okou = 'OK' if tot<th else 'KO'
            else:
                okou = ''
            iv = tg.replace('g','') if isinstance(tg,str) else ''
            a_,b_ = [int(x) for x in iv.split('-')] if '-' in iv else (None,None)
            oktg = 'OK' if (a_ is not None and b_ is not None and tot>=a_ and tot<=b_) else ('KO' if (a_ is not None) else '')
        except Exception:
            okou = oktg = ''
        tail = f" | final: {fhg}-{fag} [{ok1} {okou} {oktg}]"
    dt_str = pd.to_datetime(d, errors='coerce').strftime('%Y-%m-%d %H:%M') if d else ''
    return f"{dt_str} | game {idx}) {h} vs {a} : 1x2 => {k1} / {pct(p1)} | over/under => {ou} {pct(po)} | goal interval : {tg} / {pct(pt)}{tail}"

def main() -> None:
    ap = argparse.ArgumentParser(description='Last completed round report with final scores (API-Football rounds)')
    ap.add_argument('--league', required=True)
    ap.add_argument('--season', type=int, default=None)
    ap.add_argument('--expect-n', type=int, default=10)
    args = ap.parse_args()
    lg = args.league.strip().upper(); league_id = LEAGUE_IDS.get(lg)
    if not league_id: raise SystemExit(f'Unknown league code: {lg}')
    season = int(args.season or _season_for_today())
    lbl, fixtures = _choose_last_completed_round(league_id, season, expect_n=args.expect_n)
    if not fixtures: raise SystemExit('No completed fixtures found for last round (check API key/plan and season).')
    fx_csv = _write_fixtures_csv(lg, fixtures)
    earliest = min([pd.to_datetime((it.get('fixture') or {}).get('date'), errors='coerce') for it in fixtures])
    earliest_str = pd.to_datetime(earliest).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(earliest) else None
    cfg = fusion.load_config(); cfg['league']=lg; cfg['fixtures_csv']=str(fx_csv)
    if earliest_str:
        os.environ['BOT_SNAPSHOT_AS_OF']=earliest_str; os.environ['BOT_FIXTURES_FROM']=earliest_str[:10]
    mb = fusion.generate_market_book(cfg)
    if mb.empty: raise SystemExit('No markets generated for last round fixtures.')
    finals = {}
    for it in fixtures:
        teams = it.get('teams') or {}
        h = config.normalize_team_name((teams.get('home') or {}).get('name') or '')
        a = config.normalize_team_name((teams.get('away') or {}).get('name') or '')
        gh = (it.get('goals') or {}).get('home'); ga = (it.get('goals') or {}).get('away')
        finals[(h,a)] = (gh,ga)
    rows = []
    for (d,h,a), g in mb.groupby(['date','home','away'], dropna=False):
        def gp(mkt,out):
            r = g[(g['market'].astype(str)==mkt) & (g['outcome'].astype(str)==out)]
            return float(r['prob'].iloc[0]) if not r.empty else float('nan')
        vals={'1':gp('1X2','H'),'X':gp('1X2','D'),'2':gp('1X2','A')}
        k1 = max(vals, key=lambda k: (-1.0 if pd.isna(vals[k]) else vals[k])); p1=vals[k1]
        pov,pun = gp('OU 2.5','Over'), gp('OU 2.5','Under')
        if not pd.isna(pov) and not pd.isna(pun):
            ou,po = (f'over 2.5g',pov) if pov>=pun else (f'under 2.5g',pun)
        else:
            ous = g[g['market'].astype(str).str.startswith('OU ')]
            if not ous.empty:
                row = ous.sort_values('prob', ascending=False).iloc[0]
                line = str(row['market']).split(' ',1)[1] if ' ' in str(row['market']) else str(row['market']).replace('OU ','')
                ou,po = (f"{str(row['outcome']).lower()} {line}g", float(row['prob']))
            else:
                ou,po = ('n/a', float('nan'))
        tgs = g[g['market'].astype(str)=='TG Interval']
        if not tgs.empty:
            row = tgs.sort_values('prob', ascending=False).iloc[0]
            tg,pt = (f"{row['outcome']}g", float(row['prob']))
        else:
            tg,pt = ('n/a', float('nan'))
        fhg,fag = finals.get((str(h),str(a)), (None,None))
        rows.append(_compact_line(len(rows)+1, str(d), str(h), str(a), k1, p1, ou, po, tg, pt, fhg, fag))
    print(f"Round: {lbl}")
    for line in rows:
        print(line)

if __name__ == '__main__':
    main()
