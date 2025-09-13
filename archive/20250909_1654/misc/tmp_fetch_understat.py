import re, json, requests, sys
from pathlib import Path
import argparse
import csv
RE_STATE = re.compile(r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\});", re.DOTALL)

def _extract_state(html):
    m = RE_STATE.search(html)
    if not m:
        raise ValueError('INITIAL_STATE not found')
    return json.loads(m.group(1))

def fetch_league_match_ids(league, season):
    url = f"https://understat.com/league/{league}/{season}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    state = _extract_state(r.text)
    matches = state.get('leaguetable', {}).get('matchesData') or state.get('matchesData') or []
    ids = []
    for m in matches:
        try:
            ids.append(int(m.get('id') or m.get('match_id')))
        except Exception:
            pass
    return sorted(set(ids))

def fetch_match_shots(match_id):
    url = f"https://understat.com/match/{match_id}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    state = _extract_state(r.text)
    shots = state.get('match', {}).get('shotsData') or {}
    out = []
    for side in ('h','a'):
        for s in shots.get(side, []) or []:
            try:
                rec = {
                    'match_id': match_id,
                    'team': s.get('h_team') if side=='h' else (s.get('a_team') or s.get('team')),
                    'player': s.get('player'),
                    'minute': s.get('minute'),
                    'x': float(s.get('X') or s.get('x')),
                    'y': float(s.get('Y') or s.get('y')),
                    'result': s.get('result'),
                    'is_goal': 1 if str(s.get('result','')).lower()=='goal' else 0,
                    'body_part': s.get('shotType') or s.get('body_part'),
                    'situation': s.get('situation'),
                    'assist_type': s.get('assist'),
                    'under_pressure': s.get('under_pressure', 0),
                }
                out.append(rec)
            except Exception:
                pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--league', default='EPL')
    ap.add_argument('--seasons', nargs='+', type=int, required=True)
    ap.add_argument('--out_dir', default='data/understat')
    ap.add_argument('--limit', type=int, default=50)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_ids = set()
    for s in args.seasons:
        for mid in fetch_league_match_ids(args.league, s):
            all_ids.add(mid)
    ids = sorted(all_ids)
    if args.limit:
        ids = ids[:args.limit]
    print('Fetching', len(ids), 'matches')
    rows = []
    for i, mid in enumerate(ids, 1):
        try:
            shots = fetch_match_shots(mid)
            (out_dir / f"match_{mid}.json").write_text(json.dumps({'shots': shots}, ensure_ascii=False), encoding='utf-8')
            rows.extend(shots)
            print(f"[{i}/{len(ids)}] match {mid}, shots={len(shots)}")
        except Exception as e:
            print(f"[{i}/{len(ids)}] failed {mid}: {e}")
    # write combined CSV for quick training prototype
    csv_path = Path('data/shots'); csv_path.mkdir(parents=True, exist_ok=True)
    out_csv = csv_path / 'understat_shots.csv'
    if rows:
        cols = ['match_id','team','player','minute','x','y','result','is_goal','body_part','situation','assist_type','under_pressure']
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in cols})
        print('Saved CSV ->', out_csv)
    else:
        print('No shots collected')

if __name__ == '__main__':
    main()
