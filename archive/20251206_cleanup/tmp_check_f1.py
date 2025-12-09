import os, datetime as dt, requests, json
key=os.getenv('API_FOOTBALL_ODDS_KEY') or os.getenv('API_FOOTBALL_KEY')
print('key?', bool(key))
base='https://v3.football.api-sports.io'
headers={'x-apisports-key': key} if key else {}
leagues={'E0':39,'D1':78,'F1':61,'I1':135,'SP1':140}
lg='F1'
league_id=leagues[lg]
today=dt.date.today(); to=today+dt.timedelta(days=10)
params={'league': league_id, 'season': (today.year if today.month>=8 else today.year-1), 'from': today.isoformat(), 'to': to.isoformat()}
print('params', params)
r=requests.get(f"{base}/fixtures", params=params, headers=headers, timeout=20)
print('status', r.status_code)
try:
    data=r.json()
except Exception as e:
    print('json err', e); data={}
resp=data.get('response', [])
print('count', len(resp))
for it in resp[:20]:
    fx=it.get('fixture',{}); teams=it.get('teams',{})
    print((fx.get('id'), (fx.get('date') or '')[:10], (teams.get('home') or {}).get('name'), (teams.get('away') or {}).get('name')))
