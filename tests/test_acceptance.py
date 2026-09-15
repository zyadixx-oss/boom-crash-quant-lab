from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_required_project_surfaces_exist():
    for rel in ['backend/app/main.py','frontend/src/main.tsx','frontend/src/MarketChart.tsx','docs/RESULTS.md','scripts/run_shadow.py','.env.example']:
        assert (ROOT/rel).exists(), rel

def test_no_live_trade_endpoint_or_true_safety_default():
    text=(ROOT/'backend/app/main.py').read_text()+ (ROOT/'.env.example').read_text()
    assert 'LIVE_TRADING=false' in text
    assert 'live_allowed=false' in text
    assert "@app.post('/buy')" not in text and "@app.post('/trade')" not in text
