from fastapi.testclient import TestClient
from app.main import app

def test_health_and_symbols():
    with TestClient(app) as c:
        h=c.get('/health'); assert h.status_code==200; assert h.json()['live_trading'] is False; assert h.json()['opened_trades'] is False
        s=c.get('/symbols'); assert s.status_code==200 and len(s.json())==10
