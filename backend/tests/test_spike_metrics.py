from app.services.metrics import trading_metrics, classification_metrics

def test_metrics():
    m=trading_metrics([2,-1,3,-1]); assert m['win_rate']==0.5; assert m['profit_factor']==2.5
    c=classification_metrics([True,False,True],[True,True,False]); assert round(c['precision'],2)==0.5 and round(c['recall'],2)==0.5
