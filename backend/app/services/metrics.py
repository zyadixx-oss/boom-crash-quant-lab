import numpy as np

def classification_metrics(y_true,y_pred):
    tp=sum(1 for a,b in zip(y_true,y_pred) if a and b); fp=sum(1 for a,b in zip(y_true,y_pred) if not a and b); fn=sum(1 for a,b in zip(y_true,y_pred) if a and not b)
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; f1=2*p*r/(p+r) if p+r else 0
    return {'precision':p,'recall':r,'f1':f1,'false_signal_rate':fp/(tp+fp) if tp+fp else 0}

def trading_metrics(pnls:list[float]):
    if not pnls: return {'total_trades':0,'wins':0,'losses':0,'win_rate':0,'profit_factor':0,'expectancy':0,'average_win':0,'average_loss':0,'max_drawdown':0,'max_consecutive_losses':0,'recovery_factor':0,'sharpe':0,'sortino':0}
    a=np.array(pnls,float); wins=a[a>0]; losses=a[a<=0]; gross_w=wins.sum(); gross_l=abs(losses.sum()); equity=np.cumsum(a); peaks=np.maximum.accumulate(np.r_[0,equity])[1:]; dd=peaks-equity; maxdd=float(dd.max(initial=0)); maxcl=cur=0
    for v in a:
        cur=cur+1 if v<=0 else 0; maxcl=max(maxcl,cur)
    std=a.std(ddof=1) if len(a)>1 else 0; downside=a[a<0].std(ddof=1) if (a<0).sum()>1 else 0
    return {'total_trades':len(a),'wins':len(wins),'losses':len(losses),'win_rate':len(wins)/len(a),'profit_factor':float(gross_w/gross_l) if gross_l else (999.0 if gross_w else 0),'expectancy':float(a.mean()),'average_win':float(wins.mean()) if len(wins) else 0,'average_loss':float(losses.mean()) if len(losses) else 0,'max_drawdown':maxdd,'max_consecutive_losses':maxcl,'recovery_factor':float(equity[-1]/maxdd) if maxdd else 0,'sharpe':float(a.mean()/std*np.sqrt(len(a))) if std else 0,'sortino':float(a.mean()/downside*np.sqrt(len(a))) if downside else 0}
