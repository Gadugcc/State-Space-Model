import time, numpy as np
from sklearn.linear_model import LinearRegression

def make_panel_windows(panel,context,horizon=1):
    X=[]; y=[]; ids=[]; n_ent,n_time=panel.shape
    for i in range(n_ent):
        for t in range(context,n_time-horizon+1): X.append(panel[i,t-context:t]); y.append(panel[i,t+horizon-1]); ids.append(i)
    return np.asarray(X,np.float32),np.asarray(y,np.float32),np.asarray(ids,np.int64)

class PanelLagRegression:
    def __init__(self,n_entities,use_entity_effects=True): self.n_entities=n_entities; self.use_entity_effects=use_entity_effects; self.model=LinearRegression()
    def _features(self,X,ids):
        # float64: com ~600 mil linhas, mínimos quadrados em float32 perde precisão (RMSE 0,126 vs 0,100)
        X=np.asarray(X,dtype=np.float64)
        if not self.use_entity_effects: return X
        return np.concatenate([X,np.eye(self.n_entities,dtype=np.float64)[ids]],axis=1)
    def fit(self,X,y,ids):
        s=time.perf_counter(); self.model.fit(self._features(X,ids),y); self.train_time=time.perf_counter()-s; return self
    def predict(self,X,ids):
        s=time.perf_counter(); p=self.model.predict(self._features(X,ids)); self.inference_time=time.perf_counter()-s; return p
    @property
    def parameters(self): return int(self.model.coef_.size+1)
