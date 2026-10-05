import numpy as np

def mse(y_true,y_pred): return float(np.mean((np.asarray(y_true)-np.asarray(y_pred))**2))
def rmse(y_true,y_pred): return float(np.sqrt(mse(y_true,y_pred)))
def mae(y_true,y_pred): return float(np.mean(np.abs(np.asarray(y_true)-np.asarray(y_pred))))
