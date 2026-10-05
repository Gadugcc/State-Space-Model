import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self,input_size=1,hidden_size=64,num_layers=2,dropout=0.0):
        super().__init__(); self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=dropout if num_layers>1 else 0); self.output=nn.Linear(hidden_size,1)
    def forward(self,x): return self.output(self.lstm(x)[0][:,-1,:])
