"""
Verificações numéricas das deduções (para mostrar no seminário):
  1. SSM LTI: modo convolucional (FFT) == modo recorrente           (seções 7–9)
  2. Mamba:   scan em blocos (paralelo) == recorrência sequencial   (seção 17)
  3. Teorema 1 do Mamba: N=1, A=-1, B=1, ZOH  ⇒  h_t = (1-σ(z))h_{t-1} + σ(z)x_t   (seção 15)
  4. Associatividade do operador do scan                             (seção 17)
"""
import torch
import torch.nn.functional as F

from models.mamba import SelectiveSSM
from models.ssm import S4DLayer

torch.manual_seed(0)

# 1
layer = S4DLayer(d_model=8, state_size=16)
u = torch.randn(2, 300, 8)
print(f"1) S4D  conv vs recorrente   max|dif| = {(layer(u) - layer.forward_recurrent(u)).abs().max():.2e}")

# 2
ssm = SelectiveSSM(d_model=16, d_state=8, dt_rank=4)
x = torch.randn(2, 300, 16)
with torch.no_grad():
    d = (ssm(x) - ssm(x, reference=True)).abs().max()
print(f"2) Mamba scan blocos vs laço max|dif| = {d:.2e}")

# 3
z, xs = torch.randn(200), torch.randn(200)
delta = F.softplus(z)
a_bar = torch.exp(-delta)                 # exp(ΔA), A = -1
b_bar = (a_bar - 1) / (-1.0)              # ZOH: (e^{ΔA}-1)/A · B, B = 1
h1 = h2 = torch.tensor(0.0)
err = 0.0
for t in range(200):
    h1 = a_bar[t] * h1 + b_bar[t] * xs[t]
    g = torch.sigmoid(z[t])
    h2 = (1 - g) * h2 + g * xs[t]
    err = max(err, (h1 - h2).abs().item())
print(f"3) Teorema 1 (SSM seletivo == RNN com porta) max|dif| = {err:.2e}")

# 4
op = lambda p, q: (p[0] * q[0], q[0] * p[1] + q[1])
p, q, r = [(torch.rand(1), torch.randn(1)) for _ in range(3)]
l, rr = op(op(p, q), r), op(p, op(q, r))
print(f"4) Associatividade: |dif| = {max((l[0]-rr[0]).abs().item(), (l[1]-rr[1]).abs().item()):.2e}")
