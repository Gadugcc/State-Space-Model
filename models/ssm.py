"""
SSM linear e invariante no tempo (LTI) no estilo S4D (Gu et al., 2022).

Modelo contínuo:      h'(t) = A h(t) + B u(t),     y(t) = C h(t) + D u(t)
Discretização (ZOH):  Ā = exp(ΔA),   B̄ = (ΔA)^{-1}(exp(ΔA) - I) ΔB
Recorrência:          h_k = Ā h_{k-1} + B̄ u_k,     y_k = C h_k + D u_k
Convolução:           y = K * u,  com  K_k = C Ā^k B̄   (k = 0..L-1)

Como Ā, B̄ e C NÃO dependem da entrada (LTI), o kernel K pode ser pré-computado
e a sequência inteira é processada por uma convolução causal via FFT
(modo "convolucional", usado no treino). O modo recorrente (`step`) é usado
apenas para verificar que os dois modos são equivalentes.

A é diagonal e complexa (inicialização S4D-Lin: a_n = -1/2 + i·π·n); como a
entrada é real, usamos N/2 pares conjugados e K_k = 2·Re(Σ_n C_n B̄_n Ā_n^k).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class S4DLayer(nn.Module):
    def __init__(self, d_model=32, state_size=32, dt_min=1e-3, dt_max=1e-1, complex_A=True):
        super().__init__()
        H, N2 = d_model, state_size // 2
        self.H, self.N2 = H, N2
        # Δ por canal, log-uniforme em [dt_min, dt_max]
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        # A = -exp(log_A_real) + i·A_imag   (parte real < 0  ⇒  |Ā| < 1, estável)
        if complex_A:   # S4D-Lin: a_n = -1/2 + iπn  (kernel com oscilações)
            self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(H, N2)))
            self.A_imag = nn.Parameter(math.pi * torch.arange(N2).float().repeat(H, 1))
        else:           # S4D-Real (o mesmo do Mamba): a_n = -(n+1), kernel só decai
            self.log_A_real = nn.Parameter(torch.log(torch.arange(1, N2 + 1).float()).repeat(H, 1))
            self.register_buffer("A_imag", torch.zeros(H, N2))
        # B = 1 (absorvido em C, como no S4D); C complexo aleatório
        self.C = nn.Parameter(torch.randn(H, N2, 2) * (0.5 ** 0.5))
        self.D = nn.Parameter(torch.randn(H))

    def discretize(self):
        dt = torch.exp(self.log_dt)                                   # (H)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag            # (H, N2)
        dtA = A * dt[:, None]
        A_bar = torch.exp(dtA)                                        # ZOH
        B_bar = (A_bar - 1.0) / A                                     # (ΔA)^{-1}(e^{ΔA}-1)ΔB com B=1
        C = torch.view_as_complex(self.C)
        return A_bar, B_bar, C

    def kernel(self, L):
        A_bar, B_bar, C = self.discretize()
        k = torch.arange(L, device=A_bar.device)
        vander = torch.exp(torch.log(A_bar)[:, :, None] * k)         # Ā^k  (H, N2, L)
        K = 2 * torch.einsum("hn,hnl->hl", C * B_bar, vander).real   # (H, L)
        return K

    def forward(self, u):                     # u: (batch, L, H)
        L = u.size(1)
        K = self.kernel(L)
        u_ = u.transpose(1, 2)                # (b, H, L)
        n = 2 * L                             # zero-padding ⇒ convolução linear (causal), não circular
        y = torch.fft.irfft(torch.fft.rfft(u_, n) * torch.fft.rfft(K, n), n)[..., :L]
        y = y + self.D[None, :, None] * u_
        return y.transpose(1, 2)

    @torch.no_grad()
    def forward_recurrent(self, u):           # mesmo resultado, O(L) passos sequenciais
        A_bar, B_bar, C = self.discretize()
        b, L, H = u.shape
        h = torch.zeros(b, H, self.N2, dtype=A_bar.dtype, device=u.device)
        ys = []
        for t in range(L):
            h = A_bar * h + B_bar * u[:, t, :, None]
            ys.append(2 * (C * h).sum(-1).real + self.D * u[:, t])
        return torch.stack(ys, 1)


class SSMModel(nn.Module):
    """Entrada (b, L, 1) → projeção → [LayerNorm → S4D → GELU → Linear] × num_layers (residual) → ŷ."""

    def __init__(self, state_size=32, d_model=32, num_layers=2, dropout=0.0, use_norm=True, complex_A=True):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) if use_norm else nn.Identity() for _ in range(num_layers)])
        self.layers = nn.ModuleList([S4DLayer(d_model, state_size, complex_A=complex_A) for _ in range(num_layers)])
        self.mix = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        for norm, ssm, mix in zip(self.norms, self.layers, self.mix):
            x = x + self.dropout(mix(F.gelu(ssm(norm(x)))))
        return self.output(x[:, -1, :])
