"""
Mamba educacional em PyTorch puro (Gu & Dao, 2023) — "Selective SSM".

Diferença para o SSM LTI: Δ, B e C passam a ser FUNÇÕES DA ENTRADA x_t:
    Δ_t = softplus(W_Δ x_t + b_Δ),   B_t = W_B x_t,   C_t = W_C x_t
    Ā_t = exp(Δ_t A),                B̄_t = Δ_t B_t      (aprox. de Euler para B, como no código oficial)
    h_t = Ā_t ⊙ h_{t-1} + B̄_t x_t,   y_t = C_t h_t + D x_t

Como os parâmetros variam no tempo, não existe mais um kernel fixo K: a
convolução via FFT deixa de valer e é preciso um SCAN. Aqui usamos um scan
por blocos (chunks): dentro de cada bloco, com S_t = Σ_{r≤t} Δ_r A,
    h_t = exp(S_t) · ( h_0 + Σ_{s≤t} exp(-S_s) B̄_s x_s )
(somas cumulativas, paralelas), e entre blocos o estado é propagado
sequencialmente. `selective_scan_ref` é a recorrência literal, usada para
verificação.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan_ref(u, delta, A, B, C, D):
    """Recorrência sequencial literal. u, delta: (b,L,d); A: (d,n); B, C: (b,L,n)."""
    b, L, d = u.shape
    h = torch.zeros(b, d, A.size(1), dtype=A.dtype if A.is_complex() else u.dtype, device=u.device)
    ys = []
    for t in range(L):
        A_bar = torch.exp(delta[:, t, :, None] * A)
        h = A_bar * h + delta[:, t, :, None] * B[:, t, None, :] * u[:, t, :, None]
        y = (h * C[:, t, None, :]).sum(-1)
        ys.append((2 * y.real if torch.is_complex(y) else y) + D * u[:, t])
    return torch.stack(ys, 1)


def selective_scan(u, delta, A, B, C, D, chunk=16):
    """Scan por blocos, matematicamente idêntico a `selective_scan_ref`."""
    b, L, d = u.shape
    # log(Ā_t) = Δ_t A ≤ 0. Limite inferior só por estabilidade numérica de exp(-S) no bloco:
    # Ā ≥ e^{-80/chunk} já é praticamente "esquecimento total" em um passo.
    dA = delta[..., None] * A                                              # (b,L,d,n)
    if dA.is_complex():   # A complexo: o limite vale para a parte real (é ela que dá |Ā|)
        dA = torch.complex(torch.clamp(dA.real, min=-80.0 / chunk), dA.imag)
    else:
        dA = torch.clamp(dA, min=-80.0 / chunk)
    Bx = (delta * u)[..., None] * B[:, :, None, :]                         # (b,L,d,n)
    h = torch.zeros(b, d, A.size(1), dtype=dA.dtype, device=u.device)
    ys = []
    for c in range(0, L, chunk):
        S = torch.cumsum(dA[:, c:c + chunk], 1)
        hc = torch.exp(S) * (h[:, None] + torch.cumsum(torch.exp(-S) * Bx[:, c:c + chunk], 1))
        y = (hc * C[:, c:c + chunk, None, :]).sum(-1)
        ys.append(2 * y.real if torch.is_complex(y) else y)   # par conjugado ⇒ 2·Re
        h = hc[:, -1]
    return torch.cat(ys, 1) + D * u


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, dt_rank=4, dt_min=1e-3, dt_max=1e-1, complex_A=False):
        super().__init__()
        self.d_state, self.dt_rank, self.complex_A = d_state, dt_rank, complex_A
        # com A complexo, C também é complexo (parte real + imaginária)
        self.x_proj = nn.Linear(d_model, dt_rank + (3 if complex_A else 2) * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model)
        # Inicialização do artigo: Δ ~ LogUniforme[dt_min, dt_max]  ⇒  bias = softplus^{-1}(Δ)
        dt = torch.exp(torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            nn.init.uniform_(self.dt_proj.weight, -dt_rank ** -0.5, dt_rank ** -0.5)
        if complex_A:   # S4D-Lin: A_n = -1/2 + iπn  (permite kernels oscilatórios / sazonais)
            self.A_log = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
            self.A_imag = nn.Parameter(math.pi * torch.arange(d_state, dtype=torch.float32).repeat(d_model, 1))
        else:           # S4D-Real (padrão do Mamba): A_n = -(n+1)
            self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))

    def params(self, x):
        p = self.x_proj(x)
        if self.complex_A:
            dt, B, Cr, Ci = torch.split(p, [self.dt_rank] + [self.d_state] * 3, dim=-1)
            C = torch.complex(Cr, Ci)
            A = torch.complex(-torch.exp(self.A_log), self.A_imag)
        else:
            dt, B, C = torch.split(p, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            A = -torch.exp(self.A_log)                 # (d,n)
        delta = F.softplus(self.dt_proj(dt))           # (b,L,d)  — seleção: depende de x_t
        return delta, A, B, C

    def forward(self, x, reference=False):
        delta, A, B, C = self.params(x)
        f = selective_scan_ref if reference else selective_scan
        return f(x, delta, A, B, C, self.D)


class MambaBlock(nn.Module):
    """in_proj → [conv1d causal → SiLU → SSM seletivo] ⊙ SiLU(z) → out_proj"""

    def __init__(self, d_model=32, d_state=8, d_conv=4, expand=2, dt_rank=4, complex_A=False):
        super().__init__()
        inner = d_model * expand
        self.in_proj = nn.Linear(d_model, 2 * inner)
        self.conv = nn.Conv1d(inner, inner, d_conv, groups=inner, padding=d_conv - 1)
        self.ssm = SelectiveSSM(inner, d_state, dt_rank, complex_A=complex_A)
        self.out_proj = nn.Linear(inner, d_model)

    def forward(self, x):
        xb, z = self.in_proj(x).chunk(2, -1)
        xb = self.conv(xb.transpose(1, 2))[:, :, : x.size(1)].transpose(1, 2)   # corta o padding ⇒ causal
        y = self.ssm(F.silu(xb))
        return self.out_proj(y * F.silu(z))


class MambaModel(nn.Module):
    """
    use_norm: o Mamba original usa RMSNorm antes de cada bloco. Aqui a entrada é UNIVARIADA:
    o embedding w·x_t + b tem posto 1 e normalizar ao longo das features praticamente
    descarta a amplitude de x_t (RMSNorm(w·x) = sign(x)·w/rms(w)). Por isso o padrão é False.
    """

    def __init__(self, d_model=32, d_state=8, d_conv=4, expand=2, dt_rank=4, num_layers=1,
                 use_norm=False, complex_A=False):
        super().__init__()
        norm = (lambda: nn.LayerNorm(d_model)) if use_norm else (lambda: nn.Identity())
        self.input_proj = nn.Linear(1, d_model)
        self.norms = nn.ModuleList([norm() for _ in range(num_layers)])
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand, dt_rank, complex_A) for _ in range(num_layers)])
        self.norm = norm()
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))            # residual (pré-norma opcional)
        return self.output(self.norm(x)[:, -1, :])
