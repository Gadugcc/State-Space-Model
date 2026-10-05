import numpy as np

def generate_series(n_samples, dependencies, noise_std=0.1, seed=42, warmup=1000):
    if not dependencies: raise ValueError('dependencies não pode ser vazio')
    # Condição suficiente de estacionariedade: Σ|α_i| < 1 (ver deduções, seção 1).
    # Tolerância explícita: 0.4+0.3+0.2+0.1 dá 0.9999999999999999 em float e passaria,
    # mas Σα = 1 é raiz unitária (φ(1) = 0) → processo NÃO estacionário.
    if sum(abs(float(d['coefficient'])) for d in dependencies) >= 1 - 1e-9:
        raise ValueError('Para este DGP, use soma dos coefficients absolutos < 1 (estritamente).')
    max_lag=max(int(d['lag']) for d in dependencies)
    total=n_samples+warmup+max_lag
    rng=np.random.default_rng(seed); y=np.zeros(total); noise=rng.normal(0,noise_std,total)
    for t in range(max_lag,total):
        y[t]=sum(float(d['coefficient'])*y[t-int(d['lag'])] for d in dependencies)+noise[t]
    start=max_lag+warmup
    return y[start:start+n_samples].astype('float32')

def generate_panel(n_entities,n_samples,dependencies,noise_std=0.1,entity_effect_std=1.0,seed=42,warmup=1000):
    rng=np.random.default_rng(seed); effects=rng.normal(0,entity_effect_std,n_entities)
    return np.stack([generate_series(n_samples,dependencies,noise_std,seed+i+1,warmup)+effects[i] for i in range(n_entities)]).astype('float32')
