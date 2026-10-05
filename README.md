# Seminário — SSM, S4D, Mamba vs. LSTM e modelos clássicos

Projeto controlado por JSON que compara Regressão Linear com lags, Regressão de Painel,
ARIMA, LSTM, SSM (S4D) e Mamba na previsão one-step-ahead de séries com dependências longas.

## Instalação

```bash
pip install -r requirements.txt   # numpy, pandas, scikit-learn, statsmodels, torch, matplotlib
```

## Execução

```bash
# Experimentos principais (todos os modelos, contextos 24/168/720)
python run_model.py configs/all_models.json               # cenário A: lags curtos dominam
python run_model.py configs/all_models_long_memory.json   # cenário B: lags longos dominam

# Figuras e tabela-resumo (results/figuras/)
python make_figures.py

# Verificações numéricas das deduções (conv == recorrência, scan == laço, Teorema 1)
python verify_equivalences.py

# Treino completo das redes no cenário B (todos os dados) e ablação A real × complexo
python run_model.py configs/all_models_long_memory_neurais_full.json
python run_partes.py mamba:24 mamba:168 mamba:720 ssm_real:24 ssm_real:168 ssm_real:720 \
                     mamba_complexo:24 mamba_complexo:168 mamba_complexo:720   # retomável

# Exporta as séries usadas para CSV (dados/)
python export_dados.py

# Modelos individuais
python run_model.py configs/linear_regression.json
python run_model.py configs/panel_regression.json
python run_model.py configs/arima.json
python run_model.py configs/lstm.json
python run_model.py configs/ssm.json
python run_model.py configs/mamba.json
```

## Dados (DGP)

`y_t = Σ α_i · y_(t−lag_i) + ε_t`, com lags 1, 24, 168, 720 e ε ~ N(0, 0,1²).
Estacionariedade exige Σ|α_i| < 1 (estritamente; o gerador recusa ≥ 1 − 1e-9).

| Cenário | α₁ | α₂₄ | α₁₆₈ | α₇₂₀ |
|---|---|---|---|---|
| A | 0,38 | 0,285 | 0,19 | 0,095 |
| B | 0,20 | 0,20 | 0,25 | 0,30 |

Obs.: a versão anterior usava (0,4; 0,3; 0,2; 0,1), cuja soma é exatamente 1: raiz unitária
(processo não estacionário). Passava na checagem só por arredondamento de ponto flutuante.

## Protocolo

- Split cronológico 70/10/20; todos os modelos avaliados nos mesmos alvos de teste.
- StandardScaler ajustado só no treino.
- "oracle" = fórmula verdadeira do DGP (piso teórico, RMSE ≈ σ = 0,1).
- ARIMA: ajustado no treino, previsão one-step no teste com parâmetros congelados.
- Neurais: MSE, AdamW, early stopping na validação.
- `experiment.train_stride` (padrão 1): usa uma janela de treino a cada `stride`.
  Os configs `all_models*.json` usam 4 para caber em CPU; com GPU use 1.

## Modelos

- `models/ssm.py` — S4D (Gu et al., 2022): A diagonal complexa (S4D-Lin), discretização ZOH
  com Δ aprendido (log-uniforme em [1e-3, 1e-1]), kernel por Vandermonde e convolução via FFT.
  `forward_recurrent` implementa o modo recorrente equivalente.
- `models/mamba.py` — Mamba educacional em PyTorch puro: SSM seletivo (Δ, B, C dependentes da
  entrada), conv1d causal, porta SiLU e conexão residual. Scan paralelo por blocos
  (`selective_scan`) verificado contra o laço literal (`selective_scan_ref`).
  `complex_A=True` usa A complexo (S4D-Lin); `use_norm=False` por padrão, porque normalizar um
  embedding univariado (posto 1) apaga a amplitude da entrada.
  Não é o kernel CUDA oficial (`mamba_ssm`).
- `models/lstm.py`, `models/arima.py`, `models/linear_regression.py`, `models/panel_regression.py`.

## Resultados principais (RMSE no teste, cenário B; piso teórico 0,1005)

| Modelo | L=24 | L=168 | L=720 |
|---|---|---|---|
| Regressão linear | 0,1135 | 0,1086 | **0,1057** |
| ARIMA(24,0,2) | 0,1135 | – | – |
| LSTM | 0,1187 | 0,1188 | 0,1190 |
| SSM S4D (A complexo) | 0,1143 | **0,1104** | 0,1126 |
| SSM (A real) | 0,1164 | 0,1185 | 0,1185 |
| Mamba (A real) | 0,1133 | 0,1187 | 0,1191 |
| Mamba (A complexo) | 0,1140 | 0,1166 | não rodado |

Cenário A (regressão): 0,1048 / 0,1025 / 0,1051; ARIMA 0,1048. Painel (50 séries): 0,1032 (L=24), 0,1001 (L=168).
As redes do cenário A foram treinadas só na rodada preliminar (train_stride=4) e não entram nas figuras finais.

Figuras em `results/figuras/`; tabela consolidada em `results/figuras/tabela_resumo.csv`.

## Referências

- Gu, Goel, Ré. *Efficiently Modeling Long Sequences with Structured State Spaces* (S4), ICLR 2022.
- Gu, Gupta, Goel, Ré. *On the Parameterization and Initialization of Diagonal State Space Models* (S4D), NeurIPS 2022.
- Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023.
- Dao, Gu. *Transformers are SSMs* (Mamba-2 / SSD), ICML 2024.
- Hochreiter, Schmidhuber. *Long Short-Term Memory*, 1997.
- Box, Jenkins. *Time Series Analysis: Forecasting and Control*.
