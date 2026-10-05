"""
Executa experimentos definidos em JSON.

Uso:
    python run_model.py configs/linear_regression.json      # um modelo  ("model": {...})
    python run_model.py configs/all_models.json             # vários     ("models": [...])

Tarefa: previsão one-step-ahead de y_t a partir da janela [y_{t-L}, ..., y_{t-1}].
Split cronológico: treino | validação | teste. TODOS os modelos são avaliados
exatamente nos mesmos alvos de teste (t >= validation_end).

Saídas em output.directory:
    <run>_results.csv / .json   métricas por (modelo, contexto)
    <run>_config.json           cópia da configuração
    <run>_predictions.npz       previsões no teste (para os gráficos)
    <run>_history.json          curvas de perda (treino/validação) dos modelos neurais
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data.generator import generate_panel, generate_series
from evaluation.metrics import mae, mse, rmse
from models import (ARIMAModel, LagLinearRegression, LSTMModel, MambaModel,
                    PanelLagRegression, SSMModel)
from models.linear_regression import make_windows
from models.panel_regression import make_panel_windows
from utils.reproducibility import set_seed

TORCH_MODELS = {"lstm": LSTMModel, "ssm": SSMModel, "mamba": MambaModel}


# ============================================================
# UTILITÁRIOS
# ============================================================

def split_indices(n_samples, train_ratio, validation_ratio):
    """Pontos de corte temporais (ex.: 70% treino, 10% validação, 20% teste)."""
    train_end = int(n_samples * train_ratio)
    validation_end = train_end + int(n_samples * validation_ratio)
    return train_end, validation_end


def get_device(config):
    requested = config.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def fit_scaler(X_train):
    """Scaler ajustado SOMENTE no treino (evita vazamento do teste)."""
    return StandardScaler().fit(X_train.reshape(-1, 1))


def scale(scaler, a):
    return scaler.transform(a.reshape(-1, 1)).reshape(a.shape).astype(np.float32)


def unscale(scaler, a):
    return scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()


def windowed_split(series, context, horizon, train_end, validation_end):
    """Janelas + máscaras pelo índice temporal do ALVO."""
    X, y = make_windows(series, context, horizon)
    target_idx = np.arange(context, context + len(y)) + horizon - 1
    tr = target_idx < train_end
    va = (target_idx >= train_end) & (target_idx < validation_end)
    te = target_idx >= validation_end
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


# ============================================================
# ORÁCULO (DGP verdadeiro) — piso teórico do erro
# ============================================================

def run_oracle(series, dependencies, validation_end):
    """ŷ_t = Σ α_i y_{t-lag_i}. O erro que sobra é só o ruído ε_t (RMSE ≈ noise_std)."""
    t = np.arange(validation_end, len(series))
    pred = sum(float(d["coefficient"]) * series[t - int(d["lag"])] for d in dependencies)
    return {"prediction": pred, "true": series[t], "train_time": 0.0, "inference_time": 0.0,
            "parameters": len(dependencies), "validation_loss": np.nan}


# ============================================================
# REGRESSÃO LINEAR COM LAGS
# ============================================================

def run_linear_regression(series, context, horizon, train_end, validation_end, params):
    (X_tr, y_tr), _, (X_te, y_te) = windowed_split(series, context, horizon, train_end, validation_end)
    scaler = fit_scaler(X_tr)
    model = LagLinearRegression(**params).fit(scale(scaler, X_tr), scale(scaler, y_tr))
    pred = unscale(scaler, model.predict(scale(scaler, X_te)))
    return {"prediction": pred, "true": y_te, "train_time": model.train_time,
            "inference_time": model.inference_time, "parameters": model.parameters,
            "validation_loss": np.nan, "coefficients": model.coefficients}


# ============================================================
# REGRESSÃO DE PAINEL (efeitos fixos por entidade)
# ============================================================

def run_panel_regression(panel, context, horizon, train_end, validation_end, params):
    n_entities, n_time = panel.shape
    X, y, ids = make_panel_windows(panel, context, horizon)
    per_entity = n_time - context - horizon + 1
    target_idx = np.tile(np.arange(context, context + per_entity) + horizon - 1, n_entities)
    tr, te = target_idx < train_end, target_idx >= validation_end
    scaler = fit_scaler(X[tr])
    model = PanelLagRegression(n_entities, params.get("use_entity_effects", True))
    model.fit(scale(scaler, X[tr]), scale(scaler, y[tr]), ids[tr])
    pred = unscale(scaler, model.predict(scale(scaler, X[te]), ids[te]))
    return {"prediction": pred, "true": y[te], "train_time": model.train_time,
            "inference_time": model.inference_time, "parameters": model.parameters,
            "validation_loss": np.nan}


# ============================================================
# ARIMA
# ============================================================

def run_arima(series, train_end, validation_end, params):
    """Ajuste no treino; previsão one-step-ahead no teste (parâmetros congelados)."""
    model = ARIMAModel(**params)
    pred = model.fit_predict(series, train_end, validation_end)
    return {"prediction": pred, "true": series[validation_end:], "train_time": model.train_time,
            "inference_time": model.inference_time, "parameters": model.parameters,
            "validation_loss": np.nan}


# ============================================================
# MODELOS PYTORCH (LSTM / SSM / MAMBA)
# ============================================================

@torch.no_grad()
def predict_batched(model, X, device, batch_size=256):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + batch_size]).unsqueeze(-1).to(device)).squeeze(-1).cpu()
           for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


def train_torch_model(model, X_train, y_train, X_val, y_val, config, device):
    tc = config.get("training", {})
    epochs = int(tc.get("epochs", 10))
    batch_size = int(tc.get("batch_size", 64))
    patience = int(tc.get("patience", 3))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc.get("learning_rate", 1e-3)),
                                  weight_decay=float(tc.get("weight_decay", 0.0)))
    loss_fn = torch.nn.MSELoss()

    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train).unsqueeze(-1),
                                      torch.from_numpy(y_train).unsqueeze(-1)),
                        batch_size=batch_size, shuffle=True, generator=generator)

    best, best_state, wait = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * len(xb)
            n += len(xb)
        val_loss = float(np.mean((predict_batched(model, X_val, device) - y_val) ** 2))
        history["train_loss"].append(running / n)
        history["val_loss"].append(val_loss)
        print(f"    época {epoch + 1:3d}  treino={running / n:.5f}  val={val_loss:.5f}", flush=True)
        if val_loss < best:
            best, wait = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    train_time = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_time, best, history


def run_torch_model(model_name, model_params, series, context, horizon, train_end, validation_end,
                    config, device):
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = windowed_split(series, context, horizon,
                                                              train_end, validation_end)
    # train_stride > 1 usa uma janela a cada `stride` (janelas vizinhas são quase idênticas);
    # útil para rodar em CPU. Com GPU, use 1.
    stride = int(config["experiment"].get("train_stride", 1))
    X_tr, y_tr = X_tr[::stride], y_tr[::stride]

    scaler = fit_scaler(X_tr)
    X_tr, X_va, X_te = scale(scaler, X_tr), scale(scaler, X_va), scale(scaler, X_te)
    y_tr, y_va = scale(scaler, y_tr), scale(scaler, y_va)

    # instance_norm (normalização por janela, ideia do RevIN): tira a média de cada janela
    # da entrada e do alvo, e soma de volta na previsão. Assim a rede vê só as variações
    # dentro da janela e não precisa "adivinhar" o nível da série, que oscila devagar.
    inorm = bool(config.get("training", {}).get("instance_norm", False))
    if inorm:
        m_tr, m_va, m_te = (X.mean(axis=1, keepdims=True) for X in (X_tr, X_va, X_te))
        X_tr, X_va, X_te = X_tr - m_tr, X_va - m_va, X_te - m_te
        y_tr, y_va = y_tr - m_tr.ravel(), y_va - m_va.ravel()

    set_seed(int(config["seed"]) + sum(ord(c) for c in model_name) + context)
    model = TORCH_MODELS[model_name](**model_params)
    model, train_time, best_val, history = train_torch_model(model, X_tr, y_tr, X_va, y_va,
                                                             config, device)
    start = time.perf_counter()
    pred_s = predict_batched(model, X_te, device)
    if inorm:
        pred_s = pred_s + m_te.ravel()
    pred = unscale(scaler, pred_s)
    inference_time = time.perf_counter() - start
    return {"prediction": pred, "true": y_te, "train_time": train_time,
            "inference_time": inference_time,
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "validation_loss": best_val, "history": history, "model": model}


# ============================================================
# MAIN
# ============================================================

def load_data(config):
    dc = config["data"]
    # "csv": usa a série gerada no notebook 01 (fonte única da série para todos os modelos)
    if dc.get("type") == "csv":
        path = Path(dc["path"])
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        series = pd.read_csv(path)[dc.get("column", "y")].to_numpy(dtype=np.float32)
        return "univariate", series, series
    common = dict(n_samples=int(dc["n_samples"]), dependencies=dc["dependencies"],
                  noise_std=float(dc.get("noise_std", 0.1)), seed=int(config["seed"]),
                  warmup=int(dc.get("warmup", 1000)))
    if dc.get("type", "univariate") == "panel":
        panel = generate_panel(n_entities=int(dc["n_entities"]),
                               entity_effect_std=float(dc.get("entity_effect_std", 1.0)), **common)
        return "panel", panel, panel.mean(axis=0)
    series = generate_series(**common)
    return "univariate", series, series


def main():
    parser = argparse.ArgumentParser(description="Executa um experimento definido por um JSON.")
    parser.add_argument("config")
    config_path = Path(parser.parse_args().config)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    set_seed(int(config["seed"]))
    device = get_device(config)
    data_type, data, series = load_data(config)
    ec = config["experiment"]
    train_end, validation_end = split_indices(len(series), float(ec["train_ratio"]),
                                              float(ec["validation_ratio"]))
    horizon = int(ec.get("horizon", 1))
    model_list = config["models"] if "models" in config else [config["model"]]

    out_cfg = config.get("output", {})
    out_dir = Path(out_cfg.get("directory", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = out_cfg.get("run_name", config_path.stem)

    print("=" * 70)
    print(f"Configuração: {config_path} | dado: {data_type} | device: {device} | seed: {config['seed']}")
    print(f"Modelos: {[m['name'] for m in model_list]} | split: treino<{train_end} "
          f"val<{validation_end} teste<{len(series)}")
    print("=" * 70)

    rows, predictions, histories = [], {}, {}
    predictions["true"] = series[validation_end:]
    np.save(out_dir / f"{run_name}_series.npy", series)

    for mcfg in model_list:
        mtype, params = mcfg["name"], mcfg.get("params", {})
        name = mcfg.get("label", mtype)          # rótulo permite variantes (ex.: mamba_complexo)
        contexts = mcfg.get("context_lengths", ec.get("context_lengths", [24]))
        if mtype in {"arima", "oracle"}:
            contexts = [None]                     # não usam janela de contexto
        for context in contexts:
            print(f"\n>>> {name} | contexto = {context if context else 'N/A'}", flush=True)
            if mtype == "oracle":
                r = run_oracle(series, config["data"]["dependencies"], validation_end)
            elif mtype == "arima":
                r = run_arima(series, train_end, validation_end, params)
            elif mtype == "linear_regression":
                r = run_linear_regression(series, context, horizon, train_end, validation_end, params)
            elif mtype == "panel_regression":
                if data_type != "panel":
                    raise ValueError("panel_regression exige data.type='panel'.")
                r = run_panel_regression(data, context, horizon, train_end, validation_end, params)
            elif mtype in TORCH_MODELS:
                r = run_torch_model(mtype, params, series, context, horizon, train_end,
                                    validation_end, config, device)
            else:
                raise ValueError(f"Modelo desconhecido: {name}")

            key = f"{name}__{context}"
            row = {"model": name, "context_length": context,
                   "mae": mae(r["true"], r["prediction"]), "rmse": rmse(r["true"], r["prediction"]),
                   "mse": mse(r["true"], r["prediction"]), "train_time_s": r["train_time"],
                   "inference_time_s": r["inference_time"], "parameters": r["parameters"],
                   "best_val_loss": r["validation_loss"], "seed": config["seed"],
                   "dependencies": json.dumps(config["data"].get("dependencies"), separators=(",", ":"))}
            rows.append(row)
            if mtype != "panel_regression":
                predictions[key] = np.asarray(r["prediction"], np.float32)
            if "coefficients" in r:
                predictions[key + "__coef"] = np.asarray(r["coefficients"], np.float32)
            if "history" in r:
                histories[key] = r["history"]
            if mtype == "ssm":                     # kernel K_k aprendido (1ª camada) p/ visualização
                with torch.no_grad():
                    predictions[key + "__kernel"] = r["model"].layers[0].kernel(context).cpu().numpy()
            print(f"  MAE={row['mae']:.5f}  RMSE={row['rmse']:.5f}  treino={row['train_time_s']:.1f}s"
                  f"  params={row['parameters']}", flush=True)

            # salva a cada modelo (se algo travar, nada se perde)
            df = pd.DataFrame(rows)
            df.to_csv(out_dir / f"{run_name}_results.csv", index=False)
            (out_dir / f"{run_name}_results.json").write_text(
                json.dumps(rows, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
            np.savez(out_dir / f"{run_name}_predictions.npz", **predictions)
            (out_dir / f"{run_name}_history.json").write_text(json.dumps(histories), encoding="utf-8")

    (out_dir / f"{run_name}_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print("\n" + "=" * 70 + "\nRESULTADOS\n" + "=" * 70)
    print(pd.DataFrame(rows).drop(columns=["dependencies", "seed"]).to_string(index=False))


if __name__ == "__main__":
    main()
