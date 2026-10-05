"""Treina os modelos neurais (LSTM, SSM, Mamba) um (modelo, contexto) por vez.

Por que um de cada vez?
  O treino completo leva horas na CPU. Rodando por partes, se o terminal fechar
  ou o computador dormir, basta rodar de novo: o que já terminou é pulado.

Uso (na pasta do projeto, com o venv ativado):
    python run_neurais.py                      # usa configs/exp_neurais.json
    python run_neurais.py configs/outro.json   # outra configuração

Saídas:
    results/partes_neurais/<modelo>__<L>_*     um resultado por (modelo, contexto)
    results/exp_neurais_results.csv            tudo junto (mesmo formato do run_model.py)
    results/exp_neurais_predictions.npz
    results/exp_neurais_history.json
    results/exp_neurais_config.json
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "exp_neurais.json"
base = json.loads(cfg_path.read_text(encoding="utf-8"))

run_name = base["output"]["run_name"]
out_dir = ROOT / base["output"]["directory"]
parts_dir = out_dir / f"partes_{run_name}"
(parts_dir / "cfg").mkdir(parents=True, exist_ok=True)

contexts = base["experiment"]["context_lengths"]
models = base["models"]

# ordem: contextos curtos primeiro (resultados chegam mais rápido)
fila = [(m, L) for L in contexts for m in models]
print(f"{len(fila)} treinos: {len(models)} modelos × {len(contexts)} contextos")

for i, (m, L) in enumerate(fila, 1):
    label = m.get("label", m["name"])
    part = f"{label}__{L}"
    if (parts_dir / f"{part}_results.csv").exists():
        print(f"[{i}/{len(fila)}] já existe: {part}")
        continue
    c = json.loads(json.dumps(base))
    c["models"] = [m]
    c["experiment"]["context_lengths"] = [L]
    c["output"] = {"directory": str(parts_dir), "run_name": part}
    cfg_file = parts_dir / "cfg" / f"{part}.json"
    cfg_file.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[{i}/{len(fila)}] >>> {part}  ({time.strftime('%H:%M:%S')})", flush=True)
    subprocess.run([sys.executable, str(ROOT / "run_model.py"), str(cfg_file)], check=True)

# ---- junta as partes em um único conjunto de arquivos ----
frames, preds, hist = [], {}, {}
for m in models:
    for L in contexts:
        part = f"{m.get('label', m['name'])}__{L}"
        frames.append(pd.read_csv(parts_dir / f"{part}_results.csv"))
        with np.load(parts_dir / f"{part}_predictions.npz") as d:
            for k in d.files:
                preds[k] = d[k]
        h = parts_dir / f"{part}_history.json"
        if h.exists():
            hist.update(json.loads(h.read_text(encoding="utf-8")))

res = pd.concat(frames, ignore_index=True)
res.to_csv(out_dir / f"{run_name}_results.csv", index=False)
res.to_json(out_dir / f"{run_name}_results.json", orient="records", indent=2, force_ascii=False)
np.savez(out_dir / f"{run_name}_predictions.npz", **preds)
(out_dir / f"{run_name}_history.json").write_text(json.dumps(hist), encoding="utf-8")
(out_dir / f"{run_name}_config.json").write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 70)
print("RESULTADOS (juntos em", out_dir / f"{run_name}_results.csv", ")")
print("=" * 70)
cols = ["model", "context_length", "mae", "rmse", "train_time_s", "parameters"]
print(res[cols].sort_values(["model", "context_length"]).to_string(index=False))
