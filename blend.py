"""Blend saved Bi-LSTM checkpoints (v1 + v2) and score on the fixed validation split.

Rebuilds exactly the same features, breath-level split (seed 42) and train-split
standardisation as lstm.py, so checkpoint inputs match training. Scores blend
weights w*v2 + (1-w)*v1 on inspiratory (u_out == 0) validation rows, then writes
submission_blend.csv with the best weight.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED, VAL_FRAC, SEQ = 42, 0.2, 80
DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent
dev = torch.device("cuda")
t0 = time.time()


def add_features(df):  # identical to lstm.py
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
    g = df.groupby("breath_id")
    df["u_in_cumsum"] = g["u_in"].cumsum()
    dt = g["time_step"].diff().fillna(0)
    df["u_in_area"] = (df["u_in"] * dt).groupby(df["breath_id"]).cumsum()
    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0)
    df["u_in_lead1"] = g["u_in"].shift(-1).fillna(0)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff_lead"] = df["u_in_lead1"] - df["u_in"]
    df["dt"] = dt
    for r in (5, 20, 50):
        df[f"R_{r}"] = (df["R"] == r).astype(np.float32)
    for c in (10, 20, 50):
        df[f"C_{c}"] = (df["C"] == c).astype(np.float32)
    return df


FEATURES = [
    "R", "C", "time_step", "u_in", "u_out", "u_in_cumsum", "u_in_area",
    "u_in_lag1", "u_in_lag2", "u_in_lead1", "u_in_diff1", "u_in_diff_lead", "dt",
    "R_5", "R_20", "R_50", "C_10", "C_20", "C_50",
]

train = add_features(pd.read_csv(DATA / "train.csv"))
test = add_features(pd.read_csv(DATA / "test.csv"))

breaths = train["breath_id"].unique()
rng = np.random.default_rng(SEED)
val_breaths = set(rng.choice(breaths, size=int(len(breaths) * VAL_FRAC), replace=False))
saved = set(pd.read_csv(OUT / "val_breaths.csv")["breath_id"])
assert val_breaths == saved, "validation split differs from the one used in training"
is_val_b = np.isin(train["breath_id"].to_numpy()[::SEQ], list(val_breaths))

X = train[FEATURES].to_numpy(np.float32).reshape(-1, SEQ, len(FEATURES))
y = train["pressure"].to_numpy(np.float32).reshape(-1, SEQ)
Xte = test[FEATURES].to_numpy(np.float32).reshape(-1, SEQ, len(FEATURES))
te_ids = test["id"].to_numpy().reshape(-1, SEQ)
u = FEATURES.index("u_out")
mu = X[~is_val_b].reshape(-1, len(FEATURES)).mean(0)
sd = X[~is_val_b].reshape(-1, len(FEATURES)).std(0) + 1e-6
mva = X[is_val_b][:, :, u] == 0
yva = y[is_val_b]
Xva_t = torch.tensor((X[is_val_b] - mu) / sd, device=dev)
Xte_t = torch.tensor((Xte - mu) / sd, device=dev)
R_va, C_va = X[is_val_b][:, 0, 0], X[is_val_b][:, 0, 1]
del X, train


class BiLSTM(nn.Module):
    """Matches lstm.py; head_dropout=True gives v2's layout (Dropout at head[0])."""

    def __init__(self, n_in, hidden=256, layers=4, head_dropout=False):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        head = [nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)]
        self.head = nn.Sequential(*([nn.Dropout(0.0)] if head_dropout else []), *head)

    def forward(self, x):
        h, _ = self.lstm(self.inp(x))
        return self.head(h).squeeze(-1)


@torch.no_grad()
def predict(model, Xt, bs=2048):
    model.eval()
    out = []
    for i in range(0, len(Xt), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(Xt[i:i + bs]).float())
    return torch.cat(out).cpu().numpy()


def mae(p):
    return float(np.abs(p - yva)[mva].mean())


preds = {}
for tag, hd in (("bilstm_v1", False), ("bilstm_v2", True)):
    m = BiLSTM(len(FEATURES), head_dropout=hd).to(dev)
    m.load_state_dict(torch.load(OUT / "checkpoints" / f"{tag}.pt"))
    preds[tag] = (predict(m, Xva_t), predict(m, Xte_t))
    print(f"{tag}: val MAE {mae(preds[tag][0]):.5f}")
    del m

v1v, v1t = preds["bilstm_v1"]
v2v, v2t = preds["bilstm_v2"]
print("\nweight on v2 -> val MAE")
scores = {}
for w in np.round(np.arange(0, 1.01, 0.1), 1):
    scores[w] = mae(w * v2v + (1 - w) * v1v)
    print(f"  w={w:.1f}  {scores[w]:.5f}")
best_w = min(scores, key=scores.get)
best = scores[best_w]
print(f"\nBEST BLEND w_v2={best_w:.1f}  val MAE {best:.5f}  "
      f"(v2 alone {scores[1.0]:.5f}, gain {scores[1.0] - best:+.5f})")
print(f"Plain 50/50 average: {scores[0.5]:.5f}")

bv = best_w * v2v + (1 - best_w) * v1v
err = (np.abs(bv - yva) * mva).sum(1) / mva.sum(1)
print(pd.DataFrame({"R": R_va, "C": C_va, "err": err}).groupby(["R", "C"])["err"].mean().round(4).to_string())

bt = best_w * v2t + (1 - best_w) * v1t
sub = pd.DataFrame({"id": te_ids.ravel(), "pressure": bt.ravel()}).sort_values("id")
sample = pd.read_csv(DATA / "sample_submission.csv")
assert len(sub) == len(sample) and (sub["id"].to_numpy() == sample["id"].to_numpy()).all()
assert np.isfinite(sub["pressure"]).all()
sub.to_csv(OUT / "submission_blend.csv", index=False)
print(f"wrote submission_blend.csv  runtime={time.time() - t0:.1f}s")

# ---- optional 3rd model trained on ALL breaths (lstm.py --full) ----
# Its val predictions are in-sample, so its weight can't be tuned on the split.
# Fixed rule: 50% v3 + 50% of the tuned v1/v2 blend.
v3_ckpt = OUT / "checkpoints" / "bilstm_v3_full.pt"
if v3_ckpt.exists():
    m = BiLSTM(len(FEATURES), head_dropout=True).to(dev)
    m.load_state_dict(torch.load(v3_ckpt))
    v3v, v3t = predict(m, Xva_t), predict(m, Xte_t)
    print(f"\nbilstm_v3_full: in-sample val MAE {mae(v3v):.5f} (NOT a validation score)")
    print(f"corr(v3, v1v2 blend) on test insp rows: "
          f"{np.corrcoef(v3t[Xte[:, :, u] == 0], bt[Xte[:, :, u] == 0])[0, 1]:.5f}")
    final = 0.5 * v3t + 0.5 * bt
    sub3 = pd.DataFrame({"id": te_ids.ravel(), "pressure": final.ravel()}).sort_values("id")
    assert (sub3["id"].to_numpy() == sample["id"].to_numpy()).all()
    assert np.isfinite(sub3["pressure"]).all()
    sub3.to_csv(OUT / "submission_blend3.csv", index=False)
    print(f"wrote submission_blend3.csv  (0.5*v3 + {0.5 * best_w:.2f}*v2 + {0.5 * (1 - best_w):.2f}*v1)")

log = OUT / "experiments.csv"
pd.DataFrame([{
    "experiment": f"blend_v1v2_w{best_w:.1f}", "val_mae": round(best, 5), "best_iter": "",
    "runtime_s": round(time.time() - t0, 1),
    "notes": f"{best_w:.1f}*bilstm_v2 + {1 - best_w:.1f}*bilstm_v1; weight picked on the val split "
             f"(slightly optimistic); 50/50 = {scores[0.5]:.5f}",
}]).to_csv(log, mode="a", header=False, index=False)
