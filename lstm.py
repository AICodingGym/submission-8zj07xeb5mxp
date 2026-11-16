"""Bi-LSTM sequence model for Ventilator Pressure Prediction (DSC 102 PA3).

Same validation protocol as baseline.py:
- 20% of whole breaths held out, chosen with np.random.default_rng(42) over the
  breath_ids in sorted order (identical to baseline.py's split).
- Metric: MAE on inspiratory rows only (u_out == 0).

Each breath (80 steps) is one sequence. Loss is L1 masked to u_out == 0 rows,
matching the competition metric. Whole dataset lives on the GPU (~300 MB).

Usage: python -u lstm.py [--epochs N] [--batch B] [--hidden H] [--layers L] [--tag NAME]
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--batch", type=int, default=512)
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--dropout", type=float, default=0.0, help="between LSTM layers and in the head")
p.add_argument("--patience", type=int, default=30, help="early-stop patience (epochs)")
p.add_argument("--tag", default="bilstm_v1")
p.add_argument("--seed", type=int, default=42, help="torch init/shuffle seed (split seed is fixed)")
p.add_argument("--resume", action="store_true",
               help="continue from checkpoints/<tag>_state.pt (saved after every epoch)")
p.add_argument("--full", action="store_true",
               help="train on ALL breaths (no holdout); keep final-epoch weights; val MAE becomes in-sample")
args = p.parse_args()

SEED = 42
VAL_FRAC = 0.2
SEQ = 80
DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent
CKPT = OUT / "checkpoints"
CKPT.mkdir(exist_ok=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True
# Safety cap: never let PyTorch take more than ~5.2 GB of the 6 GB card.
torch.cuda.set_per_process_memory_fraction(0.85, 0)

t0 = time.time()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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

# float32 columns halve the pandas footprint (values identical to ~7 significant digits;
# the model already trains on float32 arrays).
DTYPES = {"time_step": np.float32, "u_in": np.float32, "pressure": np.float32,
          "R": np.int8, "C": np.int8, "u_out": np.int8, "breath_id": np.int32, "id": np.int32}
train = add_features(pd.read_csv(DATA / "train.csv", dtype=DTYPES))
test = add_features(pd.read_csv(DATA / "test.csv", dtype=DTYPES))
assert len(train) % SEQ == 0 and len(test) % SEQ == 0
assert (train.groupby("breath_id").size() == SEQ).all()
assert (test.groupby("breath_id").size() == SEQ).all()

# ---- identical breath-level split to baseline.py ----
breaths = train["breath_id"].unique()
rng = np.random.default_rng(SEED)
val_breaths = set(rng.choice(breaths, size=int(len(breaths) * VAL_FRAC), replace=False))
pd.Series(sorted(val_breaths), name="breath_id").to_csv(OUT / "val_breaths.csv", index=False)

tr_breath_ids = train["breath_id"].to_numpy()[::SEQ]
is_val_b = np.isin(tr_breath_ids, list(val_breaths))

X = train[FEATURES].to_numpy(np.float32).reshape(-1, SEQ, len(FEATURES))
y = train["pressure"].to_numpy(np.float32).reshape(-1, SEQ)
Xte = test[FEATURES].to_numpy(np.float32).reshape(-1, SEQ, len(FEATURES))
te_ids = test["id"].to_numpy().reshape(-1, SEQ)
u_out_idx = FEATURES.index("u_out")

# Standardise with statistics from the training breaths only.
mu = X[~is_val_b].reshape(-1, len(FEATURES)).mean(0)
sd = X[~is_val_b].reshape(-1, len(FEATURES)).std(0) + 1e-6
mask_tr = X[:, :, u_out_idx] == 0          # inspiratory mask (raw u_out)
del train, test
gc.collect()
X = (X - mu) / sd
Xte = (Xte - mu) / sd

# Normalisation stats stay from the train split in --full mode too, so blend.py can reuse them.
tr_sel = np.ones(len(X), bool) if args.full else ~is_val_b
Xtr_t = torch.tensor(X[tr_sel], device=dev)
ytr_t = torch.tensor(y[tr_sel], device=dev)
mtr_t = torch.tensor(mask_tr[tr_sel], device=dev)
Xva_t = torch.tensor(X[is_val_b], device=dev)
yva_t = torch.tensor(y[is_val_b], device=dev)
mva_t = torch.tensor(mask_tr[is_val_b], device=dev)
Xte_t = torch.tensor(Xte, device=dev)
# Everything now lives on the GPU; free every host-side copy so the long training loop
# holds only a small amount of system RAM.
del X, Xte, y, mask_tr, tr_sel
gc.collect()
try:
    import psutil
    rss = f"  host RAM {psutil.Process().memory_info().rss / 2**30:.2f}G"
except ImportError:
    rss = ""
print(f"data ready {time.time() - t0:.1f}s  train breaths={len(Xtr_t)} val breaths={len(Xva_t)} "
      f"test breaths={len(Xte_t)}  feats={len(FEATURES)}{rss}")


class BiLSTM(nn.Module):
    def __init__(self, n_in, hidden, layers, dropout=0.0):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x):
        h, _ = self.lstm(self.inp(x))
        return self.head(h).squeeze(-1)


model = BiLSTM(len(FEATURES), args.hidden, args.layers, args.dropout).to(dev)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
steps_per_epoch = (len(Xtr_t) + args.batch - 1) // args.batch
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.05,
)
scaler = torch.amp.GradScaler("cuda")
print(f"model params={n_params:,}  steps/epoch={steps_per_epoch}")


def masked_l1(pred, target, mask):
    return (pred - target).abs()[mask].mean()


@torch.no_grad()
def predict(Xt, bs=2048):
    model.eval()
    out = []
    for i in range(0, len(Xt), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(Xt[i:i + bs]).float())
    return torch.cat(out)


best, best_ep, bad = float("inf"), -1, 0
ckpt_path = CKPT / f"{args.tag}.pt"            # weights used for prediction (best, or latest in --full)
state_path = CKPT / f"{args.tag}_state.pt"     # full training state for --resume
hist = []
start_ep = 1


def save_state(ep):
    """Write everything needed to continue training after epoch `ep` (atomic replace)."""
    tmp = state_path.with_suffix(".tmp")
    torch.save({
        "epoch": ep, "args": vars(args), "model": model.state_dict(), "opt": opt.state_dict(),
        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
        "best": best, "best_ep": best_ep, "bad": bad, "hist": hist,
        "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
    }, tmp)
    tmp.replace(state_path)


if args.resume:
    if not state_path.exists():
        raise SystemExit(f"--resume: no saved state at {state_path}")
    st = torch.load(state_path, weights_only=False)
    keep = ("lr", "dropout", "hidden", "layers", "batch", "epochs", "full", "seed")
    diff = {k: (st["args"][k], getattr(args, k)) for k in keep if st["args"].get(k) != getattr(args, k)}
    if diff:
        raise SystemExit(f"--resume: arguments differ from the saved run (saved, now): {diff}")
    model.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    scaler.load_state_dict(st["scaler"])
    best, best_ep, bad, hist = st["best"], st["best_ep"], st["bad"], st["hist"]
    torch.set_rng_state(st["rng_cpu"])
    torch.cuda.set_rng_state(st["rng_cuda"])
    start_ep = st["epoch"] + 1
    print(f"RESUMED from epoch {st['epoch']} (best {best:.4f} at epoch {best_ep})", flush=True)

for ep in range(start_ep, args.epochs + 1):
    te0 = time.time()
    model.train()
    perm = torch.randperm(len(Xtr_t), device=dev)
    tot = 0.0
    for i in range(0, len(perm), args.batch):
        idx = perm[i:i + args.batch]
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(Xtr_t[idx])
        loss = masked_l1(pred.float(), ytr_t[idx], mtr_t[idx])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        tot += loss.item() * len(idx)
    tr_mae = tot / len(Xtr_t)
    va_mae = masked_l1(predict(Xva_t), yva_t, mva_t).item()

    alloc = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    hist.append(dict(epoch=ep, train_mae=tr_mae, val_mae=va_mae, peak_alloc_gb=alloc,
                     reserved_gb=reserved, sec=time.time() - te0))
    flag = ""
    if args.full:  # no honest validation: just keep the latest weights
        torch.save(model.state_dict(), ckpt_path)
        best, best_ep = va_mae, ep
    elif va_mae < best:
        best, best_ep, bad = va_mae, ep, 0
        torch.save(model.state_dict(), ckpt_path)
        flag = " *"
    else:
        bad += 1
    if args.full:
        flag = " (in-sample)"
    save_state(ep)
    print(f"ep {ep:3d}  train {tr_mae:.4f}  val {va_mae:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
          f"vram peak {alloc:.2f}G reserved {reserved:.2f}G  {time.time() - te0:.1f}s{flag}", flush=True)
    if bad >= args.patience:
        print(f"early stop at epoch {ep}")
        break

pd.DataFrame(hist).to_csv(OUT / f"history_{args.tag}.csv", index=False)

# ---- evaluate best checkpoint ----
model.load_state_dict(torch.load(ckpt_path))
pva = predict(Xva_t)
val_mae = masked_l1(pva, yva_t, mva_t).item()
if args.full:
    print(f"\n--full: val breaths were in training; MAE below is IN-SAMPLE, not a validation score")
print(f"\nVALIDATION MAE (inspiratory, u_out==0): {val_mae:.5f}  (best epoch {best_ep})")
print(f"MLEvolve benchmark: 0.14237  gap: {val_mae - 0.14237:+.5f}")

# Per lung-setting breakdown on the validation set
va_R = (Xva_t[:, 0, 0] * sd[0] + mu[0]).round().cpu().numpy()
va_C = (Xva_t[:, 0, 1] * sd[1] + mu[1]).round().cpu().numpy()
err = ((pva - yva_t).abs() * mva_t).sum(1) / mva_t.sum(1)
bd = pd.DataFrame({"R": va_R, "C": va_C, "err": err.cpu().numpy()})
print(bd.groupby(["R", "C"])["err"].mean().round(4).to_string())

# ---- test predictions ----
pte = predict(Xte_t).cpu().numpy()
sub = pd.DataFrame({"id": te_ids.ravel(), "pressure": pte.ravel()}).sort_values("id")
sample = pd.read_csv(DATA / "sample_submission.csv")
assert len(sub) == len(sample) and (sub["id"].to_numpy() == sample["id"].to_numpy()).all()
assert np.isfinite(sub["pressure"]).all()
if not args.full:  # a --full model can't be validated, so never overwrite submission.csv
    sub.to_csv(OUT / "submission.csv", index=False)
sub.to_csv(OUT / f"submission_{args.tag}.csv", index=False)

runtime = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 2**30
print(f"wrote {'submission_' + args.tag + '.csv' if args.full else 'submission.csv'} ({len(sub)} rows)  runtime={runtime / 60:.1f} min  peak VRAM alloc={peak:.2f} GB")

log = OUT / "experiments.csv"
pd.DataFrame([{
    "experiment": args.tag, "val_mae": (f"in-sample {val_mae:.5f}" if args.full else round(val_mae, 5)),
    "best_iter": best_ep,
    "runtime_s": round(runtime, 1),
    "notes": f"BiLSTM h={args.hidden} L={args.layers} bs={args.batch} lr={args.lr} drop={args.dropout} "
             f"OneCycle {args.epochs}ep, masked L1, AMP, {len(FEATURES)} feats; peak VRAM {peak:.2f}GB; "
             + json.dumps({"params": n_params, "seed": args.seed, "full": args.full}),
}]).to_csv(log, mode="a", header=not log.exists(), index=False)
