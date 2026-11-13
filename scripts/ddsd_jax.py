"""Directed-speech detection head in JAX/Optax.

Trains a 2-layer MLP (written directly in JAX: jit-compiled loss/grad,
Adam via Optax, early stopping on a speaker-held-out validation split) on the
same Whisper acoustic features, filter and leave-corpus-out folds as
ddsd_eval.py, and reports it next to the scikit-learn logistic regression so
the two implementations can be compared on identical splits.

  python scripts/ddsd_jax.py --feat_dir results/ddsd
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speechlab import ddsd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--feat_dir", default="results/ddsd")
ap.add_argument("--per_corpus", type=int, default=1500)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--weight_decay", type=float, default=1e-4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--select", type=int, default=1, help="pick hidden size / weight decay on the validation split")
ap.add_argument("--corpus_norm", type=int, default=0)
args = ap.parse_args()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402


def init(key, d_in, h):
    k1, k2 = jax.random.split(key)
    if h == 0:  # linear model (logistic regression written in JAX)
        return {"w": jnp.zeros((d_in, 1)), "b": jnp.zeros(1)}
    return {"w1": jax.random.normal(k1, (d_in, h)) * jnp.sqrt(2.0 / d_in), "b1": jnp.zeros(h),
            "w2": jax.random.normal(k2, (h, 1)) * jnp.sqrt(1.0 / h), "b2": jnp.zeros(1)}


def logits(p, x):
    if "w" in p:
        return (x @ p["w"] + p["b"])[:, 0]
    return (jax.nn.gelu(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"])[:, 0]


def loss_fn(p, x, y):
    return optax.sigmoid_binary_cross_entropy(logits(p, x), y).mean()


def train_mlp(Xtr, ytr, Xva, yva, seed, hidden=None, weight_decay=None):
    hidden = args.hidden if hidden is None else hidden
    weight_decay = args.weight_decay if weight_decay is None else weight_decay
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    params = init(jax.random.PRNGKey(seed), Xtr.shape[1], hidden)
    opt = optax.adamw(args.lr, weight_decay=weight_decay)
    state = opt.init(params)

    @jax.jit
    def step(p, s, x, y):
        val, g = jax.value_and_grad(loss_fn)(p, x, y)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, val

    val_loss = jax.jit(loss_fn)
    best, best_p, bad = np.inf, params, 0
    xt, yt, xv, yv = map(jnp.asarray, (Xtr, ytr, Xva, yva))
    for ep in range(args.epochs):  # full-batch: a few thousand rows
        params, state, _ = step(params, state, xt, yt)
        v = float(val_loss(params, xv, yv))
        if v < best - 1e-4:
            best, best_p, bad = v, params, 0
        else:
            bad += 1
            if bad >= 20:
                break
    predict = jax.jit(lambda p, x: jax.nn.sigmoid(logits(p, x)))
    return lambda X: np.asarray(predict(best_p, jnp.asarray((X - mu) / sd))), ep + 1


d = Path(args.feat_dir)
z = np.load(d / "features.npz")
feats = {i: a for i, a in zip(z["ids"], z["acoustic"])}
rows = [json.loads(line) for line in open(d / "rows.jsonl")]
kept = [r for r in rows if ddsd.keep_utterance(r["duration"], r["hyp"])]
present = {r["corpus"] for r in kept}
res = {"backend": f"jax {jax.__version__} ({jax.default_backend()}), optax {optax.__version__}",
       "model": f"MLP {args.hidden} hidden, GELU, AdamW, early stopping", "folds": {}}
for f in ddsd.make_folds(present):
    tr_all = ddsd.balanced_sample([r for r in kept if r["corpus"] in f["train"]], args.per_corpus, args.seed)
    te = ddsd.balanced_sample([r for r in kept if r["corpus"] in f["test"]], args.per_corpus, args.seed)
    tr, te_in = ddsd.group_split(tr_all, 0.2, args.seed)       # same splits as ddsd_eval.py
    fit, va = ddsd.group_split(tr, 0.15, args.seed + 1)        # early-stopping split, speaker-held-out
    X = lambda rs: np.stack([feats[r["id"]] for r in rs]).astype(np.float32)  # noqa: E731
    y = lambda rs: np.array([r["label"] for r in rs], dtype=np.float32)  # noqa: E731
    Xf, Xv, Xin, Xte = X(fit), X(va), X(te_in), X(te)
    if args.corpus_norm:
        Xf, Xv, Xin, Xte = (ddsd.corpus_standardize(r, a) for r, a in ((fit, Xf), (va, Xv), (te_in, Xin), (te, Xte)))
    grid = [(h, wd) for h in (0, 64, 256) for wd in (1e-4, 1e-2)] if args.select else [(args.hidden, args.weight_decay)]
    best = None
    from sklearn.metrics import roc_auc_score
    for h, wd in grid:  # model selection on the speaker-held-out validation split only
        pr, ep = train_mlp(Xf, y(fit), Xv, y(va), args.seed, h, wd)
        auc = roc_auc_score(y(va), pr(Xv))
        if best is None or auc > best[0]:
            best = (auc, h, wd, pr, ep)
    _, best_h, best_wd, predict, epochs = best
    Xtr_lr = X(tr) if not args.corpus_norm else ddsd.corpus_standardize(tr, X(tr))
    lr = ddsd._dense_model().fit(Xtr_lr, y(tr).astype(int))
    res["folds"][f["name"]] = {
        "epochs_run": epochs, "selected_hidden": best_h, "selected_weight_decay": best_wd,
        "jax_mlp": {"in_domain": ddsd.score(y(te_in), predict(Xin)),
                    "cross_corpus": ddsd.score(y(te), predict(Xte))},
        "sklearn_logreg": {"in_domain": ddsd.score(y(te_in), lr.predict_proba(Xin)[:, 1]),
                           "cross_corpus": ddsd.score(y(te), lr.predict_proba(Xte)[:, 1])},
    }
    v = res["folds"][f["name"]]
    print(f"{f['name']}: JAX MLP cross-corpus AUC {v['jax_mlp']['cross_corpus']['auc']} "
          f"| sklearn LR {v['sklearn_logreg']['cross_corpus']['auc']}")
res["selection"] = "hidden in {0 (linear), 64, 256} x weight decay in {1e-4, 1e-2}, by validation AUC" if args.select else "fixed"
res["corpus_norm"] = bool(args.corpus_norm)
json.dump(res, open(d / ("ddsd_jax_cnorm.json" if args.corpus_norm else "ddsd_jax.json"), "w"), indent=2)
