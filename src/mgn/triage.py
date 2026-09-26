"""Calibrated System-1 triage: which failure (or none) is this frame? A gradient-boosted classifier on per-detector
frame scores, temperature-scaled, with selective prediction (abstain below a confidence threshold tau).

Splits (by simulation): fit on validation 50-69, temperature and tau on validation 70-99, test on --test-records.
Shift tests: held-out severity (fit and calibrate on severities 2-3, test on severity 1) and held-out type (fit and
calibrate without type X; report how often X frames are routed to abstain). No stress-derived features, by assert.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from mgn.evaluate import RESULTS
from mgn.inject import TYPES

LABELS = TYPES + ["none"]


def features(records):
    keys = sorted(k for k in records[0] if k.startswith("frame_") and not k.startswith("frame_clip"))
    assert not any("stress" in k for k in keys), "stress must never reach triage features (label leakage)"
    X = np.log1p(np.abs(np.array([[r[k] for k in keys] for r in records], float)))
    y = np.array([LABELS.index(r["type"] if r["label"] else "none") for r in records])
    return X, y, keys


def fit_temperature(logp, y):
    nll = lambda T: -np.mean(np.log(np.take_along_axis(_softmax(logp / T), y[:, None], 1).clip(1e-12)))
    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def selective(conf, correct, target=0.95):
    order = np.argsort(-conf)
    cum = np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    at80 = float(cum[int(0.8 * len(conf)) - 1])
    ok = np.flatnonzero(cum >= target)
    return at80, float((ok.max() + 1) / len(conf)) if len(ok) else 0.0


def fit(train, calib, classes):
    """GBM on `train`, temperature + tau (95% selective accuracy) on `calib`; restricted to `classes`."""
    Xt, yt, keys = features(train)
    Xc, yc, _ = features(calib)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xt, yt)
    cols = [list(clf.classes_).index(c) for c in classes]
    logp = np.log(clf.predict_proba(Xc)[:, cols].clip(1e-12))
    yc_idx = np.array([classes.index(v) for v in yc])
    T = fit_temperature(logp, yc_idx)
    p = _softmax(logp / T)
    conf, correct = p.max(1), p.argmax(1) == yc_idx
    order = np.argsort(-conf)
    cum = np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    ok = np.flatnonzero(cum >= 0.95)
    tau = float(conf[order][ok.max()]) if len(ok) else 1.0
    return dict(clf=clf, cols=cols, classes=classes, T=T, tau=tau, keys=keys)


def predict(m, records):
    X, y, _ = features(records)
    p = _softmax(np.log(m["clf"].predict_proba(X)[:, m["cols"]].clip(1e-12)) / m["T"])
    return p, y


def evaluate_model(m, test):
    p, y = predict(m, test)
    pred = np.array(m["classes"])[p.argmax(1)]
    conf, correct = p.max(1), pred == y
    at80, cov95 = selective(conf, correct)
    onehot = np.eye(len(m["classes"]))[[m["classes"].index(v) for v in y]]
    return dict(n=len(y), accuracy=float(correct.mean()),
                macro_f1=float(f1_score(y, pred, labels=m["classes"], average="macro", zero_division=0)),
                ece=ece(conf, correct), brier=float(((p - onehot) ** 2).sum(1).mean()),
                sel_acc_at_80cov=at80, coverage_at_95acc=cov95, temperature=float(m["T"]), tau=m["tau"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-records", type=Path, default=RESULTS / "triage_valid" / "records.pkl")
    ap.add_argument("--test-records", type=Path, required=True)
    ap.add_argument("--heldout-types", nargs="+", default=["frozen", "penetration"])
    ap.add_argument("--out", type=Path, default=RESULTS / "triage")
    a = ap.parse_args()
    V = pickle.load(open(a.valid_records, "rb"))["records"]
    test = pickle.load(open(a.test_records, "rb"))["records"]
    tr = [r for r in V if 50 <= r["traj"] < 70]
    ca = [r for r in V if 70 <= r["traj"] < 100]
    allc = list(range(len(LABELS)))
    res = {"in_distribution": evaluate_model(fit(tr, ca, allc), test)}
    sev23 = lambda R: [r for r in R if not r["label"] or r["sev"] in (2, 3)]
    res["heldout_severity1"] = evaluate_model(fit(sev23(tr), sev23(ca), allc),
                                              [r for r in test if not r["label"] or r["sev"] == 1])
    for x in a.heldout_types:
        keep = [c for c in allc if LABELS[c] != x]
        drop = lambda R: [r for r in R if not (r["label"] and r["type"] == x)]
        m = fit(drop(tr), drop(ca), keep)
        p, _ = predict(m, [r for r in test if r["label"] and r["type"] == x])
        res[f"heldout_type_{x}"] = dict(abstain_rate_on_unseen_type=float((p.max(1) < m["tau"]).mean()),
                                        in_distribution_on_rest=evaluate_model(m, drop(test)))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "triage.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
