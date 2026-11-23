#!/usr/bin/env python
"""Development diagnostics that use the VALIDATION split only (never the test split).

Trains CLARA variants and reports, on validation data: macro-F1, behaviour under item-level and
unit-level replacement of the LLM analysis, and how well the two gates do their jobs.

    python scripts/dev/val_check.py DATASET.json --seeds 1 2 --lr 2e-5 [--set KEY=VALUE ...] [--train KEY=VALUE ...]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from clara.analysis import instance_table  # noqa: E402
from clara.corrupt import replace_units  # noqa: E402
from clara.data import Dataset  # noqa: E402
from clara.experiment import Session  # noqa: E402
from clara.metrics import detection_metrics  # noqa: E402


def kv(pairs):
    out = {}
    for p in pairs:
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def auc(y, s):
    y = np.asarray(y)
    return float("nan") if y.min() == y.max() else roc_auc_score(y, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--method", default="clara")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    ds = Dataset.load(args.dataset)
    # the test split is replaced by a copy of the validation split: nothing here can see test data
    ds = ds.resplit(lambda it: None if it["split"] == "test" else it["split"])
    ds.by_split["test"] = ds.by_split["val"]
    session = Session(ds, log=lambda s: None)
    evals = [{"name": "noise50", "transform": "replace", "rate": 0.5, "seed": 0, "base": "val"},
             {"name": "noise100", "transform": "replace", "rate": 1.0, "seed": 0, "base": "val"},
             {"name": "unit50", "transform": "units", "rate": 0.5, "seed": 0, "base": "val"}]
    rows = []
    for seed in args.seeds:
        res = session.run(args.method, seed, model=kv(args.set), train=dict(kv(args.train), lr=args.lr),
                          evals=evals, keep=True)
        P, m = res["_preds"], res["metrics"]
        row = {"val": m["val"]["macf1"], "noise50": m["noise50"]["macf1"], "noise100": m["noise100"]["macf1"],
               "unit50": m["unit50"]["macf1"], "ep": res["summary"]["best_epoch"], "sec": res["seconds"]}
        if "inst_r" in P["val"]:
            pred = {"val/" + k: v for k, v in P["val"].items()}
            pred["val/id"] = np.array([it["id"] for it in ds.split("val")])
            pred["val/off"] = res["_data"]["val"]["off"].cpu().numpy()
            cfg = res["config"]["model"]
            T = instance_table(ds, ds.split("val"), pred, "val", cfg)
            ov = T["type"] == "overall"
            score = np.where(ov, 0.5 * (T["v"] + T["v0"]), T["v"])
            dec = np.abs(score - 0.5) > cfg["delta"]
            y = T["label"] == 1
            row["g_auc_overall"] = auc(((score > 0.5) == y)[ov & dec], T["g"][ov & dec]) if cfg["gate"] != "none" else float("nan")
            # what does the verdict gate look at?  mean g by (LLM says fake?, label) on the holistic instance
            m_ = ov & dec
            sf = score > 0.5
            cells = {"sayF/y1": m_ & sf & y, "sayF/y0": m_ & sf & ~y, "sayR/y1": m_ & ~sf & y, "sayR/y0": m_ & ~sf & ~y}
            row["g_cells"] = {k: (int(v.sum()), round(float(T["g"][v].mean()), 3)) for k, v in cells.items()}
            row["g_auc_within_sayF"] = auc(y[m_ & sf], T["g"][m_ & sf])
            row["g_auc_within_sayR"] = auc(~y[m_ & ~sf], T["g"][m_ & ~sf])
            row["conf_auc_overall"] = auc(((score > 0.5) == y)[m_], np.abs(score - 0.5)[m_])
            sx = P["val"]["s_x"]
            agree = np.where(sf[ov], sx, -sx)          # text branch agrees with the LLM verdict
            row["textagree_auc_overall"] = auc(((score > 0.5) == y)[ov][dec[ov]], agree[dec[ov]])
            cl_ = ~ov
            row["g_by_type"] = {t: round(float(T["g"][cl_ & (T["type"] == t)].mean()), 3) for t in ("fact", "quote", "number", "opinion")}
            mm = ~ov & dec & ~y
            row["g_auc_realclaims"] = auc((score < 0.5)[mm], T["g"][mm])
            row["r_genuine"] = float(T["r"].mean())
            # item-level replacement
            items50 = res["_data"]["noise50"]
            c = np.array([it.get("corrupted", False) for it in session._eval_items(evals[0])])
            row["r_auc_item"] = auc(~c, P["noise50"]["r_mean"])
            row["r_foreign_item"] = float(P["noise50"]["r_mean"][c].mean())
            # unit-level replacement
            unit_items = replace_units(ds.split("val"), 0.5, ds.lang, 0)
            pu = {"u/" + k: v for k, v in P["unit50"].items()}
            pu["u/id"] = np.array([it["id"] for it in unit_items])
            pu["u/off"] = res["_data"]["unit50"]["off"].cpu().numpy()
            U = instance_table(ds, unit_items, pu, "u", cfg)
            cl = U["type"] != "overall"
            row["r_auc_unit"] = auc(~U["corrupted"][cl], U["r"][cl])
        row["scalars"] = {k: round(v, 2) for k, v in res.get("scalars", {}).items()}
        rows.append(row)
        print("seed {}: ".format(seed) + "  ".join("{}={:.4f}".format(k, v) if isinstance(v, float) else "{}={}".format(k, v)
                                                    for k, v in row.items()), flush=True)
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    print("MEAN [{}] ".format(args.label or args.method) + "  ".join("{}={:.4f}".format(k, np.mean([r[k] for r in rows])) for k in keys), flush=True)


if __name__ == "__main__":
    main()
