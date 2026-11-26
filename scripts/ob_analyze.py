#!/usr/bin/env python
"""Everything the thesis reports about the open-book experiments -> results/ob/analysis.json.

Reads results/ob/runs (metrics), outputs/ob/runs (predictions, for the paired tests and the
subset analyses), outputs/ob/<dataset>.json (which claims have flagged reports) and
results/ob/*.stats.json.  Sections that cannot be computed yet are skipped with a message.
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from openbook import report as RP  # noqa: E402
from openbook.data import LABELS, LEAK_FLAGS, Corpus  # noqa: E402
from openbook.metrics import classification_metrics  # noqa: E402

DATASETS = ["RAWFC", "LIAR-RAW"]
#: canonical method id -> run name (learning-rate grids are resolved by validation)
FIXED = {"zs_closed": "zs-closed", "zs_ev": "zs-bm25-m16", "lora_closed": "lora-closed", "lora_ev": "lora-bm25-m16",
         "sieve": "sieve", "noswap": "sieve-noswap", "nohead": "sieve-nohead", "nosample": "sieve-nosample",
         "noclaim": "lora-noclaim-bm25-m16"}
GRIDS = {"slm_closed": "roberta-large-closed@", "slm_ev": "roberta-large-bm25@"}
PERTURBED = ("swap", "nearswap", "halfswap", "droptop")


class Missing(Exception):
    pass


def resolve(runs):
    names = {k: v for k, v in FIXED.items() if v in runs}
    for k, prefix in GRIDS.items():
        best = RP.select_by_val(runs, prefix)
        if best:
            names[k] = best
    return names


def leak_sets(corpus):
    """ids of the test claims with at least one flagged report, in total and per flag"""
    out = {"any": set()}
    for it in corpus.split("test"):
        flags = set(f for r in it["reports"] for f in r["flags"]) & set(LEAK_FLAGS)
        if flags:
            out["any"].add(it["id"])
        for f in flags:
            out.setdefault(f, set()).add(it["id"])
    out["none"] = {it["id"] for it in corpus.split("test")} - out["any"]
    return out


def audit(corpus):
    """Where the raw reports come from: flagged reports / claims per split and in total."""
    out = {}
    for sp in ("train", "val", "test", "total"):
        items = corpus.items if sp == "total" else corpus.split(sp)
        row = {"claims": len(items), "reports": 0, "reports_any": 0, "claims_any": 0,
               "reports_by_flag": {f: 0 for f in LEAK_FLAGS}, "claims_by_flag": {f: 0 for f in LEAK_FLAGS}}
        for it in items:
            seen = set()
            for r in it["reports"]:
                row["reports"] += 1
                fl = set(r["flags"]) & set(LEAK_FLAGS)
                row["reports_any"] += bool(fl)
                for f in fl:
                    row["reports_by_flag"][f] += 1
                seen |= fl
            row["claims_any"] += bool(seen)
            for f in seen:
                row["claims_by_flag"][f] += 1
        row["reports_any_share"] = row["reports_any"] / max(row["reports"], 1)
        row["claims_any_share"] = row["claims_any"] / max(row["claims"], 1)
        row["reports_share"] = {f: v / max(row["reports"], 1) for f, v in row["reports_by_flag"].items()}
        row["claims_share"] = {f: v / max(row["claims"], 1) for f, v in row["claims_by_flag"].items()}
        out[sp] = row
    return out


_WORD = re.compile(r"[a-z0-9]+")


def _norm(s):
    return " ".join(_WORD.findall(s.lower()))


def dump_rows(corpus):
    """Copies of the benchmark among the reports: for how many claims does a report from a code /
    data hosting site contain the claim's own row, and is a label printed right before it?"""
    names = sorted((" ".join(_WORD.findall(n)) for n in corpus.labels), key=len, reverse=True)
    out = {}
    for sp in ("train", "val", "test"):
        row = {"claims_with_dump": 0, "own_row": 0, "own_row_exact_label": 0, "own_row_other_label": 0}
        for it in corpus.split(sp):
            key = _norm(it["claim"])[:40]
            reps = [r for r in it["reports"] if "dump" in r["flags"]]
            if not reps:
                continue
            row["claims_with_dump"] += 1
            if len(key) < 20:
                continue
            hit = None
            for r in reps:
                for s in r["sents"]:
                    n = _norm(s)
                    k = n.find(key)
                    if k >= 0:
                        hit = n[:k].strip()
                        break
                if hit is not None:
                    break
            if hit is None:
                continue
            row["own_row"] += 1
            shown = next((x for x in names if hit.endswith(x)), None)
            gold = " ".join(_WORD.findall(corpus.labels[it["label"]]))
            if shown == gold:
                row["own_row_exact_label"] += 1
            elif shown is not None:
                row["own_row_other_label"] += 1
        out[sp] = row
    return out


def subset_metrics(tag, name, seeds, cond, keep, n_classes):
    """Seed-averaged accuracy / macro-F1 of one condition on a subset of the test claims."""
    acc, f1 = [], []
    for s in seeds:
        p = RP.load_preds(tag, name, s)
        if p is None or cond + "/prob" not in p:
            continue
        m = np.array([str(i) in keep for i in p[cond + "/id"]])
        if m.sum() == 0:
            continue
        r = classification_metrics(p[cond + "/label"][m], p[cond + "/prob"][m])
        acc.append(r["acc"])
        f1.append(r["macf1"])
    if not acc:
        return None
    return {"acc": float(np.mean(acc)), "macf1": float(np.mean(f1)), "n": int(len(keep))}


def protocol_section(D, protocol, corpus, other_tag_runs=None):
    tag = "{}.{}".format(D, protocol)
    runs = RP.load_runs(tag)
    if not runs:
        raise Missing("no runs for " + tag)
    names = resolve(runs)
    C = len(LABELS[D])
    other = "all" if protocol == "strict" else "strict"
    out = {"names": names, "methods": {k: RP.summarise(runs[n]) for k, n in names.items()}, "grid": {}}
    for k, prefix in GRIDS.items():           # the whole learning-rate grid of the SLM baselines
        for n, s in runs.items():
            if n.startswith(prefix):
                out["grid"][n] = {"val": RP.ms([r["metrics"]["val"]["macf1"] for r in s.values()]),
                                  "test": RP.ms([r["metrics"]["test"]["macf1"] for r in s.values()])}

    def cmp(a, ca, b, cb, keep=None):
        if a not in names or b not in names:
            return None
        r = RP.compare(tag, runs[names[a]], names[a], ca, tag, runs[names[b]], names[b], cb, C, keep)
        return r or None

    pairs = {}

    def add(key, value):
        if value:
            pairs[key] = value

    # ---- headline comparisons on the test split
    for a, b in (("sieve", "lora_ev"), ("sieve", "lora_closed"), ("lora_ev", "lora_closed"), ("lora_ev", "slm_ev"),
                 ("sieve", "slm_ev"), ("slm_ev", "slm_closed"), ("lora_ev", "zs_ev"), ("lora_closed", "slm_closed"),
                 ("noclaim", "lora_ev"), ("noclaim", "lora_closed"),
                 ("noswap", "sieve"), ("nohead", "sieve"), ("nosample", "sieve"), ("nohead", "lora_ev")):
        add("{}_vs_{}".format(a, b), cmp(a, "test", b, "test"))
    # the proposed method against the baseline that reads the same pool without selecting
    add("sieve_vs_lora_ev_read", cmp("sieve", "test", "lora_ev", "test@read"))
    add("lora_ev_read_vs_lora_ev", cmp("lora_ev", "test@read", "lora_ev", "test"))
    # inference ablations inside the proposed method (same checkpoints)
    for alt in ("bm25", "read", "avg", "oracle"):
        add("sieve_vs_sieve@" + alt, cmp("sieve", "test", "sieve", "test@" + alt))
    add("lora_ev_oracle_vs_lora_ev", cmp("lora_ev", "test@oracle", "lora_ev", "test"))
    # ---- perturbed evidence: between methods, and each method against its own unperturbed result
    for kind in PERTURBED + ("closed",):
        c = "test@" + kind
        for a, b in (("sieve", "lora_ev"), ("sieve", "lora_closed"), ("noswap", "sieve"), ("nohead", "sieve"),
                     ("nosample", "sieve"), ("sieve", "slm_ev") if kind == "swap" else ("sieve", "lora_ev")):
            cb = "test" if b == "lora_closed" else c
            add("{}_vs_{}@{}".format(a, b, kind), cmp(a, c, b, cb))
        for m in ("sieve", "lora_ev", "zs_ev", "noswap", "nohead", "nosample", "slm_ev"):
            add("{}@{}_vs_own".format(m, kind), cmp(m, c, m, "test"))
    # ---- evidence protocol at test time: the other pool minus the training pool
    for m in ("sieve", "lora_ev", "zs_ev", "slm_ev", "noclaim"):
        add("{}@{}_vs_own".format(m, other), cmp(m, "test@" + other, m, "test"))
    out["compare"] = pairs

    # ---- where does the other protocol matter?  claims with / without flagged reports
    sets = leak_sets(corpus)
    leak = {"n": {k: len(v) for k, v in sets.items()}, "methods": {}}
    for m in ("sieve", "lora_ev", "zs_ev", "slm_ev", "noclaim"):
        if m not in names:
            continue
        row = {}
        for key, keep in sets.items():
            if not keep:
                continue
            own = subset_metrics(tag, names[m], runs[names[m]], "test", keep, C)
            oth = subset_metrics(tag, names[m], runs[names[m]], "test@" + other, keep, C)
            if own and oth:
                row[key] = {"own": own, "other": oth,
                            "compare": RP.compare(tag, runs[names[m]], names[m], "test@" + other, tag, runs[names[m]], names[m], "test", C, keep)}
        if row:
            leak["methods"][m] = row
    out["leak"] = leak
    return out, runs, names


def cross_protocol(D, sections):
    """Models trained with all reports against models trained under the strict protocol."""
    if "strict" not in sections or "all" not in sections:
        raise Missing("needs both protocols")
    C = len(LABELS[D])
    (_, runs_s, names_s), (_, runs_a, names_a) = sections["strict"], sections["all"]
    out = {}
    for m in ("sieve", "lora_ev"):
        if m in names_s and m in names_a:
            common = sorted(set(runs_s[names_s[m]]) & set(runs_a[names_a[m]]))
            a = {s: runs_a[names_a[m]][s] for s in common}
            b = {s: runs_s[names_s[m]][s] for s in common}
            # (trained on all, tested on all) minus (trained strict, tested strict / tested all)
            out[m] = {"all_all_vs_strict_strict": RP.compare(D + ".all", a, names_a[m], "test", D + ".strict", b, names_s[m], "test", C),
                      "all_all_vs_strict_all": RP.compare(D + ".all", a, names_a[m], "test", D + ".strict", b, names_s[m], "test@all", C),
                      "all_strict_vs_strict_strict": RP.compare(D + ".all", a, names_a[m], "test@strict", D + ".strict", b, names_s[m], "test", C),
                      "seeds": common}
    if not out:
        raise Missing("no method trained under both protocols")
    return out


def transfer(D, runs, names):
    """Cross-dataset evaluation in three classes, and - for the six-class dataset - the in-domain
    result of the same models after merging their six classes into three."""
    out = {}
    tag = D + ".strict"
    for m in ("sieve", "lora_ev"):
        if m not in names:
            continue
        xs = [r["metrics"]["xfer"] for r in runs[names[m]].values() if "xfer" in r["metrics"]]
        if xs:
            out[m] = {k: RP.ms([x[k] for x in xs]) for k in ("macf1", "acc", "macp", "macr")}
            out[m]["seeds"] = sorted(s for s, r in runs[names[m]].items() if "xfer" in r["metrics"])
        if len(LABELS[D]) == 6:
            three = []
            for s in sorted(runs[names[m]]):
                p = RP.load_preds(tag, names[m], s)
                if p is not None:
                    prob = p["test/prob"].reshape(len(p["test/prob"]), 3, 2).sum(-1)
                    three.append(classification_metrics(p["test/label"] // 2, prob))
            if three:
                out.setdefault(m, {})["in_domain_three"] = {k: RP.ms([x[k] for x in three]) for k in ("macf1", "acc")}
    if not out:
        raise Missing("no transfer evaluations")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "ob", "analysis.json"))
    args = ap.parse_args()
    result, skipped = {}, []
    with open(os.path.join(ROOT, "results", "ob", "published.json"), encoding="utf-8") as f:
        published = json.load(f)
    for D in DATASETS:
        res = result.setdefault(D, {})
        stats_path = os.path.join(ROOT, "results", "ob", D + ".stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, encoding="utf-8") as f:
                st = json.load(f)
            res["stats"], res["bm25"] = st["splits"], st["bm25"]
        corpus = Corpus.load(os.path.join(ROOT, "outputs", "ob", D + ".json"))
        res["audit"] = audit(corpus)
        res["dump_rows"] = dump_rows(corpus)
        res["labels"] = {sp: np.bincount([it["label"] for it in corpus.split(sp)], minlength=corpus.n_classes).tolist()
                         for sp in ("train", "val", "test")}
        sections = {}
        for protocol in ("strict", "all"):
            try:
                sections[protocol] = protocol_section(D, protocol, corpus)
                res[protocol] = sections[protocol][0]
            except Missing as e:
                skipped.append("{}/{}: {}".format(D, protocol, e))
        for name, fn in (("cross_protocol", lambda: cross_protocol(D, sections)),
                         ("transfer", lambda: transfer(D, *sections["strict"][1:]))):
            try:
                res[name] = fn()
            except (Missing, KeyError) as e:
                skipped.append("{}/{}: {}".format(D, name, e))
    result["published"] = published
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("wrote", args.out)
    for D in DATASETS:
        for p in ("strict", "all"):
            if p in result[D]:
                print(D, p, {k: (v["n"], round(100 * v["test"]["macf1"]["mean"], 2)) for k, v in result[D][p]["methods"].items()})
    for s in skipped:
        print("SKIPPED", s)


if __name__ == "__main__":
    main()
