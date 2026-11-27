#!/usr/bin/env python
"""Everything the thesis reports about the open-book experiments -> results/ob/analysis.json.

Reads results/ob/runs (metrics), outputs/ob/runs (predictions, for the paired tests and the
subset analyses), outputs/ob/<dataset>.json (which claims have flagged reports) and
results/ob/*.stats.json.  Sections that cannot be computed yet are skipped with a message.

Method ids used in the output:

    zs_closed / zs_ev      zero-shot LLM, claim only / with first-stage evidence
    slm_closed / slm_ev    RoBERTa-large (learning rate selected on validation)
    lora_closed / lora_ev  LoRA-tuned LLM, claim only / with first-stage evidence
    sieve / sievej         the proposed verifier without / with the explanation target
    ours                   whichever of the two results/ob/decision.json selected (validation only);
                           ``final`` names it
    noswap / nosample / nohead   ``ours`` with one component removed
    swap30                 ``ours`` trained with mismatched evidence for 30% (instead of 15%) of the claims
    noclaim                evidence-only control
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
from openbook.metrics import classification_metrics, paired_bootstrap  # noqa: E402

DATASETS = ["RAWFC", "LIAR-RAW"]
RUN_OF = {"sieve": "sieve", "sievej": "sieve-j"}
GRIDS = {"slm_closed": "roberta-large-closed@", "slm_ev": "roberta-large-bm25@"}
PERTURBED = ("swap", "nearswap", "halfswap", "droptop")
GATE_Q = (0.01, 0.05, 0.10, 0.20)       # 0.05 is the value fixed in advance; the others show the sensitivity
LLM2 = "llama2"


class Missing(Exception):
    pass


def load_decision():
    path = os.path.join(ROOT, "results", "ob", "decision.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve(runs, final_run, prefix=""):
    """method id -> run name, for the runs that exist.  ``final_run`` is the run name of the final
    method ("sieve" or "sieve-j") or None while undecided; ``prefix`` selects another backbone."""
    fixed = {"zs_closed": "zs-closed", "zs_ev": "zs-bm25-m16", "lora_closed": "lora-closed", "lora_ev": "lora-bm25-m16",
             "noclaim": "lora-noclaim-bm25-m16", "sieve": "sieve", "sievej": "sieve-j"}
    if final_run:
        fixed.update({"ours": final_run, "noswap": final_run + "-noswap", "nosample": final_run + "-nosample",
                      "nohead": final_run + "-nohead", "swap30": final_run + "-swap30"})
    names = {k: prefix + v for k, v in fixed.items() if prefix + v in runs}
    if not prefix:
        for k, grid in GRIDS.items():
            best = RP.select_by_val(runs, grid)
            if best:
                names[k] = best
    return names


# --------------------------------------------------------------------------------------- data audit
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


# ------------------------------------------------------------------------------------ method tables
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


def comparisons(tag, runs, names, C, other):
    def cmp(a, ca, b, cb):
        if a not in names or b not in names:
            return None
        return RP.compare(tag, runs[names[a]], names[a], ca, tag, runs[names[b]], names[b], cb, C) or None

    pairs = {}

    def add(key, value):
        if value:
            pairs[key] = value

    # ---- test split, each method with its own inference procedure
    for a, b in (("ours", "lora_ev"), ("ours", "lora_closed"), ("ours", "slm_ev"), ("sieve", "lora_ev"), ("sievej", "lora_ev"),
                 ("sievej", "sieve"), ("lora_ev", "lora_closed"), ("lora_ev", "slm_ev"), ("lora_ev", "zs_ev"),
                 ("slm_ev", "slm_closed"), ("lora_closed", "slm_closed"), ("lora_closed", "zs_closed"),
                 ("noclaim", "lora_ev"), ("noclaim", "lora_closed"),
                 ("noswap", "ours"), ("nosample", "ours"), ("nohead", "ours"), ("nohead", "lora_ev")):
        add("{}_vs_{}".format(a, b), cmp(a, "test", b, "test"))
    # ---- the same checkpoints with another inference procedure
    for m in ("ours", "sieve", "sievej"):
        for alt in ("bm25", "read", "avg", "oracle"):
            add("{}_vs_{}@{}".format(m, m, alt), cmp(m, "test", m, "test@" + alt))
    for alt in ("read", "oracle"):
        add("lora_ev@{}_vs_own".format(alt), cmp("lora_ev", "test@" + alt, "lora_ev", "test"))
    add("ours_vs_lora_ev@read", cmp("ours", "test", "lora_ev", "test@read"))
    # ---- changed evidence: between methods, against the closed-book verifier, and each method against itself
    for kind in PERTURBED + ("closed",):
        c = "test@" + kind
        for a, b in (("ours", "lora_ev"), ("ours", "slm_ev"), ("sievej", "sieve"), ("noswap", "ours"), ("nosample", "ours"),
                     ("nohead", "ours")):
            add("{}_vs_{}@{}".format(a, b, kind), cmp(a, c, b, c))
        for m in ("ours", "lora_ev", "nohead", "noswap"):
            add("{}@{}_vs_lora_closed".format(m, kind), cmp(m, c, "lora_closed", "test"))
        for m in ("ours", "sieve", "sievej", "lora_ev", "zs_ev", "slm_ev", "noswap", "nosample", "nohead"):
            add("{}@{}_vs_own".format(m, kind), cmp(m, c, m, "test"))
    # ---- "no evidence" on the claims the perturbed conditions use (a subset on LIAR-RAW): side a
    for m in ("ours", "sieve", "sievej", "lora_ev", "zs_ev", "slm_ev"):
        add("{}@closed_vs_{}@swap".format(m, m), cmp(m, "test@closed", m, "test@swap"))
    # ---- the other evidence protocol at test time
    for m in ("ours", "sieve", "sievej", "lora_ev", "zs_ev", "slm_ev", "noclaim"):
        add("{}@{}_vs_own".format(m, other), cmp(m, "test@" + other, m, "test"))
    return pairs


def ensembles(tag, runs, names, C):
    """Seed ensembles (mean class probabilities) and paired bootstraps between them.  There is one
    ensemble per method, so only the test over items applies."""
    ens = {k: RP.ensemble(tag, n, runs[n]) for k, n in names.items() if len(runs[n]) > 1}
    ens = {k: e for k, e in ens.items() if e is not None}
    out = {"methods": {k: dict(e["metrics"], n_seeds=e["n_seeds"]) for k, e in ens.items()}, "compare": {}}
    for a, b in (("ours", "lora_ev"), ("ours", "lora_closed"), ("ours", "slm_ev"), ("lora_ev", "lora_closed"),
                 ("lora_ev", "slm_ev"), ("sievej", "sieve")):
        if a in ens and b in ens and np.array_equal(ens[a]["y"], ens[b]["y"]):
            bs = paired_bootstrap(ens[a]["y"], [ens[a]["pred"]], [ens[b]["pred"]], C)
            out["compare"]["{}_vs_{}".format(a, b)] = {"diff_mean": bs["diff"], "bootstrap": bs}
    return out


def _with_usable(tag, name, seed):
    """Predictions that carry per-claim usability scores: those of the run, or those of its
    re-evaluation from the saved adapter if the run predates the storing of the scores."""
    for reeval in (False, True):
        p = RP.load_preds(tag, name, seed, reeval=reeval)
        if p is not None and "test/usable" in p and "val/usable" in p:
            return p, reeval
    return None, False


def gate(tag, runs, names, C):
    """Usability-gated fallback: if the claim-level usability score a(c) of the proposed verifier is
    below a threshold, answer with the closed-book verifier of the same seed instead.  The threshold
    is the q-quantile of a(c) on the validation claims with their own evidence - no perturbed data
    and no test data are used to set it."""
    if "ours" not in names or "lora_closed" not in names:
        raise Missing("needs the final method and the closed-book verifier")
    conds = ["test"] + ["test@" + k for k in PERTURBED]
    rows = {q: {c: {"gated": [], "plain": [], "closed": [], "flagged": []} for c in conds} for q in GATE_Q}
    paired = {q: {c: [] for c in conds} for q in GATE_Q}
    used, from_reeval = [], []
    edges = np.linspace(0.0, 1.0, 21)
    hist = {c: [] for c in conds}
    for s in sorted(set(runs[names["ours"]]) & set(runs[names["lora_closed"]])):
        p, reeval = _with_usable(tag, names["ours"], s)
        cl = RP.load_preds(tag, names["lora_closed"], s)
        if p is None or cl is None:
            continue
        used.append(s)
        if reeval:
            from_reeval.append(s)
        pos = {str(i): k for k, i in enumerate(cl["test/id"])}
        for c in conds:
            if c + "/usable" in p:
                h, _ = np.histogram(np.clip(p[c + "/usable"], 0.0, 1.0), bins=edges)
                hist[c].append(h / max(h.sum(), 1))
        for q in GATE_Q:
            tau = float(np.quantile(p["val/usable"], q))
            for c in conds:
                if c + "/usable" not in p:
                    continue
                y = p[c + "/label"]
                plain = p[c + "/prob"].argmax(1)
                closed = cl["test/prob"].argmax(1)[[pos[str(i)] for i in p[c + "/id"]]]
                flag = p[c + "/usable"] < tau
                gated = np.where(flag, closed, plain)
                r = rows[q][c]
                r["flagged"].append(float(flag.mean()))
                for key, pred in (("gated", gated), ("plain", plain), ("closed", closed)):
                    r[key].append(classification_metrics(y, np.eye(C)[pred])["macf1"])
                paired[q][c].append((y, gated, plain, closed))
    if not used:
        raise Missing("no stored usability scores")
    out = {"seeds": used, "seeds_from_reeval": from_reeval, "q": {},
           "hist": {"edges": edges.tolist(), "share": {c: np.mean(v, axis=0).tolist() for c, v in hist.items() if v}}}
    for q in GATE_Q:
        qrow = {}
        for c in conds:
            r = rows[q][c]
            if not r["gated"]:
                continue
            ys = paired[q][c]
            same = all(np.array_equal(ys[0][0], t[0]) for t in ys)
            qrow[c] = {k: RP.ms(v) for k, v in r.items()}
            for other, idx in (("plain", 2), ("closed", 3)):
                cmp = {"diff_mean": float(np.mean(r["gated"]) - np.mean(r[other])), "welch_p": RP.welch(r["gated"], r[other])}
                if same:
                    cmp["bootstrap"] = paired_bootstrap(ys[0][0], [t[1] for t in ys], [t[idx] for t in ys], C)
                qrow[c]["vs_" + other] = cmp
        out["q"]["{:g}".format(q)] = qrow
    return out


def ablation(tag, runs, names, C):
    """The final method and its variants on the seeds that all ablation runs have: per-condition
    macro-F1 and the difference of every variant to the full method (variant minus full)."""
    variants = [k for k in ("noswap", "nosample", "nohead") if k in names]
    if "ours" not in names or not variants:
        raise Missing("no ablation runs")
    seeds = sorted(set.intersection(*[set(runs[names[k]]) for k in variants + ["ours"]]))
    if not seeds:
        raise Missing("ablation runs share no seed with the final method")
    other = "sieve" if names["ours"] == RUN_OF["sievej"] else "sievej"       # the variant without / with the explanation target
    rows = ["ours"] + variants + [k for k in ("lora_ev", other, "swap30") if k in names and set(seeds) <= set(runs[names[k]])]
    sub = {k: {s_: runs[names[k]][s_] for s_ in seeds} for k in rows}
    conds = ["test"] + ["test@" + k for k in PERTURBED + ("closed", "bm25")]
    out = {"seeds": seeds, "explain_row": other if other in rows else None, "methods": {}, "vs_full": {}}
    for k in rows:
        out["methods"][k] = {c: RP.ms([r["metrics"][c]["macf1"] for r in sub[k].values()])
                             for c in conds if all(c in r["metrics"] for r in sub[k].values())}
        if k == "ours":
            continue
        out["vs_full"][k] = {}
        for c in conds:
            cmp = RP.compare(tag, sub[k], names[k], c, tag, sub["ours"], names["ours"], c, C)
            if cmp:
                out["vs_full"][k][c] = cmp
    return out


def explain(tag, runs, names, C):
    """The verifier with and without the explanation target on their common seeds (sieve-j minus
    sieve), and the quality of the generated justifications."""
    if "sieve" not in names or "sievej" not in names:
        raise Missing("needs runs with and without the explanation target")
    seeds = sorted(set(runs[names["sieve"]]) & set(runs[names["sievej"]]))
    if not seeds:
        raise Missing("no common seed")
    a = {s_: runs[names["sievej"]][s_] for s_ in seeds}
    b = {s_: runs[names["sieve"]][s_] for s_ in seeds}
    out = {"seeds": seeds, "methods": {}, "compare": {}}
    conds = ["val", "test"] + ["test@" + k for k in PERTURBED + ("closed",)]
    for key, sub in (("sievej", a), ("sieve", b)):
        out["methods"][key] = {c: RP.ms([r["metrics"][c]["macf1"] for r in sub.values()])
                               for c in conds if all(c in r["metrics"] for r in sub.values())}
    for c in conds[1:]:
        cmp = RP.compare(tag, a, names["sievej"], c, tag, b, names["sieve"], c, C)
        if cmp:
            out["compare"][c] = cmp
    va, vb = [r["metrics"]["val"]["macf1"] for r in a.values()], [r["metrics"]["val"]["macf1"] for r in b.values()]
    out["compare"]["val"] = {"diff_mean": float(np.mean(va) - np.mean(vb)), "welch_p": RP.welch(va, vb)}
    rouge = {}
    for k in ("rouge_conclusion_rouge1", "rouge_conclusion_rouge2", "rouge_conclusion_rougeL", "rouge_full_rouge1",
              "rouge_full_rougeL", "justify_words", "justify_n"):
        vals = [r["extra"][k] for r in a.values() if k in (r.get("extra") or {})]
        if vals:
            rouge[k] = RP.ms(vals)
    out["justify"] = rouge
    out["train_seconds"] = {"sievej": float(np.mean([r["summary"]["train_seconds"] for r in a.values()])),
                            "sieve": float(np.mean([r["summary"]["train_seconds"] for r in b.values()]))}
    return out


def reeval_check(tag, runs, names):
    """Re-evaluations from saved adapters against the original runs: largest absolute difference in
    macro-F1 over the conditions both have (adapters of the early runs were stored in half precision)."""
    out = {}
    for k, n in names.items():
        diffs = []
        for s, r in runs[n].items():
            path = os.path.join(ROOT, "results", "ob", "runs", tag, n, "s{}.reeval.json".format(s))
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    r2 = json.load(f)
                diffs += [abs(r2["metrics"][c]["macf1"] - m["macf1"]) for c, m in r["metrics"].items() if c in r2["metrics"]]
        if diffs:
            out[k] = {"max_abs_diff": float(np.max(diffs)), "mean_abs_diff": float(np.mean(diffs)), "n": len(diffs)}
    return out


def protocol_section(D, protocol, corpus, final_run):
    tag = "{}.{}".format(D, protocol)
    runs = RP.load_runs(tag)
    if not runs:
        raise Missing("no runs for " + tag)
    names = resolve(runs, final_run)
    C = len(LABELS[D])
    other = "all" if protocol == "strict" else "strict"
    out = {"names": names, "methods": {k: RP.summarise(runs[n]) for k, n in names.items()}, "grid": {}}
    for m in out["methods"].values():      # share of the claims predicted as the lowest level, per perturbed condition
        low = {kind: m["extra"][kind + "_pred_share"][0] for kind in PERTURBED if kind + "_pred_share" in m["extra"]}
        if low:
            m["low_share"] = low
    if final_run:
        out["final"] = {v: k for k, v in RUN_OF.items()}[final_run]
    for grid in GRIDS.values():               # the whole learning-rate grid of the SLM baselines
        for n, s in runs.items():
            if n.startswith(grid):
                out["grid"][n] = {"val": RP.ms([r["metrics"]["val"]["macf1"] for r in s.values()]),
                                  "test": RP.ms([r["metrics"]["test"]["macf1"] for r in s.values()])}
    out["compare"] = comparisons(tag, runs, names, C, other)
    out["ensemble"] = ensembles(tag, runs, names, C)
    for key, fn in (("gate", lambda: gate(tag, runs, names, C)), ("reeval", lambda: reeval_check(tag, runs, names) or None),
                    ("ablation", lambda: ablation(tag, runs, names, C)), ("explain", lambda: explain(tag, runs, names, C))):
        try:
            val = fn()
            if val:
                out[key] = val
        except Missing:
            pass

    # ---- where does the other protocol matter?  claims with / without flagged reports
    sets = leak_sets(corpus)
    leak = {"n": {k: len(v) for k, v in sets.items()}, "methods": {}}
    for m in ("ours", "sieve", "sievej", "lora_ev", "zs_ev", "slm_ev", "noclaim"):
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
    """Models trained with all reports against models trained under the strict protocol (common seeds)."""
    if "strict" not in sections or "all" not in sections:
        raise Missing("needs both protocols")
    C = len(LABELS[D])
    (_, runs_s, names_s), (_, runs_a, names_a) = sections["strict"], sections["all"]
    out = {}
    for m in ("ours", "lora_ev"):
        if m in names_s and m in names_a:
            common = sorted(set(runs_s[names_s[m]]) & set(runs_a[names_a[m]]))
            if not common:
                continue
            a = {s: runs_a[names_a[m]][s] for s in common}
            b = {s: runs_s[names_s[m]][s] for s in common}
            sets = {"any": leak_sets_cache[D]["any"], "none": leak_sets_cache[D]["none"]}
            row = {"seeds": common,
                   # (trained on all, tested on all) minus (trained strict, tested strict / tested all)
                   "all_all_vs_strict_strict": RP.compare(D + ".all", a, names_a[m], "test", D + ".strict", b, names_s[m], "test", C),
                   "all_all_vs_strict_all": RP.compare(D + ".all", a, names_a[m], "test", D + ".strict", b, names_s[m], "test@all", C),
                   "all_strict_vs_strict_strict": RP.compare(D + ".all", a, names_a[m], "test@strict", D + ".strict", b, names_s[m], "test", C)}
            for key, keep in sets.items():      # the same two contrasts on the claims with / without flagged reports
                row["all_all_vs_strict_strict@" + key] = RP.compare(D + ".all", a, names_a[m], "test", D + ".strict", b, names_s[m], "test", C, keep)
            out[m] = row
    if not out:
        raise Missing("no method trained under both protocols")
    return out


leak_sets_cache = {}


def transfer(D, runs, names):
    """Cross-dataset evaluation in three classes, and - for the six-class dataset - the in-domain
    result of the same models after merging their six classes into three."""
    out = {}
    tag = D + ".strict"
    for m in ("ours", "sieve", "sievej", "lora_ev"):
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


def llm2_section(D, final_run):
    """The second backbone: the same runs, names prefixed with the backbone tag."""
    tag = D + ".strict"
    runs = RP.load_runs(tag)
    names = resolve(runs, final_run, prefix=LLM2 + "-")
    if not names:
        raise Missing("no runs of the second backbone")
    C = len(LABELS[D])
    out = {"names": names, "methods": {k: RP.summarise(runs[n]) for k, n in names.items()},
           "compare": comparisons(tag, runs, names, C, "all")}
    if names:
        out["encoder"] = next(iter(runs[next(iter(names.values()))].values()))["encoder"]
    return out


def cases(D, corpus, names, n=2, seed=1, max_claim_words=30, top=5, max_words=45):
    """Worked examples, chosen by a rule fixed before any output was read: the first ``n`` test claims
    in id order with at most ``max_claim_words`` words, outputs of seed ``seed`` of the final method."""
    if "ours" not in names:
        raise Missing("no final method")
    base = os.path.join(ROOT, "outputs", "ob", "runs", D + ".strict", names["ours"], "s{}".format(seed))
    if not os.path.exists(base + ".selected.json"):
        raise Missing("no stored evidence scores")
    with open(base + ".selected.json", encoding="utf-8") as f:
        sel = json.load(f)
    if "test_scores" not in sel:
        raise Missing("stored selection has no scores")
    just = {}
    if os.path.exists(base + ".justify.json"):
        with open(base + ".justify.json", encoding="utf-8") as f:
            just = json.load(f)
    p = RP.load_preds(D + ".strict", names["ours"], seed)
    prob = {str(i): x for i, x in zip(p["test/id"], p["test/prob"])}
    out = []
    for it in sorted(corpus.split("test"), key=lambda x: x["id"]):
        if len(it["claim"].split()) > max_claim_words or it["id"] not in sel["test_scores"]:
            continue
        if just and it["id"] not in just:              # justifications exist for a subset only: stay inside it
            continue
        scored = sorted(sel["test_scores"][it["id"]], key=lambda t: -t[2])[:top]

        def clip(s):
            w = s.split()
            return s if len(w) <= max_words else " ".join(w[:max_words]) + " ..."

        out.append({"id": it["id"], "claim": it["claim"], "label": int(it["label"]), "pred": int(np.argmax(prob[it["id"]])),
                    "prob": [float(x) for x in prob[it["id"]]],
                    "evidence": [{"host": it["reports"][ri]["host"], "text": clip(it["reports"][ri]["sents"][si]), "score": float(x),
                                  "silver": int(it["reports"][ri]["ev"][si])} for ri, si, x in scored],
                    "justification": just.get(it["id"], {}).get("text", ""), "seed": seed})
        if len(out) == n:
            break
    if not out:
        raise Missing("no claim matches the rule")
    return out


def main():
    ap = argparse.ArgumentParser()
    default_out = os.path.join(ROOT, "results", "ob", "analysis.json")
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--assume-final", choices=sorted(RUN_OF.values()), default=None,
                    help="preview only: pretend this run is the final method (needs an --out other than the default)")
    args = ap.parse_args()
    if args.assume_final and os.path.abspath(args.out) == os.path.abspath(default_out):
        ap.error("--assume-final is for previews: give another --out")
    result, skipped = {}, []
    with open(os.path.join(ROOT, "results", "ob", "published.json"), encoding="utf-8") as f:
        published = json.load(f)
    decision = load_decision()
    final_run = args.assume_final or (decision["final"] if decision else None)
    for D in DATASETS:
        res = result.setdefault(D, {})
        stats_path = os.path.join(ROOT, "results", "ob", D + ".stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, encoding="utf-8") as f:
                st = json.load(f)
            res["stats"], res["bm25"] = st["splits"], st["bm25"]
        corpus = Corpus.load(os.path.join(ROOT, "outputs", "ob", D + ".json"))
        leak_sets_cache[D] = leak_sets(corpus)
        res["audit"] = audit(corpus)
        res["dump_rows"] = dump_rows(corpus)
        res["labels"] = {sp: np.bincount([it["label"] for it in corpus.split(sp)], minlength=corpus.n_classes).tolist()
                         for sp in ("train", "val", "test")}
        res["label_share"] = {sp: {str(i): c / max(sum(v), 1) for i, c in enumerate(v)} for sp, v in res["labels"].items()}
        sections = {}
        for protocol in ("strict", "all"):
            try:
                sections[protocol] = protocol_section(D, protocol, corpus, final_run)
                res[protocol] = sections[protocol][0]
            except Missing as e:
                skipped.append("{}/{}: {}".format(D, protocol, e))
        for name, fn in (("cross_protocol", lambda: cross_protocol(D, sections)),
                         ("transfer", lambda: transfer(D, *sections["strict"][1:])),
                         ("llm2", lambda: llm2_section(D, final_run)),
                         ("cases", lambda: cases(D, corpus, sections["strict"][2]))):
            try:
                res[name] = fn()
            except (Missing, KeyError) as e:
                skipped.append("{}/{}: {}".format(D, name, e))
    result["published"] = published
    if decision and not args.assume_final:
        result["decision"] = decision
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("wrote", args.out, "| final method:", final_run)
    for D in DATASETS:
        for p in ("strict", "all"):
            if p in result[D]:
                print(D, p, {k: (v["n"], round(100 * v["test"]["macf1"]["mean"], 2)) for k, v in result[D][p]["methods"].items()})
    for s in skipped:
        print("SKIPPED", s)


if __name__ == "__main__":
    main()
