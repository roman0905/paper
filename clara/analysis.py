"""Analyses behind the tables and figures of the paper.

Every function takes finished runs (see ``clara.report``) and returns plain dicts, so that
``scripts/analyze.py`` can dump all numbers used in the paper into one JSON file.
"""
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from clara import selective
from clara.data import Dataset, instances
from clara.metrics import localisation_metrics
from clara.report import load_preds, stat

Runs = Dict[int, Dict[str, Any]]          # seed -> result of one experiment


# --------------------------------------------------------------------------------------
# significance
# --------------------------------------------------------------------------------------

def welch(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided Welch t-test over seeds."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def _macf1_rows(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Macro-F1 of every row of ``pred`` against the matching row of ``y`` (both [R, n] in {0,1})."""
    y, pred = y.astype(bool), pred.astype(bool)
    out = np.zeros(y.shape[0])
    for cls in (True, False):
        yc, pc = (y == cls), (pred == cls)
        tp = (yc & pc).sum(1).astype(np.float64)
        fp = (~yc & pc).sum(1)
        fn = (yc & ~pc).sum(1)
        denom = 2 * tp + fp + fn
        out += np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return out / 2


def paired_bootstrap(y: np.ndarray, probs_a: List[np.ndarray], probs_b: List[np.ndarray],
                     n_boot: int = 5000, seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap over test items of the difference in seed-averaged macro-F1 (A - B).

    Each resample draws test items with replacement; both systems (all their seeds) are scored
    on the same resample.  ``p`` is the two-sided bootstrap p-value of "no difference".
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.randint(0, n, size=(n_boot, n))
    yb = y[idx]

    def mean_f1(probs):
        return np.mean([_macf1_rows(yb, (p >= 0.5)[idx]) for p in probs], axis=0)

    diff = mean_f1(probs_a) - mean_f1(probs_b)
    obs = np.mean([_macf1_rows(y[None], (p >= 0.5)[None])[0] for p in probs_a]) - \
        np.mean([_macf1_rows(y[None], (p >= 0.5)[None])[0] for p in probs_b])
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    return {"diff": float(obs), "ci_lo": float(np.percentile(diff, 2.5)),
            "ci_hi": float(np.percentile(diff, 97.5)), "p": float(min(p, 1.0))}


def compare(tag: str, runs_a: Runs, name_a: str, runs_b: Runs, name_b: str,
            split: str = "test") -> Dict[str, float]:
    """A vs B on ``split``: Welch test over seeds and paired bootstrap over items."""
    fa = [r["metrics"][split]["macf1"] for r in runs_a.values()]
    fb = [r["metrics"][split]["macf1"] for r in runs_b.values()]
    out = {"diff_mean": float(np.mean(fa) - np.mean(fb)), "welch_p": welch(fa, fb)}
    pa = [load_preds(tag, name_a, s) for s in sorted(runs_a)]
    pb = [load_preds(tag, name_b, s) for s in sorted(runs_b)]
    if all(p is not None and split + "/prob" in p for p in pa + pb):
        ids = pa[0][split + "/id"]
        assert all(np.array_equal(p[split + "/id"], ids) for p in pa + pb), "predictions are not item-aligned"
        y = pa[0][split + "/label"]
        out["bootstrap"] = paired_bootstrap(y, [p[split + "/prob"] for p in pa], [p[split + "/prob"] for p in pb])
    return out


# --------------------------------------------------------------------------------------
# reliability gate
# --------------------------------------------------------------------------------------

def _auc(y, s) -> float:
    y = np.asarray(y)
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def instance_table(ds: Dataset, split_items: List[Dict[str, Any]], pred: Dict[str, np.ndarray],
                   split: str, cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Flat per-instance arrays aligned with ``<split>/inst_*`` of a CLARA run."""
    by_id = {it["id"]: it for it in split_items}
    rows = {k: [] for k in ("item", "label", "v", "v0", "type", "sent", "valid", "corrupted")}
    for i, iid in enumerate(pred[split + "/id"]):
        it = by_id[int(iid)]
        inst = instances(it, ds.lang, cfg.get("max_claims", 6), cfg.get("v0_source", "zs"))
        if not cfg.get("use_claims", True):
            inst = inst[:1]
        elif not cfg.get("use_overall", True) and len(inst) > 1:
            inst = inst[1:]
        for x in inst:
            rows["item"].append(i)
            rows["label"].append(it["label"])
            rows["v"].append(x["v"])
            rows["v0"].append(x["v0"])
            rows["type"].append(x["type"])
            rows["sent"].append(x["sent"])
            rows["valid"].append(x["valid"])
            rows["corrupted"].append(bool(x.get("corrupted", False)))
    out = {k: np.array(v) for k, v in rows.items()}
    off = pred[split + "/off"]
    assert len(out["item"]) == off[-1] == len(pred[split + "/inst_p"]), "instance alignment broken"
    out["p"], out["g"] = pred[split + "/inst_p"], pred[split + "/inst_g"]
    out["r"] = pred[split + "/inst_r"] if split + "/inst_r" in pred else np.ones_like(out["p"])
    return out


def gate_analysis(ds: Dataset, tag: str, name: str, runs: Runs, delta: float = 0.05) -> Dict[str, Any]:
    """Does the gate know when the LLM took the right side?  (test split, averaged over seeds)

    Holistic instance: right/wrong is observed.  Claims of real news: right iff not suspected
    (MIL asymmetry).  The reference signal is the LLM's own confidence |score - 0.5|.
    """
    items = ds.split("test")
    acc: Dict[str, List[float]] = {}

    def add(k, v):
        acc.setdefault(k, []).append(float(v))

    for seed, r in sorted(runs.items()):
        pred = load_preds(tag, name, seed)
        if pred is None or "test/inst_g" not in pred:
            continue
        T = instance_table(ds, items, pred, "test", r["config"]["model"])
        ov = T["type"] == "overall"
        score = np.where(ov, 0.5 * (T["v"] + T["v0"]), T["v"])
        decided = (np.abs(score - 0.5) > delta) & (T["valid"] > 0)
        says_fake = score > 0.5
        y = T["label"] == 1

        m = ov & decided
        right = says_fake[m] == y[m]
        add("overall/decided_rate", decided[ov].mean())
        add("overall/llm_acc", right.mean())
        add("overall/auc_gate", _auc(right, T["g"][m]))
        add("overall/auc_llm_conf", _auc(right, np.abs(score[m] - 0.5)))
        add("overall/gate_when_right", T["g"][m][right].mean())
        add("overall/gate_when_wrong", T["g"][m][~right].mean())
        # does the gate follow the correctness of the LLM, or the label of the news?
        for sf, sname in ((True, "sayF"), (False, "sayR")):
            for yy, yname in ((True, "y1"), (False, "y0")):
                cell = m & (says_fake == sf) & (y == yy)
                if cell.any():
                    add("cells/{}_{}".format(sname, yname), T["g"][cell].mean())
                    add("cells/n_{}_{}".format(sname, yname), cell.sum())
        add("overall/auc_gate_for_label", _auc(y[m], T["g"][m]))
        if "test/s_x" in pred:
            sx = pred["test/s_x"][T["item"][m]]
            add("overall/auc_text_agreement", _auc(right, np.where(says_fake[m], sx, -sx)))

        m = ~ov & decided & ~y
        right = ~says_fake[m]
        add("real_claims/decided_rate", decided[~ov & ~y].mean())
        add("real_claims/llm_acc", right.mean())
        add("real_claims/auc_gate", _auc(right, T["g"][m]))
        add("real_claims/auc_llm_conf", _auc(right, np.abs(score[m] - 0.5)))
        add("real_claims/gate_when_right", T["g"][m][right].mean())
        add("real_claims/gate_when_wrong", T["g"][m][~right].mean())
        for t in ("fact", "number", "quote", "opinion"):
            mt = m & (T["type"] == t)
            if mt.sum() >= 20:
                add("type/{}/n".format(t), mt.sum())
                add("type/{}/llm_acc".format(t), (~says_fake[mt]).mean())
                add("type/{}/gate".format(t), T["g"][mt].mean())
        add("p/claims_real_news", T["p"][~ov & ~y].mean())
        add("p/claims_fake_news", T["p"][~ov & y].mean())
        add("p/overall_real_news", T["p"][ov & ~y].mean())
        add("p/overall_fake_news", T["p"][ov & y].mean())
        # which instances carry the decision: per item, is any claim / the holistic instance flagged?
        n_items = int(T["item"].max()) + 1
        claim_flag = np.zeros(n_items, dtype=bool)
        np.logical_or.at(claim_flag, T["item"][~ov], T["p"][~ov] > 0.5)
        over_flag = np.zeros(n_items, dtype=bool)
        np.logical_or.at(over_flag, T["item"][ov], T["p"][ov] > 0.5)
        y_item = pred["test/label"] == 1
        add("flag/fake_any_claim", claim_flag[y_item].mean())
        add("flag/fake_overall_only", (over_flag & ~claim_flag)[y_item].mean())
        add("flag/fake_none", (~over_flag & ~claim_flag)[y_item].mean())
        add("flag/real_any_claim", claim_flag[~y_item].mean())
        add("flag/real_overall_only", (over_flag & ~claim_flag)[~y_item].mean())
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
            for k, v in acc.items()}


def relevance_under_noise(ds: Dataset, tag: str, name: str, runs: Runs, lang: str) -> Dict[str, Any]:
    """How well do the gates separate genuine analysis units from foreign ones?

    Item-level replacement (noiseXX sets): mean relevance gate of an item, AUC against "the item
    kept its own analysis".  Unit-level replacement (unitXX sets): relevance gate of a claim, AUC
    against "the claim is genuine".  For reference the same AUCs are computed with the verdict
    gate g.  Methods without a claim structure are evaluated at the item level with their own
    gate values (``kind="pair"`` in :func:`gate_under_noise`).
    """
    from clara.corrupt import replace_units

    acc: Dict[str, List[float]] = {}

    def add(k, v):
        if v == v:
            acc.setdefault(k, []).append(float(v))

    base = ds.split("test")
    for seed, r in sorted(runs.items()):
        pred = load_preds(tag, name, seed)
        if pred is None or "test/inst_r" not in pred:
            continue
        for rate in (20, 40, 60, 80):
            sp = "noise{:02d}".format(rate)
            if sp + "/corrupted" in pred:
                c = pred[sp + "/corrupted"]
                add("item/auc_r", _auc(~c, pred[sp + "/r_mean"]))
                add("item/auc_g", _auc(~c, pred[sp + "/g_mean"]))
                add("item/r_genuine", pred[sp + "/r_mean"][~c].mean())
                add("item/r_foreign", pred[sp + "/r_mean"][c].mean())
        for rate in (25, 50):
            sp = "unit{:02d}".format(rate)
            if sp + "/inst_r" not in pred:
                continue
            items = replace_units(base, rate / 100.0, lang, 0)      # same transform, same seed
            T = instance_table(ds, items, pred, sp, r["config"]["model"])
            m = T["type"] != "overall"
            add("unit/auc_r", _auc(~T["corrupted"][m], T["r"][m]))
            add("unit/auc_g", _auc(~T["corrupted"][m], T["g"][m]))
            add("unit/r_genuine", T["r"][m & ~T["corrupted"]].mean())
            add("unit/r_foreign", T["r"][m & T["corrupted"]].mean())
            add("unit/r_overall", T["r"][~m].mean())
    return {k: float(np.mean(v)) for k, v in acc.items()}


def cross_relevance(ds: Dataset, ds_tamper: Dataset, tag: str, name: str, runs: Runs) -> Dict[str, float]:
    """Relevance gate on on-topic but locally unfaithful analyses.

    "cross" = tampered text with the analysis of the original text.  Claims whose source sentence
    was edited no longer agree with the text; all other claims still do.  "tamper" = the same
    tampered text with its own analysis (reference)."""
    from clara.corrupt import LLM_FIELDS

    by_id = {it["id"]: it for it in ds.split("test")}
    crossed = []
    for it in ds_tamper.split("test"):
        o, it = by_id[it["orig_id"]], dict(it)
        for f in LLM_FIELDS:
            if f in o:
                it[f] = o[f]
        crossed.append(it)
    acc: Dict[str, List[float]] = {}
    for seed, r in sorted(runs.items()):
        pred = load_preds(tag, name, seed)
        if pred is None or "cross/inst_r" not in pred:
            continue
        cfg = r["config"]["model"]
        for split, items in (("cross", crossed), ("tamper", ds_tamper.split("test"))):
            T = instance_table(ds_tamper, items, pred, split, cfg)
            gold = np.array([items[i]["tamper_sent"] for i in T["item"]])
            claim = T["type"] != "overall"
            hit = claim & (T["sent"] == gold)
            acc.setdefault(split + "/r_edited_sentence", []).append(float(T["r"][hit].mean()))
            acc.setdefault(split + "/r_other_claims", []).append(float(T["r"][claim & ~hit].mean()))
            acc.setdefault(split + "/r_overall", []).append(float(T["r"][~claim].mean()))
            acc.setdefault(split + "/p_edited_sentence", []).append(float(T["p"][hit].mean()))
            acc.setdefault(split + "/p_other_claims", []).append(float(T["p"][claim & ~hit].mean()))
            acc.setdefault(split + "/n_edited", []).append(float(hit.sum()))
        Tc = instance_table(ds_tamper, crossed, pred, "cross", cfg)
        goldc = np.array([crossed[i]["tamper_sent"] for i in Tc["item"]])
        cl = Tc["type"] != "overall"
        # AUC of r for "this claim still agrees with the text" among the claims of the crossed set
        acc.setdefault("cross/auc_r_unfaithful", []).append(_auc(~(Tc["sent"] == goldc)[cl], Tc["r"][cl]))
        acc.setdefault("prob/cross", []).append(float(pred["cross/prob"].mean()))
        acc.setdefault("prob/tamper", []).append(float(pred["tamper/prob"].mean()))
    if not acc:
        return {}
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def gate_under_noise(tag: str, name: str, runs: Runs, kind: str,
                     rates: Sequence[int] = (20, 40, 60, 80)) -> Dict[str, Any]:
    """Mean gate value on items whose LLM outputs were swapped vs untouched items, and how well
    the gate separates the two (AUC).  ``kind``: "clara" (mean instance gate) or "pair" (ARG /
    FactGuard: mean of the two rationale gates)."""
    out: Dict[str, List[float]] = {"clean": [], "corrupted": [], "auc": []}
    for seed in sorted(runs):
        pred = load_preds(tag, name, seed)
        if pred is None:
            continue
        for r in rates:
            sp = "noise{:02d}".format(r)
            if sp + "/corrupted" not in pred:
                continue
            c = pred[sp + "/corrupted"]
            if kind == "clara":
                if sp + "/g_mean" not in pred:
                    continue
                g = pred[sp + "/g_mean"]
            else:
                if sp + "/gate" not in pred:
                    continue
                g = pred[sp + "/gate"].mean(1)
            out["clean"].append(float(g[~c].mean()))
            out["corrupted"].append(float(g[c].mean()))
            out["auc"].append(_auc(~c, g))
    return {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()}


def arg_gate_auc(ds: Dataset, tag: str, name: str, runs: Runs, source: str) -> Dict[str, float]:
    """ARG: how well do (a) the fusion weights and (b) the usefulness predictor separate rationales
    whose LLM judgment was right from those whose judgment was wrong?  For (b), also the mean
    predicted usefulness by (LLM says fake?, label of the news)."""
    by_id = {it["id"]: it for it in ds.split("test")}
    acc: Dict[str, List[float]] = {}

    def add(k, v):
        if v == v:
            acc.setdefault(k, []).append(float(v))

    for seed in sorted(runs):
        pred = load_preds(tag, name, seed)
        if pred is None or "test/gate" not in pred:
            continue
        its = [by_id[int(i)] for i in pred["test/id"]]
        y = np.array([it["label"] == 1 for it in its])
        for j, k in enumerate(("td", "cs")):
            base = [(it["gpt"] if source == "gpt" else it)[k] for it in its]
            judged = np.array([b["pred"] in (0, 1) for b in base])
            says_fake = np.array([b["pred"] == 1 for b in base])
            right = (says_fake == y) & judged
            add(k + "/llm_acc", right[judged].mean())
            add(k + "/auc_weight", _auc(right[judged], pred["test/gate"][judged, j]))
            # reference: does the detector's own prediction agree with the LLM judgment?
            prob = pred["test/prob"]
            add(k + "/auc_own_agreement", _auc(right[judged], np.where(says_fake, prob, 1 - prob)[judged]))
            if "test/useful" in pred:
                u = pred["test/useful"][:, j]
                add(k + "/auc_useful", _auc(right[judged], u[judged]))
                for sf, sname in ((True, "sayF"), (False, "sayR")):
                    for yy, yname in ((True, "y1"), (False, "y0")):
                        cell = judged & (says_fake == sf) & (y == yy)
                        if cell.any():
                            add("{}/cells/{}_{}".format(k, sname, yname), u[cell].mean())
    return {k: float(np.mean(v)) for k, v in acc.items()}


# --------------------------------------------------------------------------------------
# localisation of a known falsehood (tamper set)
# --------------------------------------------------------------------------------------

def _sentence_scores(n_sents: int, sent_idx: np.ndarray, score: np.ndarray) -> np.ndarray:
    s = np.full(n_sents, -np.inf)
    for j, v in zip(sent_idx, score):
        if 0 <= j < n_sents:
            s[j] = max(s[j], v)
    return s


def localisation(ds_tamper: Dataset, tag: str, runs_by_name: Dict[str, Tuple[str, Runs]],
                 max_claims: int = 6, only: Optional[set] = None) -> Dict[str, Any]:
    """Hit@1 / MRR of the tampered sentence.

    ``runs_by_name``: {row label: (experiment name, runs)}; the method of each row is inferred
    from what its prediction file contains.  ``only`` restricts the evaluation to a subset of
    item ids (e.g. one kind of edit).
    """
    items = ds_tamper.split("test")
    by_id = {it["id"]: it for it in items}
    gold = {it["id"]: int(it["tamper_sent"]) for it in items}
    use = (lambda i: True) if only is None else (lambda i: int(i) in only)
    out: Dict[str, Any] = {}

    # LLM alone: raw suspicion scores of the claims, mapped to sentences
    scores, golds, covered = [], [], []
    for it in items:
        if not use(it["id"]):
            continue
        inst = [x for x in instances(it, ds_tamper.lang, max_claims) if x["type"] != "overall"]
        sidx = np.array([x["sent"] for x in inst], dtype=int)
        scores.append(_sentence_scores(len(it["sents"]), sidx, np.array([x["v"] for x in inst])))
        golds.append(gold[it["id"]])
        covered.append(gold[it["id"]] in set(sidx.tolist()))
    out["LLM suspicion scores"] = dict(localisation_metrics(scores, golds), kind="llm")
    out["_claim_coverage"] = float(np.mean(covered))
    out["_n_items"] = len(golds)
    out["_mean_sents"] = float(np.mean([len(it["sents"]) for it in items if use(it["id"])]))

    for label, (name, runs) in runs_by_name.items():
        per_seed = []
        for seed, r in sorted(runs.items()):
            pred = load_preds(tag, name, seed)
            if pred is None or "tamper/prob" not in pred:
                continue
            ids = pred["tamper/id"]
            sc, gd = [], []
            if "tamper/inst_g" in pred:                       # CLARA: claims -> sentences
                T = instance_table(ds_tamper, items, pred, "tamper", r["config"]["model"])
                for i, iid in enumerate(ids):
                    if not use(iid):
                        continue
                    m = (T["item"] == i) & (T["type"] != "overall")
                    sc.append(_sentence_scores(len(by_id[int(iid)]["sents"]), T["sent"][m].astype(int), T["p"][m]))
                    gd.append(gold[int(iid)])
                kind = "claims"
            elif "tamper/inst_p" in pred:                     # CLARA-D: one unit per sentence
                off = pred["tamper/off"]
                for i, iid in enumerate(ids):
                    if not use(iid):
                        continue
                    sc.append(pred["tamper/inst_p"][off[i] + 1:off[i + 1]])
                    gd.append(gold[int(iid)])
                kind = "sentences"
            elif "occl/prob" in pred:                         # black box: occlusion
                occ = dict(zip(pred["occl/id"].tolist(), pred["occl/prob"].tolist()))
                for i, iid in enumerate(ids):
                    if not use(iid):
                        continue
                    n = len(by_id[int(iid)]["sents"])
                    sc.append(np.array([pred["tamper/prob"][i] - occ[int(iid) * 100 + j] for j in range(n)]))
                    gd.append(gold[int(iid)])
                kind = "occlusion"
            else:
                continue
            per_seed.append(localisation_metrics(sc, gd))
        if per_seed:
            out[label] = {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]}
            out[label]["hit@1_std"] = float(np.std([p["hit@1"] for p in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0
            out[label]["kind"] = kind
            out[label]["n_seeds"] = len(per_seed)
    return out


def tamper_detection(ds_tamper: Dataset, tag: str, name: str, runs: Runs) -> Dict[str, float]:
    """How often a tampered real item is flagged, and how often it is scored above its original."""
    orig_of = {it["id"]: it["orig_id"] for it in ds_tamper.split("test")}
    det, fpr, pair = [], [], []
    for seed in sorted(runs):
        pred = load_preds(tag, name, seed)
        if pred is None or "tamper/prob" not in pred:
            continue
        p_orig = dict(zip(pred["test/id"].tolist(), pred["test/prob"].tolist()))
        pt = pred["tamper/prob"]
        po = np.array([p_orig[orig_of[int(i)]] for i in pred["tamper/id"]])
        det.append(float((pt >= 0.5).mean()))
        fpr.append(float((po >= 0.5).mean()))
        pair.append(float((pt > po).mean() + 0.5 * (pt == po).mean()))
    if not det:
        return {}
    return {"flag_rate_tampered": float(np.mean(det)), "flag_rate_original": float(np.mean(fpr)),
            "pairwise_acc": float(np.mean(pair)), "pairwise_acc_std": float(np.std(pair, ddof=1)) if len(pair) > 1 else 0.0}


# --------------------------------------------------------------------------------------
# selective prediction
# --------------------------------------------------------------------------------------

def selective_summary(tag: str, name: str, runs: Runs) -> Dict[str, Dict[str, float]]:
    acc: Dict[str, Dict[str, List[float]]] = {}
    for seed in sorted(runs):
        pred = load_preds(tag, name, seed)
        if pred is None:
            continue
        for kind, m in selective.evaluate(pred).items():
            for k, v in m.items():
                acc.setdefault(kind, {}).setdefault(k, []).append(v)
    return {kind: {k: float(np.mean(v)) for k, v in d.items()} for kind, d in acc.items()}


def risk_coverage_curve(tag: str, name: str, runs: Runs, kind: str = "maxprob",
                        grid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Selective accuracy as a function of coverage, averaged over seeds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    grid = np.linspace(0.5, 1.0, 26) if grid is None else grid
    curves = []
    for seed in sorted(runs):
        pred = load_preds(tag, name, seed)
        if pred is None:
            continue
        y, p = pred["test/label"], pred["test/prob"]
        conf = np.abs(p - 0.5)
        if kind == "selector":
            fv, ft = selective.features(pred, "val"), selective.features(pred, "test")
            if fv is None:
                continue
            cv = ((pred["val/prob"] >= 0.5).astype(int) == pred["val/label"]).astype(int)
            sel = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(fv, cv)
            conf = sel.predict_proba(ft)[:, 1]
        correct = ((p >= 0.5).astype(int) == y).astype(float)[np.argsort(-conf, kind="stable")]
        cum = np.cumsum(correct) / np.arange(1, len(y) + 1)
        curves.append([cum[max(1, int(round(c * len(y)))) - 1] for c in grid])
    return grid, (np.mean(curves, axis=0) if curves else np.full(len(grid), np.nan))
