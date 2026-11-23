#!/usr/bin/env python
"""Compute every number used in the paper from finished runs -> results/analysis.json.

    python scripts/analyze.py [--llm qwen25] [--strict]

Sections whose runs are not finished yet are skipped with a message (fatal with --strict), so
the script can be run while the training pipeline is still going.
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from clara import analysis as A  # noqa: E402
from clara.data import SPLITS, Dataset  # noqa: E402
from clara.metrics import detection_metrics  # noqa: E402
from clara.report import load_preds, load_runs, select_by_val, stat, summarise  # noqa: E402

DATASETS = ["ARG-CN", "ARG-EN"]
MAIN = ["bert", "lora", "bert_zs", "arg_gpt", "arg", "factguard", "clara", "arg_d", "factguard_d", "clara_d"]
EXTRA = ["arg_swap", "bert_claims", "bert_tc"]          # controls and analysis-only classifiers
DOMAINS = ["社会生活", "财经商业", "医药健康", "灾难事故", "文体娱乐", "政治"]
NOISE = [20, 40, 60, 80, 100]
UNIT = [25, 50, 100]
SWEEP_RE = r"clara-(k\d+|swap[\d.]+|lrel[\d.]+|lmil[\d.]+|lgate[\d.]+)$"


class Missing(Exception):
    pass


def need(runs, name):
    if name not in runs or not runs[name]:
        raise Missing("no runs for '{}'".format(name))
    return runs[name]


def ms(values):
    m, s = stat(values)
    return {"mean": m, "std": s, "n": len(list(values))}


# ----------------------------------------------------------------------------------------------
def main_table(tag, runs, chosen):
    out = {}
    for base in MAIN + EXTRA:
        if base not in chosen:
            continue
        name = chosen[base]
        s = summarise(runs[name], "test")
        s["name"] = name
        s["n_trainable"] = int(np.mean([r["n_trainable"] for r in runs[name].values()]))
        s["train_seconds"] = float(np.mean([r["summary"]["train_seconds"] for r in runs[name].values()]))
        out[base] = s
    if "clara" not in out:
        raise Missing("no CLARA runs")
    return out


def grid_table(runs):
    return {name: {"val": ms(r["metrics"]["val"]["macf1"] for r in seeds.values()),
                   "test": ms(r["metrics"]["test"]["macf1"] for r in seeds.values())}
            for name, seeds in sorted(runs.items()) if "@lr" in name}


def significance(tag, runs, chosen, ref="clara"):
    out = {}
    for base in MAIN + EXTRA:
        if base == ref or base not in chosen:
            continue
        ra, rb = need(runs, chosen[ref]), runs[chosen[base]]
        common = sorted(set(ra) & set(rb))          # e.g. 3 seeds for the fine-tuned LLM
        out[base] = A.compare(tag, {s: ra[s] for s in common}, chosen[ref], {s: rb[s] for s in common}, chosen[base])
        out[base]["n_seeds"] = len(common)
    return out


def noise_significance(tag, runs, chosen):
    """CLARA vs ARG / ARG with mismatch negatives on the fully replaced test sets."""
    out = {}
    for split in ("noise100", "unit100", "unit50"):
        for base in ("arg", "arg_swap"):
            if base in chosen:
                a, b = need(runs, chosen["clara"]), runs[chosen[base]]
                if all(split in r["metrics"] for r in list(a.values()) + list(b.values())):
                    out["{}/{}".format(split, base)] = A.compare(tag, a, chosen["clara"], b, chosen[base], split=split)
    if not out:
        raise Missing("no replacement evaluations")
    return out


def noise_curves(runs, chosen):
    """macro-F1 under item-level (noiseXX) and unit-level (unitXX) replacement of the LLM analysis."""
    def curve(seeds, prefix, rates):
        c = {"0": ms(r["metrics"]["test"]["macf1"] for r in seeds.values())}
        for n in rates:
            key = "{}{:02d}".format(prefix, n)
            if all(key in r["metrics"] for r in seeds.values()):
                c[str(n)] = ms(r["metrics"][key]["macf1"] for r in seeds.values())
        return c

    out = {}
    names = {b: chosen[b] for b in ["clara", "arg", "arg_swap", "factguard", "bert_zs", "bert"] if b in chosen}
    names.update({n: n for n in runs if n.startswith("clara-") and not re.match(SWEEP_RE, n)})
    for label, name in names.items():
        out[label] = {"item": curve(runs[name], "noise", NOISE), "unit": curve(runs[name], "unit", UNIT)}
    return out


def gate_section(ds, tag, runs, chosen):
    name = chosen["clara"]
    out = {"clara": A.gate_analysis(ds, tag, name, need(runs, name)),
           "relevance": A.relevance_under_noise(ds, tag, name, runs[name], ds.lang),
           "weights_under_noise": {}, "usefulness": {}}
    for base in ("arg", "arg_swap", "factguard"):
        if base in chosen:
            out["weights_under_noise"][base] = A.gate_under_noise(tag, chosen[base], runs[chosen[base]], "pair")
    for base, source in (("arg", "llm"), ("arg_swap", "llm"), ("arg_gpt", "gpt")):
        if base in chosen:
            out["usefulness"][base] = A.arg_gate_auc(ds, tag, chosen[base], runs[chosen[base]], source)
    for abl in ("clara-item_gate", "clara-no_relevance", "clara-entangled"):
        if abl in runs:
            out[abl] = A.gate_analysis(ds, tag, abl, runs[abl])
    return out


def style_llm_section(ds, ds_style):
    """The LLM's own judgments before and after the style transfer (no trained model involved)."""
    orig = {it["id"]: it for it in ds.split("test")}
    st = [it for it in ds_style.split("test") if it.get("rewritten")]
    y = np.array([it["label"] for it in st])
    getters = {
        "zero-shot P(fake)": lambda it: it["zs"]["p_fake"],
        "overall verdict": lambda it: it["clara"]["overall"]["v"],
        "max claim suspicion": lambda it: max(c["v"] for c in it["clara"]["claims"]),
        "mean claim suspicion": lambda it: float(np.mean([c["v"] for c in it["clara"]["claims"]])),
    }
    out = {}
    for name, get in getters.items():
        a = np.array([get(orig[it["orig_id"]]) for it in st], dtype=np.float64)
        b = np.array([get(it) for it in st], dtype=np.float64)
        out[name] = {"auc_orig": detection_metrics(y, a)["auc"], "auc_style": detection_metrics(y, b)["auc"],
                     "fake_orig": float(a[y == 1].mean()), "fake_style": float(b[y == 1].mean()),
                     "real_orig": float(a[y == 0].mean()), "real_style": float(b[y == 0].mean())}
    for k in ("td", "cs"):
        a = np.array([orig[it["orig_id"]][k]["pred"] == 1 for it in st])
        b = np.array([it[k]["pred"] == 1 for it in st])
        out["hard judgment, " + k] = {"fake_orig": float(a[y == 1].mean()), "fake_style": float(b[y == 1].mean()),
                                      "real_orig": float(a[y == 0].mean()), "real_style": float(b[y == 0].mean())}
    out["_claims_orig"] = float(np.mean([len(orig[it["orig_id"]]["clara"]["claims"]) for it in st]))
    out["_claims_style"] = float(np.mean([len(it["clara"]["claims"]) for it in st]))
    return out


def style_section(ds_style, tag, runs, chosen):
    """Original vs style-transferred text, on the items that were actually rewritten."""
    items = ds_style.split("test")
    rew = {it["id"]: bool(it.get("rewritten")) for it in items}
    orig = {it["id"]: it["orig_id"] for it in items}
    out = {"_n_items": len(items), "_n_rewritten": int(sum(rew.values())),
           "_n_fake_rewritten": int(sum(rew[it["id"]] and it["label"] == 1 for it in items))}
    for base in MAIN + EXTRA:
        if base not in chosen:
            continue
        name, per = chosen[base], []
        for seed in sorted(runs[name]):
            pred = load_preds(tag, name, seed)
            if pred is None or "style/prob" not in pred:
                continue
            sid = pred["style/id"]
            keep = np.array([rew[int(i)] for i in sid])
            y, ps = pred["style/label"][keep], pred["style/prob"][keep]
            p_test = dict(zip(pred["test/id"].tolist(), pred["test/prob"].tolist()))
            po = np.array([p_test[orig[int(i)]] for i in sid[keep]])
            mo, mst = detection_metrics(y, po), detection_metrics(y, ps)
            per.append({
                "orig": mo["macf1"], "style": mst["macf1"], "delta": mst["macf1"] - mo["macf1"],
                "orig_auc": mo["auc"], "style_auc": mst["auc"],
                "fake_recall_orig": float((po[y == 1] >= 0.5).mean()), "fake_recall_style": float((ps[y == 1] >= 0.5).mean()),
                "real_recall_orig": float((po[y == 0] < 0.5).mean()), "real_recall_style": float((ps[y == 0] < 0.5).mean()),
                "flip_rate": float(((po >= 0.5) != (ps >= 0.5)).mean()),
            })
        if per:
            out[base] = {k: ms(p[k] for p in per) for k in per[0]}
    if "clara" not in out:
        raise Missing("no style predictions for CLARA")
    return out


def fact_preservation(ds, ds_style):
    """Cheap proxy: share of the digit strings of the original that survive in the rewrite."""
    by_id = {it["id"]: it for it in ds.split("test")}
    kept, total, ratios = 0, 0, []
    for it in ds_style.split("test"):
        if not it.get("rewritten"):
            continue
        o = by_id[it["orig_id"]]["content"]
        nums = set(re.findall(r"\d+(?:[.,]\d+)*", o))
        total += len(nums)
        kept += sum(n in it["content"] for n in nums)
        ratios.append(len(it["content"]) / max(len(o), 1))
    return {"digit_strings": total, "digit_strings_kept": kept,
            "kept_rate": kept / max(total, 1), "length_ratio_median": float(np.median(ratios))}


def perturb_section(D, ds_style, ds_tamper):
    st, tm = ds_style.split("test"), ds_tamper.split("test")
    types = {}
    for it in tm:
        types[it["change_type"]] = types.get(it["change_type"], 0) + 1
    out = {"style_n": len(st), "style_rewritten": int(sum(bool(it.get("rewritten")) for it in st)),
           "style_fake_rewritten": int(sum(bool(it.get("rewritten")) and it["label"] == 1 for it in st)),
           "style_real_rewritten": int(sum(bool(it.get("rewritten")) and it["label"] == 0 for it in st)),
           "tamper_n": len(tm), "tamper_types": types,
           "tamper_mean_sents": float(np.mean([len(it["sents"]) for it in tm]))}
    path = os.path.join(ROOT, "outputs", "perturb", D + "-tamper", "input.json")
    if os.path.exists(path):
        out["tamper_candidates"] = len(json.load(open(path, encoding="utf-8")))
    return out


def localisation_section(ds_tamper, tag, runs, chosen):
    rows = {}
    for label, base in [("BERT + occlusion", "bert"), ("LoRA-LLM + occlusion", "lora"),
                        ("CLARA", "clara"), ("CLARA-D", "clara_d")]:
        if base in chosen:
            rows[label] = (chosen[base], runs[chosen[base]])
    for name in sorted(runs):
        if name.startswith(("clara-", "clara_d-")) and not re.match(SWEEP_RE, name):
            rows[name] = (name, runs[name])
    out = A.localisation(ds_tamper, tag, rows)
    if "CLARA" not in out:
        raise Missing("no tamper predictions for CLARA")
    out["_by_type"] = {}
    for t in sorted({it["change_type"] for it in ds_tamper.split("test")}):
        ids = {it["id"] for it in ds_tamper.split("test") if it["change_type"] == t}
        if len(ids) >= 30:
            out["_by_type"][t] = A.localisation(
                ds_tamper, tag, {k: rows[k] for k in ("CLARA", "BERT + occlusion", "CLARA-D") if k in rows}, only=ids)
    sens = {}
    for base in MAIN:
        if base in chosen:
            d = A.tamper_detection(ds_tamper, tag, chosen[base], runs[chosen[base]])
            if d:
                sens[base] = d
    out["_sensitivity"] = sens
    return out


def case_section(ds_tamper, tag, runs, chosen, n=2, seed=1, max_chars=170):
    """Worked examples, chosen by a rule fixed before looking at any output: the first ``n``
    tampered items in id order that have 3-4 sentences and at most ``max_chars`` characters."""
    from clara.data import instances

    name = chosen["clara"]
    pred = load_preds(tag, name, seed)
    if pred is None or "tamper/inst_g" not in pred:
        raise Missing("no tamper predictions for CLARA seed {}".format(seed))
    cfg = runs[name][seed]["config"]["model"]
    items = ds_tamper.split("test")
    by_id = {it["id"]: it for it in items}
    T = A.instance_table(ds_tamper, items, pred, "tamper", cfg)
    ids = pred["tamper/id"]
    cases = []
    for i in np.argsort(ids):
        it = by_id[int(ids[i])]
        if not (3 <= len(it["sents"]) <= 4 and len(it["content"]) <= max_chars):
            continue
        m = np.where(T["item"] == i)[0]
        inst = instances(it, ds_tamper.lang, cfg["max_claims"], cfg["v0_source"])
        assert len(inst) == len(m)
        rows = [{"claim": x["claim"], "type": x["type"], "sent": int(x["sent"]), "v": float(x["v"]),
                 "v0": float(x["v0"]), "support": x["support"], "refute": x["refute"],
                 "g": float(T["g"][j]), "p": float(T["p"][j])} for x, j in zip(inst, m)]
        cases.append({"id": int(it["id"]), "sents": it["sents"], "tamper_sent": int(it["tamper_sent"]),
                      "orig_sentence": it["orig_sentence"], "new_sentence": it["new_sentence"],
                      "change_type": it["change_type"], "prob": float(pred["tamper/prob"][i]),
                      "instances": rows, "seed": seed})
        if len(cases) == n:
            break
    return cases


def selective_section(tag, runs, chosen):
    out, curves = {}, {}
    for base in ["bert", "lora", "bert_zs", "arg", "factguard", "clara", "clara_d"]:
        if base not in chosen:
            continue
        name = chosen[base]
        s = A.selective_summary(tag, name, runs[name])
        if s:
            out[base] = s
            for kind in s:
                grid, c = A.risk_coverage_curve(tag, name, runs[name], kind)
                curves["{}/{}".format(base, kind)] = {"coverage": grid.tolist(), "accuracy": c.tolist()}
    if "clara" not in out:
        raise Missing("no predictions for CLARA")
    out["_curves"] = curves
    return out


def ablation_section(tag, runs, chosen, ds_tamper=None):
    full_name = chosen["clara"]
    full = need(runs, full_name)
    names = [full_name] + sorted(n for n in runs if n.startswith("clara-") and not re.match(SWEEP_RE, n))
    if len(names) == 1:
        raise Missing("no ablation runs")
    out = {}
    for n in names:
        seeds = runs[n]
        row = {"test": ms(r["metrics"]["test"]["macf1"] for r in seeds.values()),
               "val": ms(r["metrics"]["val"]["macf1"] for r in seeds.values()),
               "auc": ms(r["metrics"]["test"]["auc"] for r in seeds.values())}
        for key in ("noise100", "unit50"):
            if all(key in r["metrics"] for r in seeds.values()):
                row[key] = ms(r["metrics"][key]["macf1"] for r in seeds.values())
        if n != full_name:
            common = sorted(set(seeds) & set(full))
            row["vs_full"] = A.compare(tag, {s: full[s] for s in common}, full_name, {s: seeds[s] for s in common}, n)
        out["full" if n == full_name else n[len("clara-"):]] = row
    return out


def sweep_section(runs, chosen, pattern, key):
    rows = {}
    for n, seeds in runs.items():
        m = re.match(pattern, n)
        if m:
            rows[m.group(1)] = {"test": ms(r["metrics"]["test"]["macf1"] for r in seeds.values()),
                                "val": ms(r["metrics"]["val"]["macf1"] for r in seeds.values())}
    if not rows:
        raise Missing("no runs matching " + pattern)
    full = runs[chosen["clara"]]
    default = str(next(iter(full.values()))["config"]["model"][key])
    seeds3 = {s: r for s, r in full.items() if s in (1, 2, 3)} or full
    rows[default] = {"test": ms(r["metrics"]["test"]["macf1"] for r in seeds3.values()),
                     "val": ms(r["metrics"]["val"]["macf1"] for r in seeds3.values()), "default": True}
    return rows


def lodo_section(D, llm, chosen):
    out = {}
    for dom in DOMAINS:
        runs = load_runs("{}.{}-holdout-{}".format(D, llm, dom))
        if not runs:
            continue
        row = {}
        for base in ("bert", "bert_zs", "arg", "clara"):
            name = chosen.get(base)
            if name in runs:
                seeds = runs[name]
                row[base] = ms(r["metrics"]["test"]["macf1"] for r in seeds.values())
                row[base]["auc"] = float(np.mean([r["metrics"]["test"]["auc"] for r in seeds.values()]))
                row["_n_test"] = next(iter(seeds.values()))["n_items"]["test"]
                row["_n_train"] = next(iter(seeds.values()))["n_items"]["train"]
        if row:
            out[dom] = row
    if not out:
        raise Missing("no leave-one-domain-out runs")
    bases = [b for b in ("bert", "bert_zs", "arg", "clara") if all(b in r for r in out.values())]
    out["_macro_avg"] = {b: float(np.mean([out[d][b]["mean"] for d in out if not d.startswith("_")])) for b in bases}
    return out


def live_section(D, llm, runs, chosen):
    live = load_runs("{}.{}-live".format(D, llm))
    if not live:
        raise Missing("no live-encoder runs")
    out = {}
    for name, seeds in live.items():
        cached = {s: r for s, r in runs.get(name, {}).items() if s in seeds}
        out[name] = {"live": ms(r["metrics"]["test"]["macf1"] for r in seeds.values()),
                     "live_seconds": float(np.mean([r["seconds"] for r in seeds.values()]))}
        if cached:
            out[name]["cached"] = ms(r["metrics"]["test"]["macf1"] for r in cached.values())
            out[name]["cached_seconds"] = float(np.mean([r["seconds"] for r in cached.values()]))
    return out


def cost_section(D, llm):
    llm_dir = os.path.join(ROOT, "outputs", "llm", llm)
    out = {"tokens": {}}
    for task in ["clara", "td", "cs", "tc", "zs", "cot"]:
        recs = []
        for sp in SPLITS:
            path = os.path.join(llm_dir, "{}.{}.{}.jsonl".format(D, sp, task))
            if not os.path.exists(path):
                raise Missing("no LLM cache " + path)
            with open(path, encoding="utf-8") as f:
                recs += [json.loads(line) for line in f if line.strip()]
        out["tokens"][task] = {"prompt": float(np.mean([r["prompt_tokens"] for r in recs])),
                               "completion": float(np.mean([r["completion_tokens"] for r in recs])), "n": len(recs)}
        if task == "clara":
            out["clara_second_attempt_rate"] = float(np.mean([r["attempt"] == 2 for r in recs]))
            out["clara_first_attempt_rate"] = 1.0 - out["clara_second_attempt_rate"]
            out["clara_parse_rate"] = float(np.mean([r["parsed"] is not None for r in recs]))
    # wall-clock of the offline generation, from the log of the generation script
    log = os.path.join(ROOT, "outputs", "logs", "gen_{}.log".format(llm))
    secs, cur = {}, None
    if os.path.exists(log):
        for line in open(log, encoding="utf-8", errors="ignore"):
            m = re.match(r"\[(\w+)\] ([\w.-]+): (\d+) items, (\d+) cached, (\d+) to run", line)
            if m:
                cur = (m.group(1), m.group(2), int(m.group(5)))
                continue
            m = re.match(r"\s+(\d+)/(\d+)\s+(\d+)s", line)
            if m and cur and cur[1].startswith(D + ".") and cur[2] > 0 and int(m.group(1)) == int(m.group(2)) == cur[2]:
                secs.setdefault(cur[0], [0.0, 0])
                secs[cur[0]][0] += float(m.group(3))
                secs[cur[0]][1] += cur[2]
    out["gen_seconds_per_item"] = {t: v[0] / v[1] for t, v in secs.items() if v[1]}
    return out


def llm2_section(D, chosen, tag2, runs1):
    """The LLM-dependent methods with a second analyst LLM, at the learning rates selected with
    the first one (seeds 1-3; the first-LLM numbers are restricted to the same seeds)."""
    runs2 = load_runs("{}.{}".format(D, tag2))
    if not runs2:
        raise Missing("no runs for the second LLM")
    out = {"_llm": tag2}
    for base in ("bert_zs", "arg", "factguard", "clara"):
        name = chosen.get(base)
        if name in runs2:
            seeds = runs2[name]
            first = {s: r for s, r in runs1[name].items() if s in seeds}
            out[base] = {"second": ms(r["metrics"]["test"]["macf1"] for r in seeds.values()),
                         "first": ms(r["metrics"]["test"]["macf1"] for r in first.values())}
            for key in ("noise100", "unit100"):
                if all(key in r["metrics"] for r in seeds.values()):
                    out[base]["second_" + key] = ms(r["metrics"][key]["macf1"] for r in seeds.values())
    path = os.path.join(ROOT, "results", "llm_only", "{}.{}.json".format(D, tag2))
    if os.path.exists(path):
        rows = json.load(open(path, encoding="utf-8"))["rows"]
        out["zero_shot"] = rows["zero-shot"]["test"]["macf1"]
        if "zero-shot, soft P(fake) @0.5" in rows:
            out["zero_shot_auc"] = rows["zero-shot, soft P(fake) @0.5"]["test"]["auc"]
    return out


def derived_section(res):
    """Differences quoted in the prose, computed here so that no number is derived by hand."""
    out = {}
    main, noise, abl = res.get("main", {}), res.get("noise", {}), res.get("ablation", {})

    def put(key, fn):
        try:
            out[key] = float(fn())
        except (KeyError, TypeError):
            pass

    bert = lambda: main["bert"]["macf1"]
    put("no_relevance_gap", lambda: bert() - abl["no_relevance"]["noise100"]["mean"])
    put("clara_item100_gap", lambda: bert() - noise["clara"]["item"]["100"]["mean"])
    put("lora_minus_clara", lambda: main["lora"]["macf1"] - main["clara"]["macf1"])
    put("lora_minus_bert", lambda: main["lora"]["macf1"] - bert())
    put("clara_minus_bert", lambda: main["clara"]["macf1"] - bert())
    put("bertzs_minus_bert", lambda: main["bert_zs"]["macf1"] - bert())
    for b in ("arg", "factguard", "clara"):
        put(b + "_item100_drop", lambda b=b: noise[b]["item"]["0"]["mean"] - noise[b]["item"]["100"]["mean"])
        put(b + "_unit100_drop", lambda b=b: noise[b]["unit"]["0"]["mean"] - noise[b]["unit"]["100"]["mean"])
    for b in ("clara_d", "arg_d", "factguard_d"):
        put(b + "_minus_bert", lambda b=b: main[b]["macf1"] - bert())
    put("clarad_minus_clara", lambda: main["clara_d"]["macf1"] - main["clara"]["macf1"])
    if not out:
        raise Missing("nothing to derive")
    return out


def data_section(ds):
    out = {}
    for sp in SPLITS:
        its = ds.split(sp)
        t = [it["time"] for it in its]
        out[sp] = {"n": len(its), "fake": int(sum(it["label"] for it in its)),
                   "chars": float(np.mean([len(it["content"]) for it in its])),
                   "sents": float(np.mean([len(it["sents"]) for it in its])),
                   "claims": float(np.mean([len(it["clara"]["claims"]) for it in its if it.get("clara")])),
                   "parsed": int(sum(bool(it.get("clara")) for it in its)),
                   "time_min": str(min(t)), "time_max": str(max(t))}
    its = ds.items
    types = {}
    for it in its:
        for c in (it.get("clara") or {}).get("claims", []):
            types[c["type"]] = types.get(c["type"], 0) + 1
    out["claim_types"] = types
    out["domains"] = {}
    for it in its:
        if "domain" in it:
            d = out["domains"].setdefault(it["domain"], {"train": 0, "val": 0, "test": 0})
            d[it["split"]] += 1
    return out


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="qwen25")
    ap.add_argument("--llm2", default="qwen3", help="tag of the second analyst LLM (robustness check)")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "analysis.json"))
    args = ap.parse_args()

    result, skipped = {}, []
    for D in DATASETS:
        tag = "{}.{}".format(D, args.llm)
        ds = Dataset.load(os.path.join(ROOT, "outputs", "datasets", tag + ".json"))
        runs = load_runs(tag)
        chosen = select_by_val(runs) if runs else {}
        res = result.setdefault(D, {"selected": {b: chosen[b] for b in MAIN if b in chosen}})

        def load_extra(kind):
            path = os.path.join(ROOT, "outputs", "datasets", "{}-{}.{}.json".format(D, kind, args.llm))
            if not os.path.exists(path):
                raise Missing("no dataset " + path)
            return Dataset.load(path)

        def section(name, fn):
            try:
                res[name] = fn()
            except (Missing, KeyError) as e:
                if args.strict:
                    raise
                skipped.append("{}/{}: {}".format(D, name, e))

        section("data", lambda: data_section(ds))
        section("cost", lambda: cost_section(D, args.llm))
        section("llm_only", lambda: json.load(open(os.path.join(ROOT, "results", "llm_only", tag + ".json"), encoding="utf-8"))["rows"])
        section("main", lambda: main_table(tag, runs, chosen))
        section("grid", lambda: grid_table(runs))
        section("significance", lambda: significance(tag, runs, chosen))
        section("noise", lambda: noise_curves(runs, chosen))
        section("noise_sig", lambda: noise_significance(tag, runs, chosen))
        section("gate", lambda: gate_section(ds, tag, runs, chosen))
        section("style", lambda: style_section(load_extra("style"), tag, runs, chosen))
        section("fact_preservation", lambda: fact_preservation(ds, load_extra("style")))
        section("perturb", lambda: perturb_section(D, load_extra("style"), load_extra("tamper")))
        section("localisation", lambda: localisation_section(load_extra("tamper"), tag, runs, chosen))
        section("cases", lambda: case_section(load_extra("tamper"), tag, runs, chosen))
        section("selective", lambda: selective_section(tag, runs, chosen))
        section("ablation", lambda: ablation_section(tag, runs, chosen))
        section("style_llm", lambda: style_llm_section(ds, load_extra("style")))
        section("claims_k", lambda: sweep_section(runs, chosen, r"clara-k(\d+)$", "max_claims"))
        section("swap_rate", lambda: sweep_section(runs, chosen, r"clara-swap([\d.]+)$", "swap_rate"))
        section("lambda_rel", lambda: sweep_section(runs, chosen, r"clara-lrel([\d.]+)$", "lambda_rel"))
        section("lambda_mil", lambda: sweep_section(runs, chosen, r"clara-lmil([\d.]+)$", "lambda_mil"))
        section("lambda_gate", lambda: sweep_section(runs, chosen, r"clara-lgate([\d.]+)$", "lambda_gate"))
        section("lodo", lambda: lodo_section(D, args.llm, chosen))
        section("live", lambda: live_section(D, args.llm, runs, chosen))
        section("llm2", lambda: llm2_section(D, chosen, args.llm2, runs))
        section("derived", lambda: derived_section(res))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("wrote", args.out)
    for D in DATASETS:
        print(D, "sections:", sorted(k for k in result[D] if k != "selected"))
    for s in skipped:
        print("SKIPPED", s)


if __name__ == "__main__":
    main()
