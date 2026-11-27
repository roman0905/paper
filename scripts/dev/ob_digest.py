#!/usr/bin/env python
"""A compact, readable digest of results/ob/analysis.json (or of a preview written elsewhere).

Used while writing: every qualitative statement of the thesis chapter is checked against these
lines.  Nothing is computed here.

    python scripts/dev/ob_digest.py [--analysis PATH] [--only RAWFC] [--grep swap]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

CONDS = ("val", "test", "test@droptop", "test@halfswap", "test@nearswap", "test@swap", "test@closed", "test@all", "test@strict",
         "test@bm25", "test@read", "test@avg", "test@oracle", "xfer")


def f(x, d=2, scale=100.0):
    return "  --  " if x is None or x != x else "{:6.{d}f}".format(scale * x, d=d)


def p(x):
    return " -- " if x is None or x != x else ("<.001" if x < 0.001 else "{:.3f}".format(x))


def sig(c):
    b, w = c.get("bootstrap"), c.get("welch_p")
    return bool(b) and b["p"] < 0.05 and w is not None and w == w and w < 0.05


def cmp_line(key, c):
    b = c.get("bootstrap") or {}
    return "{:42s} a={} b={} d={}  CI[{},{}] p_item={} p_seed={} n={}/{} items={} {}".format(
        key, f(c.get("a")), f(c.get("b")), f(c.get("diff_mean")), f(b.get("ci_lo")), f(b.get("ci_hi")), p(b.get("p")), p(c.get("welch_p")),
        c.get("n_a", "-"), c.get("n_b", "-"), c.get("n_items", "-"), "SIG" if sig(c) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=os.path.join(ROOT, "results", "ob", "analysis.json"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--grep", default=None, help="only lines containing this text")
    args = ap.parse_args()
    with open(args.analysis, encoding="utf-8") as fh:
        A = json.load(fh)
    lines = []
    out = lines.append
    if "decision" in A:
        out("DECISION " + json.dumps(A["decision"], ensure_ascii=False)[:900])
    for D in ("RAWFC", "LIAR-RAW"):
        if args.only and D != args.only:
            continue
        R = A[D]
        for protocol in ("strict", "all"):
            S = R.get(protocol)
            if not S:
                continue
            tag = "{}.{}".format(D, protocol)
            out("=" * 30 + " " + tag + "  final=" + str(S.get("final")))
            for k, m in S["methods"].items():
                row = []
                for c in CONDS:
                    if c in m:
                        t = m[c]["macf1"]
                        row.append("{}={}±{}({})".format(c.replace("test@", "@"), f(t["mean"]).strip(), f(t["std"]).strip(), m[c]["n"]))
                out("{} M {:12s} seeds={} ep={} | {}".format(tag, k, m["seeds"], m["best_epoch"], " ".join(row)))
                t = m["test"]
                out("{} M {:12s}   test P={} R={} Acc={} F1h={} MAE={} ECE={} | train {:.0f} min, {:.1f} GB".format(
                    tag, k, f(t["macp"]["mean"]), f(t["macr"]["mean"]), f(t["acc"]["mean"]), f(t["f1_h"]["mean"]),
                    f(t["mae"]["mean"], 3, 1.0), f(t.get("ece", {}).get("mean"), 3, 1.0), m["train_seconds"] / 60, m.get("max_memory_gb", 0.0)))
                ex = m.get("extra") or {}
                if ex:
                    out("{} X {:12s} ".format(tag, k) + " ".join(
                        "{}={}".format(a, ("[" + ",".join("{:.2f}".format(x) for x in v) + "]") if isinstance(v, list) else "{:.3f}".format(v["mean"]))
                        for a, v in ex.items()))
            for key, c in sorted(S.get("compare", {}).items()):
                out("{} C {}".format(tag, cmp_line(key, c)))
            E = S.get("ensemble", {})
            for k, e in E.get("methods", {}).items():
                out("{} E {:12s} n={} F1={} Acc={}".format(tag, k, e["n_seeds"], f(e["macf1"]), f(e["acc"])))
            for k, c in E.get("compare", {}).items():
                out("{} E {}".format(tag, cmp_line(k, c)))
            L = S.get("leak", {})
            out("{} L n={}".format(tag, L.get("n")))
            for k, row in L.get("methods", {}).items():
                for sub, r in row.items():
                    out("{} L {:10s} {:8s} own acc={} f1={} | other acc={} f1={} | {}".format(
                        tag, k, sub, f(r["own"]["acc"]), f(r["own"]["macf1"]), f(r["other"]["acc"]), f(r["other"]["macf1"]),
                        cmp_line("other-own", r["compare"]) if r.get("compare") else ""))
            G = S.get("gate")
            if G:
                out("{} G seeds={} from_reeval={}".format(tag, G["seeds"], G["seeds_from_reeval"]))
                for q, row in G["q"].items():
                    for c, r in row.items():
                        out("{} G q={:5s} {:14s} plain={} gated={} closed={} flagged={} | vs_plain {} | vs_closed {}".format(
                            tag, q, c.replace("test@", "@"), f(r["plain"]["mean"]), f(r["gated"]["mean"]), f(r["closed"]["mean"]),
                            f(r["flagged"]["mean"], 1), cmp_line("", r["vs_plain"]).strip(), cmp_line("", r["vs_closed"]).strip()))
            Ab = S.get("ablation")
            if Ab:
                out("{} A seeds={} explain_row={}".format(tag, Ab["seeds"], Ab["explain_row"]))
                for k, m in Ab["methods"].items():
                    out("{} A {:10s} ".format(tag, k) + " ".join("{}={}±{}".format(c.replace("test@", "@"), f(v["mean"]).strip(), f(v["std"]).strip()) for c, v in m.items()))
                for k, row in Ab["vs_full"].items():
                    for c, x in row.items():
                        out("{} A {}".format(tag, cmp_line("{}-full {}".format(k, c), x)))
            X = S.get("explain")
            if X:
                out("{} J seeds={} train_seconds={}".format(tag, X["seeds"], X["train_seconds"]))
                for k, m in X["methods"].items():
                    out("{} J {:8s} ".format(tag, k) + " ".join("{}={}±{}".format(c.replace("test@", "@"), f(v["mean"]).strip(), f(v["std"]).strip()) for c, v in m.items()))
                for c, x in X["compare"].items():
                    out("{} J {}".format(tag, cmp_line("sievej-sieve " + c, x)))
                out("{} J justify {}".format(tag, {k: round(v["mean"], 4) for k, v in X["justify"].items()}))
            if S.get("reeval"):
                out("{} REEVAL {}".format(tag, S["reeval"]))
        for m, row in (R.get("cross_protocol") or {}).items():
            for k, c in row.items():
                if isinstance(c, dict):
                    out("{} P {:8s} {}".format(D, m, cmp_line(k, c)))
        for m, t in (R.get("transfer") or {}).items():
            out("{} T {:10s} ".format(D, m) + " ".join("{}={}±{}({})".format(k, f(v["mean"]).strip(), f(v["std"]).strip(), v["n"]) for k, v in t.items() if isinstance(v, dict) and "mean" in v)
                + (" | in-domain 3-class " + " ".join("{}={}".format(k, f(v["mean"]).strip()) for k, v in t["in_domain_three"].items()) if "in_domain_three" in t else ""))
        L2 = R.get("llm2")
        if L2:
            for k, m in L2["methods"].items():
                out("{} 2 {:12s} seeds={} ".format(D, k, m["seeds"]) + " ".join(
                    "{}={}±{}".format(c.replace("test@", "@"), f(m[c]["macf1"]["mean"]).strip(), f(m[c]["macf1"]["std"]).strip()) for c in CONDS if c in m))
            for key, c in sorted(L2.get("compare", {}).items()):
                out("{} 2 {}".format(D, cmp_line(key, c)))
    for line in lines:
        if not args.grep or args.grep in line:
            print(line)


if __name__ == "__main__":
    main()
