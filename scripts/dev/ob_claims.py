#!/usr/bin/env python
"""The qualitative statements of thesis chapter 4, checked against the analysis.

The running text takes its numbers from macros, so the numbers follow the results automatically;
the *wording* around them ("significant", "about the same", "three times as large") does not.
Every such statement of thesis/chapters/ch4_results.tex is listed here with the condition under
which it holds.  Run this after the analysis changes and reword whatever is reported as FAIL.

    python scripts/dev/ob_claims.py [--analysis PATH]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class NA(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=os.path.join(ROOT, "results", "ob", "analysis.json"))
    args = ap.parse_args()
    with open(args.analysis, encoding="utf-8") as f:
        A = json.load(f)
    pub = {p["key"]: p for p in A["published"]["methods"]}

    def v(path):
        d = A
        for k in path.split("/"):
            if not isinstance(d, dict) or k not in d:
                raise NA(path)
            d = d[k]
        return d

    def c(D, key):
        return v("{}/strict/compare/{}".format(D, key))

    def d(D, key):
        return 100 * c(D, key)["diff_mean"]

    def p_item(D, key):
        x = c(D, key)
        if "bootstrap" not in x:
            raise NA(key + " bootstrap")
        return x["bootstrap"]["p"]

    def p_seed(D, key):
        w = c(D, key)["welch_p"]
        if w is None or w != w:
            raise NA(key + " welch (needs 2+ seeds on both sides)")
        return w

    def sig(D, key):
        return p_item(D, key) < 0.05 and p_seed(D, key) < 0.05

    def f1(D, m, cond="test", stat="mean"):
        return 100 * v("{}/strict/methods/{}/{}/macf1/{}".format(D, m, cond, stat))

    def ex(D, m, key):
        return v("{}/strict/methods/{}/extra/{}/mean".format(D, m, key))

    def seeds(D, m, n):
        if v("{}/strict/methods/{}/n".format(D, m)) < n:
            raise NA("{} {} has fewer than {} seeds".format(D, m, n))
        return True

    R, L = "RAWFC", "LIAR-RAW"
    claims = [
        # ------------------------------------------------------------------ 4.5.1 main results
        ("main/seeds", "table note: 5 seeds on RAWFC, 3 on LIAR-RAW for the trained LLM methods",
         lambda: all(v("{}/strict/methods/{}/n".format(D, m)) == n for D, n in ((R, 5), (L, 3)) for m in ("lora_closed", "lora_ev", "ours"))),
        ("main/evidence-sig", "LoRA-LLM: evidence helps, significant on both datasets",
         lambda: sig(R, "lora_ev_vs_lora_closed") and sig(L, "lora_ev_vs_lora_closed") and d(R, "lora_ev_vs_lora_closed") > 0 < d(L, "lora_ev_vs_lora_closed")),
        ("main/evidence-3x", "the RAWFC gain is more than three times the LIAR-RAW gain",
         lambda: d(R, "lora_ev_vs_lora_closed") > 3 * d(L, "lora_ev_vs_lora_closed")),
        ("main/zs", "zero-shot LLM gains from evidence on RAWFC; with evidence it stays below 20 on LIAR-RAW and far below the tuned models",
         lambda: f1(R, "zs_ev") > f1(R, "zs_closed") + 5 and f1(L, "zs_ev") < 20 and f1(R, "zs_ev") < f1(R, "lora_ev") - 5),
        ("main/slm-R", "RoBERTa on RAWFC: gain passes the item test, fails the seed test",
         lambda: p_item(R, "slm_ev_vs_slm_closed") < 0.05 <= p_seed(R, "slm_ev_vs_slm_closed") and d(R, "slm_ev_vs_slm_closed") > 0),
        ("main/slm-L", "RoBERTa on LIAR-RAW: no gain from evidence (|diff| < 0.5)", lambda: abs(d(L, "slm_ev_vs_slm_closed")) < 0.5),
        ("main/lora-vs-slm", "LoRA+evidence above RoBERTa+evidence on both; significant on LIAR-RAW, item test fails on RAWFC",
         lambda: d(R, "lora_ev_vs_slm_ev") > 0 < d(L, "lora_ev_vs_slm_ev") and sig(L, "lora_ev_vs_slm_ev") and p_item(R, "lora_ev_vs_slm_ev") >= 0.05),
        ("main/slm-truncation", "most evidence sentences fit into 512 tokens (> 75% on both)",
         lambda: min(json.load(open(os.path.join(ROOT, "results", "ob", "slm_truncation.json")))["datasets"][D]["sentences_fit_share"] for D in (R, L)) > 0.75),
        ("main/ours-R", "ours - LoRA+evidence on RAWFC: positive, passes the seed test, fails the item test",
         lambda: d(R, "ours_vs_lora_ev") > 0 and p_seed(R, "ours_vs_lora_ev") < 0.05 <= p_item(R, "ours_vs_lora_ev")),
        ("main/ours-L", "ours - LoRA+evidence on LIAR-RAW: positive, smaller than on RAWFC, item test fails",
         lambda: seeds(L, "ours", 3) and 0 < d(L, "ours_vs_lora_ev") < d(R, "ours_vs_lora_ev") and p_item(L, "ours_vs_lora_ev") >= 0.05),
        ("main/ours-best", "ours has the highest macro-F1 of our methods on both datasets (bold in the table)",
         lambda: all(f1(D, "ours") >= max(f1(D, m) for m in ("zs_closed", "zs_ev", "slm_closed", "slm_ev", "lora_closed", "lora_ev")) for D in (R, L))),
        # ------------------------------------------------------------------ 4.5.2 published numbers
        ("pub/R", "RAWFC: on a par with DeReC and G-Defense (within 1.5), above L-Defense, below FACETS and DelphiAgent",
         lambda: abs(f1(R, "ours") - pub["qazi2025derec"]["rawfc"][2]) < 1.5 and abs(f1(R, "ours") - pub["wang2026gdefense"]["rawfc"][2]) < 1.5
         and pub["wang2024ldefense"]["rawfc"][2] < f1(R, "ours") < min(pub["nguyen2025facets"]["rawfc"][2], pub["xiong2025delphiagent"]["rawfc"][2])),
        ("pub/L", "LIAR-RAW: on a par with DeReC (within 1), above L-Defense and G-Defense, below FACETS",
         lambda: seeds(L, "ours", 3) and abs(f1(L, "ours") - pub["qazi2025derec"]["liar"][2]) < 1.0
         and max(pub["wang2024ldefense"]["liar"][2], pub["wang2026gdefense"]["liar"][2]) < f1(L, "ours") < pub["nguyen2025facets"]["liar"][2]),
        ("pub/kgcraft", "KG-CRAFT is about 40 points above every other published number on LIAR-RAW",
         lambda: 35 < pub["lourenco2026kgcraft"]["liar"][2] - max(p["liar"][2] for k, p in pub.items() if k != "lourenco2026kgcraft" and p.get("liar")) < 45),
        # ------------------------------------------------------------------ 4.5.3 leakage
        ("leak/L-2pts", "LIAR-RAW: flagged reports at test time add more than 2 points for both LoRA verifiers",
         lambda: seeds(L, "ours", 3) and d(L, "lora_ev@all_vs_own") > 2 and d(L, "ours@all_vs_own") > 2),
        ("leak/L-where", "the gain is on the claims with flagged reports; the others do not change (|diff| < 0.5 acc points)",
         lambda: v(L + "/strict/leak/methods/ours/any/other/acc") - v(L + "/strict/leak/methods/ours/any/own/acc") > 0.03
         and abs(v(L + "/strict/leak/methods/ours/none/other/acc") - v(L + "/strict/leak/methods/ours/none/own/acc")) < 0.005),
        ("leak/share", "flagged reports are less than 3% of all reports of LIAR-RAW", lambda: v(L + "/audit/total/reports_any_share") < 0.03),
        ("leak/L-others", "RoBERTa and the zero-shot LLM gain less than the LoRA verifiers on LIAR-RAW",
         lambda: max(d(L, "slm_ev@all_vs_own"), d(L, "zs_ev@all_vs_own")) < min(d(L, "lora_ev@all_vs_own"), d(L, "ours@all_vs_own"))),
        ("leak/R", "RAWFC: changes of about 1 point (< 1.5), none significant",
         lambda: all(abs(d(R, m + "@all_vs_own")) < 1.5 for m in ("zs_ev", "slm_ev", "lora_ev", "ours"))
         and not any(sig(R, m + "@all_vs_own") for m in ("slm_ev", "lora_ev", "ours"))),
        # ------------------------------------------------------------------ 4.5.4 unusable evidence
        ("robust/lora-collapse", "RAWFC: LoRA+evidence under foreign evidence is significantly below the closed-book verifier",
         lambda: sig(R, "lora_ev@swap_vs_lora_closed") and sig(R, "lora_ev@nearswap_vs_lora_closed") and d(R, "lora_ev@swap_vs_lora_closed") < -15),
        ("robust/lora-false", "it calls > 80% of the claims false, with higher confidence than on the original input",
         lambda: v(R + "/strict/methods/lora_ev/low_share/swap") > 0.8 and ex(R, "lora_ev", "swap_conf") > ex(R, "lora_ev", "own_conf")),
        ("robust/zs-false", "zero-shot LLM: > 95% false under random-claim evidence", lambda: v(R + "/strict/methods/zs_ev/low_share/swap") > 0.95),
        ("robust/slm", "RoBERTa+evidence under random-claim evidence: smaller drop than LoRA+evidence, but far below claim-only RoBERTa",
         lambda: d(R, "slm_ev@swap_vs_own") > d(R, "lora_ev@swap_vs_own") and f1(R, "slm_ev", "test@swap") < f1(R, "slm_closed") - 5),
        ("robust/ours-vs-lora", "RAWFC: ours significantly above LoRA+evidence under random- and similar-claim evidence",
         lambda: sig(R, "ours_vs_lora_ev@swap") and sig(R, "ours_vs_lora_ev@nearswap")),
        ("robust/ours-false", "ours calls 30-55% false and its confidence drops",
         lambda: 0.30 < v(R + "/strict/methods/ours/low_share/swap") < 0.55 and ex(R, "ours", "swap_conf") < ex(R, "ours", "own_conf")),
        ("robust/ours-gap", "RAWFC: ours under foreign evidence below closed-book; passes the item test, fails the seed test",
         lambda: all(d(R, k) < 0 and p_item(R, k) < 0.05 <= p_seed(R, k) for k in ("ours@swap_vs_lora_closed", "ours@nearswap_vs_lora_closed"))),
        ("robust/ours-seeds", "some seeds reach the closed-book level (within 1.5), some are more than 10 points below",
         lambda: all(f1(R, "ours", c_, "max") > f1(R, "lora_closed") - 1.5 and f1(R, "ours", c_, "min") < f1(R, "lora_closed") - 10
                     for c_ in ("test@swap", "test@nearswap"))),
        ("robust/ours-closed", "RAWFC: ours without any evidence is below the closed-book verifier", lambda: d(R, "ours@closed_vs_lora_closed") < -3),
        ("robust/L-smaller", "LIAR-RAW: the drops under foreign evidence are much smaller than on RAWFC",
         lambda: abs(d(L, "lora_ev@swap_vs_own")) < 0.3 * abs(d(R, "lora_ev@swap_vs_own"))),
        ("robust/halfswap-ours", "mixing in half foreign sentences does not affect ours (|diff| < 1 on both datasets)",
         lambda: abs(d(R, "ours@halfswap_vs_own")) < 1 and abs(d(L, "ours@halfswap_vs_own")) < 1),
        ("robust/halfswap-lora-R", "LoRA+evidence is hardly affected by the half-mixed pool on RAWFC (|diff| < 1.5)", lambda: abs(d(R, "lora_ev@halfswap_vs_own")) < 1.5),
        ("robust/droptop", "without the top 16 both LoRA verifiers lose 2-3 points on RAWFC and stay significantly above closed-book",
         lambda: all(-3.5 < d(R, m + "@droptop_vs_own") < -1.5 and sig(R, m + "@droptop_vs_lora_closed") for m in ("lora_ev", "ours"))),
        ("robust/ece", "ours is better calibrated than LoRA+evidence on both datasets; LoRA+evidence is overconfident on RAWFC",
         lambda: all(v("{}/strict/methods/ours/test/ece/mean".format(D)) < v("{}/strict/methods/lora_ev/test/ece/mean".format(D)) for D in (R, L))
         and ex(R, "lora_ev", "own_conf") - v(R + "/strict/methods/lora_ev/test/acc/mean") > 0.15),
        # ------------------------------------------------------------------ 4.5.5 evidence scores
        ("ev/silver", "selected sentences contain more silver evidence than the BM25 top 16, by a limited margin (< 15 points)",
         lambda: all(0 < ex(D, "ours", "silver_p_selected") - ex(D, "ours", "silver_p_bm25") < 0.15 for D in (R, L))),
        ("ev/foreign", "evidence scores separate own from foreign sentences (AUC > 0.8); < 20% of the selected sentences are foreign",
         lambda: all(ex(D, "ours", "auc_own_vs_foreign") > 0.8 and ex(D, "ours", "halfswap_foreign_sel") < 0.2 for D in (R, L))),
        ("ev/usable", "usability score: AUC > 0.95 against random-claim pools on both; against similar-claim pools clearly lower on LIAR-RAW",
         lambda: all(ex(D, "ours", "auc_usable") > 0.95 for D in (R, L)) and ex(L, "ours", "auc_usable_near") < ex(R, "ours", "auc_usable_near") - 0.05),
        ("ev/rsv", "read-select-verify above single-pass BM25 for ours on both datasets, neither significant",
         lambda: all(d(D, "ours_vs_ours@bm25") > 0 and not sig(D, "ours_vs_ours@bm25") for D in (R, L)) and seeds(L, "ours", 3)),
        ("ev/read", "averaging over the pool is on a par with read-select-verify (|diff| < 1)", lambda: all(abs(d(D, "ours_vs_ours@read")) < 1 for D in (R, L))),
        ("ev/lora-read", "LoRA+evidence: reading the whole pool changes less than 1 point", lambda: all(abs(d(D, "lora_ev@read_vs_own")) < 1 for D in (R, L))),
        ("ev/oracle", "selecting by silver labels brings no gain (|diff| < 1 for both verifiers on both datasets)",
         lambda: all(abs(d(D, k)) < 1 for D in (R, L) for k in ("ours_vs_ours@oracle", "lora_ev@oracle_vs_own"))),
    ]
    n_fail = 0
    for name, text, fn in claims:
        try:
            status = "ok  " if fn() else "FAIL"
        except NA as e:
            status = "n/a "
            text += "   [missing: {}]".format(e)
        n_fail += status == "FAIL"
        print("{} {:22s} {}".format(status, name, text))
    print("{} claims, {} FAIL".format(len(claims), n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
