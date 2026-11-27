#!/usr/bin/env python
"""results/ob/analysis.json -> tables, figures and number macros of the open-book chapter.

    python scripts/ob_assets.py            # writes thesis/gen/ob_tables, ob_figures, ob_numbers.tex

Nothing here computes a result; values that are not available yet are printed as "--".  The
proposed method is the entry ``ours`` of the analysis (see scripts/ob_analyze.py).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import make_assets as M  # noqa: E402  (formatting helpers shared with the closed-book chapter)

DS = ["RAWFC", "LIAR-RAW"]
SHORT = {"RAWFC": "R", "LIAR-RAW": "L"}
DS_LABEL = {"RAWFC": "RAWFC（三分类）", "LIAR-RAW": "LIAR-RAW（六分类）"}
NAME = {"zs_closed": "零样本 LLM", "zs_ev": "零样本 LLM", "slm_closed": "RoBERTa-large", "slm_ev": "RoBERTa-large",
        "lora_closed": "LoRA-LLM", "lora_ev": "LoRA-LLM", "ours": r"\ours{}（本章）"}
LONG = {"zs_closed": "零样本 LLM（无证据）", "zs_ev": "零样本 LLM + 证据", "slm_closed": "RoBERTa-large（无证据）",
        "slm_ev": "RoBERTa-large + 证据", "lora_closed": "LoRA-LLM（无证据）", "lora_ev": "LoRA-LLM + 证据", "ours": r"\ours{}",
        "noclaim": "LoRA-LLM（只读证据）"}
CONDS = [("test", "原始"), ("test@droptop", "缺少前排"), ("test@halfswap", "混入一半"), ("test@nearswap", "相近声明"),
         ("test@swap", "随机声明"), ("test@closed", "无证据")]
OUT = os.path.join(ROOT, "thesis", "gen")
SKIPPED = []


def get(d, path, default=None):
    for k in path.split("/"):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def num(x, d=2, scale=100.0):
    return "--" if x is None else "{:.{d}f}".format(scale * x, d=d)


def pm(msd, d=2, bold=False):
    if not msd:
        return "--"
    core = M.pct(msd["mean"], d)
    if bold:
        core = r"\textbf{" + core + "}"
    return core + (r"\pmstd{" + M.pct(msd["std"], d) + "}" if msd.get("n", 1) > 1 else "")


def table(name, fn, A):
    try:
        M.write(os.path.join(OUT, "ob_tables", name + ".tex"), fn(A))
    except (KeyError, TypeError) as e:
        SKIPPED.append("{}: {!r}".format(name, e))


def methods(A, D, protocol="strict"):
    return get(A, "{}/{}/methods".format(D, protocol), {})


def cmp_of(A, D, key, protocol="strict"):
    return get(A, "{}/{}/compare/{}".format(D, protocol, key))


def star(cmp):
    return "$^{*}$" if M.both_sig(cmp) else ""


def cond_value(A, D, key, cond):
    """macro-F1 of a method under a condition, on the claims the perturbed conditions use (on
    LIAR-RAW a fixed subset of the test claims), so that a row of a table is comparable."""
    if cond == "test":
        c = cmp_of(A, D, key + "@swap_vs_own")
        return c["b"] if c else get(methods(A, D), key + "/test/macf1/mean")
    c = cmp_of(A, D, "{}@{}_vs_own".format(key, cond.split("@")[1]))
    return c["a"] if c else get(methods(A, D), "{}/{}/macf1/mean".format(key, cond))


# ------------------------------------------------------------------------------------------ tables
def tab_data(A):
    out = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r" & \multicolumn{3}{c}{RAWFC} & \multicolumn{3}{c}{LIAR-RAW} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r" & 训练 & 验证 & 测试 & 训练 & 验证 & 测试 \\", r"\midrule"]
    rows = [("声明数", lambda s: "{:,}".format(s["claims"])),
            ("每条声明的报道数", lambda s: "{:.1f}".format(s["reports"] / s["claims"])),
            ("每条声明的候选句数（全部报道）", lambda s: "{:.1f}".format(s["pool_all"]["mean"])),
            ("每条声明的候选句数（严格协议）", lambda s: "{:.1f}".format(s["pool_strict"]["mean"])),
            ("候选句中银标证据的占比（\\%）", lambda s: "{:.1f}".format(100 * s["pool_all"]["silver_rate"])),
            ("声明的平均词数", lambda s: "{:.1f}".format(s["claim_words"])),
            ("核查结论文本的平均词数", lambda s: "{:.0f}".format(s["explain_words"]))]
    for label, fn in rows:
        out.append(label + " & " + " & ".join(fn(A[D]["stats"][sp]) for D in DS for sp in ("train", "val", "test")) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


FLAG_ZH = [("source", "标签来源的核查机构"), ("checker", "其他事实核查网站"), ("dump", "代码与数据托管站点"),
           ("fc_url", "新闻媒体的核查栏目"), ("fc_text", "正文提及事实核查")]


def tab_audit(A):
    out = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r" & \multicolumn{3}{c}{RAWFC} & \multicolumn{3}{c}{LIAR-RAW} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"被标记的原因 & 报道数 & 涉及声明 & 测试集声明 & 报道数 & 涉及声明 & 测试集声明 \\", r"\midrule"]

    def cells(D, f):
        t, te = A[D]["audit"]["total"], A[D]["audit"]["test"]
        if f is None:
            return ["{:,}".format(t["reports_any"]), "{:,}".format(t["claims_any"]), "{:,}".format(te["claims_any"])]
        if f == "share":
            return ["{:.1f}".format(100 * t["reports_any_share"]), "{:.1f}".format(100 * t["claims_any_share"]),
                    "{:.1f}".format(100 * te["claims_any_share"])]
        return ["{:,}".format(t["reports_by_flag"][f]), "{:,}".format(t["claims_by_flag"][f]), "{:,}".format(te["claims_by_flag"][f])]

    for f, label in FLAG_ZH:
        out.append(label + " & " + " & ".join(c for D in DS for c in cells(D, f)) + r" \\")
    out.append(r"\midrule")
    out.append("合计（任一原因） & " + " & ".join(c for D in DS for c in cells(D, None)) + r" \\")
    out.append("占总数的比例（\\%） & " + " & ".join(c for D in DS for c in cells(D, "share")) + r" \\")
    out.append("报道 / 声明总数 & " + " & ".join(
        c for D in DS for c in ["{:,}".format(A[D]["audit"]["total"]["reports"]), "{:,}".format(A[D]["audit"]["total"]["claims"]),
                                "{:,}".format(A[D]["audit"]["test"]["claims"])]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


MAIN_ROWS = [("zs_closed", "无"), ("zs_ev", "BM25 前 16 句"), ("slm_closed", "无"), ("slm_ev", "BM25 前 16 句"),
             ("lora_closed", "无"), ("lora_ev", "BM25 前 16 句"), ("ours", "读—选—判")]


def tab_main(A):
    """Our implementations under the strict protocol."""
    out = [r"\begin{tabular}{llcccccccc}", r"\toprule",
           r" & & \multicolumn{4}{c}{" + DS_LABEL["RAWFC"] + r"} & \multicolumn{4}{c}{" + DS_LABEL["LIAR-RAW"] + r"} \\",
           r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
           r"方法 & 证据 & P & R & F1 & Acc & P & R & F1 & Acc \\", r"\midrule"]
    best = {}
    for D in DS:
        vals = {k: get(methods(A, D), k + "/test/macf1/mean") for k, _ in MAIN_ROWS}
        vals = {k: v for k, v in vals.items() if v is not None}
        best[D] = max(vals, key=vals.get) if vals else None
    for i, (key, ev) in enumerate(MAIN_ROWS):
        cells = []
        for D in DS:
            m = methods(A, D).get(key)
            if not m:
                cells += ["--"] * 4
            else:
                t = m["test"]
                cells += [M.pct(t["macp"]["mean"]), M.pct(t["macr"]["mean"]), pm(t["macf1"], bold=best[D] == key), M.pct(t["acc"]["mean"])]
        if key == "ours":
            out.append(r"\midrule")
        out.append("{} & {} & {} \\\\".format(NAME[key], ev, " & ".join(cells)))
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_published(A):
    """Numbers other papers report for their own methods (all reports), next to ours."""
    out = [r"\begin{tabular}{llcccccc}", r"\toprule",
           r" & & \multicolumn{3}{c}{RAWFC} & \multicolumn{3}{c}{LIAR-RAW} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
           r"方法 & 骨干模型 & P & R & F1 & P & R & F1 \\", r"\midrule",
           r"\multicolumn{8}{l}{\emph{已发表数值（各论文自行报告，使用数据集的全部原始报道）}} \\"]
    for p in A["published"]["methods"]:
        mark = ("$^{\\dagger}$" if p["f1"] == "H" else "") + ("$^{\\ddagger}$" if p["checked"] == "secondhand" else "")
        cells = []
        for k in ("rawfc", "liar"):
            v = p.get(k)
            cells += ["{:.2f}".format(x) if x is not None else "--" for x in v] if v else ["--"] * 3
        out.append("{}\\citep{{{}}}{} & {} & {} \\\\".format(p["name"], p["key"], mark, p["backbone"].replace("&", r"\&"), " & ".join(cells)))
    out.append(r"\midrule")
    out.append(r"\multicolumn{8}{l}{\emph{本章结果（多个随机种子的均值）}} \\")
    for key, label, protocol in (("lora_ev", "LoRA-LLM + 证据，严格协议", "strict"), ("ours", r"\ours{}，严格协议", "strict"),
                                 ("ours", r"\ours{}，全部报道协议", "all")):
        cells = []
        for D in DS:
            t = get(methods(A, D, protocol), key + "/test")
            cells += [M.pct(t["macp"]["mean"]), M.pct(t["macr"]["mean"]), pm(t["macf1"])] if t else ["--"] * 3
        out.append("{} & Qwen2.5-7B（LoRA） & {} \\\\".format(label, " & ".join(cells)))
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_other(A):
    """Published numbers obtained with other evidence sources: listed for reference only."""
    out = [r"\begin{tabular}{llp{4.6cm}cccc}", r"\toprule",
           r"方法 & 骨干模型 & 证据来源 & \multicolumn{2}{c}{RAWFC} & \multicolumn{2}{c}{LIAR / LIAR-RAW} \\",
           r" & & & F1 & 口径 & F1 & 口径 \\", r"\midrule"]
    zh = {"Google search; LIAR instead of LIAR-RAW": "在线搜索（Google）；LIAR 的声明",
          "Wikipedia; LIAR instead of LIAR-RAW": "维基百科；LIAR 的声明", "live web search": "在线搜索",
          "none at inference; trained with the gold explanations": "推理时无证据；以核查结论文本为训练目标",
          "the datasets' evidence sentences": "数据集的证据句"}
    for p in A["published"]["other_evidence"]:
        f = lambda v: "--" if not v or v[2] is None else "{:.2f}".format(v[2])   # noqa: E731
        kind = "调和" if p["f1"] == "H" else "宏平均"
        out.append("{}\\citep{{{}}} & {} & {} & {} & {} & {} & {} \\\\".format(
            p["name"].replace("LoRA on evidence sentences (Kumar et al.)", "LoRA 读证据句"), p["key"],
            p["backbone"].replace("&", r"\&"), zh.get(p["evidence"], p["evidence"]),
            f(p.get("rawfc")), kind if p.get("rawfc") else "--", f(p.get("liar")), kind if p.get("liar") else "--"))
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


SIG_ROWS = [("原始测试集", [("lora_ev_vs_lora_closed", "LoRA-LLM：有证据 $-$ 无证据"),
                         ("slm_ev_vs_slm_closed", "RoBERTa-large：有证据 $-$ 无证据"),
                         ("lora_ev_vs_slm_ev", "LoRA-LLM + 证据 $-$ RoBERTa-large + 证据"),
                         ("ours_vs_lora_ev", r"\ours{} $-$ LoRA-LLM + 证据"),
                         ("ours_vs_slm_ev", r"\ours{} $-$ RoBERTa-large + 证据"),
                         ("ours_vs_lora_closed", r"\ours{} $-$ LoRA-LLM（无证据）")]),
            ("证据来自相近声明", [("ours_vs_lora_ev@nearswap", r"\ours{} $-$ LoRA-LLM + 证据"),
                            ("ours@nearswap_vs_lora_closed", r"\ours{} $-$ LoRA-LLM（无证据，原始输入）")]),
            ("证据来自随机声明", [("ours_vs_lora_ev@swap", r"\ours{} $-$ LoRA-LLM + 证据"),
                            ("ours@swap_vs_lora_closed", r"\ours{} $-$ LoRA-LLM（无证据，原始输入）")])]


def sig_cells(c):
    if not c or "bootstrap" not in c:
        return ["--"] * 4
    b = c["bootstrap"]
    return [M.signed(100 * c["diff_mean"]), "[{}, {}]".format(M.signed(100 * b["ci_lo"]), M.signed(100 * b["ci_hi"])),
            M.fmt_p(b["p"]), M.fmt_p(c.get("welch_p", float("nan")))]


def tab_sig(A):
    out = [r"\begin{tabular}{lcccccccc}", r"\toprule",
           r" & \multicolumn{4}{c}{RAWFC} & \multicolumn{4}{c}{LIAR-RAW} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
           r"比较（前者减后者） & $\Delta$ & 95\% CI & $p_{\text{样本}}$ & $p_{\text{种子}}$ & $\Delta$ & 95\% CI & $p_{\text{样本}}$ & $p_{\text{种子}}$ \\"]
    for title, rows in SIG_ROWS:
        out += [r"\midrule", r"\multicolumn{9}{l}{\emph{" + title + r"}} \\"]
        for key, label in rows:
            out.append(label + " & " + " & ".join(c for D in DS for c in sig_cells(cmp_of(A, D, key))) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_robust(A):
    n = len(CONDS)
    out = [r"\begin{tabular}{l" + "c" * (2 * n) + "}", r"\toprule",
           r" & \multicolumn{%d}{c}{" % n + DS_LABEL["RAWFC"] + r"} & \multicolumn{%d}{c}{" % n + DS_LABEL["LIAR-RAW"] + r"} \\",
           r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (n + 1, n + 2, 2 * n + 1),
           "方法 & " + " & ".join(c for _ in DS for _, c in CONDS) + r" \\", r"\midrule"]
    for key in ("zs_ev", "slm_ev", "lora_ev", "ours"):
        cells = [num(cond_value(A, D, key, cond)) for D in DS for cond, _ in CONDS]
        out.append(LONG[key] + " & " + " & ".join(cells) + r" \\")
    out.append(r"\midrule")
    cells = []
    for D in DS:
        c = cmp_of(A, D, "ours@swap_vs_lora_closed") or cmp_of(A, D, "lora_ev@swap_vs_lora_closed")
        v = c["b"] if c else get(methods(A, D), "lora_closed/test/macf1/mean")
        cells += [num(v)] + ["--"] * (n - 2) + [num(v)]
    out.append(LONG["lora_closed"] + "（参照） & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_behaviour(A):
    """What the verifiers predict when the pool belongs to another claim."""
    rows = [("own_conf", "原始输入下的平均置信度", 2, 1.0), ("nearswap_conf", "证据来自相近声明时的平均置信度", 2, 1.0),
            ("swap_conf", "证据来自随机声明时的平均置信度", 2, 1.0),
            ("swap_low", "证据来自随机声明时，预测为最低等级的比例（\\%）", 1, 100.0),
            ("nearswap_acc_source_label", "证据来自相近声明时，预测等于\\emph{证据所属声明}标签的比例（\\%）", 1, 100.0),
            ("nearswap_same_label", "参照：相近声明与原声明标签相同的比例（\\%）", 1, 100.0)]
    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{RAWFC} & \multicolumn{2}{c}{LIAR-RAW} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r" & LoRA-LLM + 证据 & \ours{} & LoRA-LLM + 证据 & \ours{} \\", r"\midrule"]
    for key, label, d, scale in rows:
        cells = []
        for D in DS:
            for m in ("lora_ev", "ours"):
                if key == "swap_low":
                    v = get(methods(A, D), m + "/extra/swap_pred_share")
                    v = v[0] if v else None
                else:
                    v = get(methods(A, D), "{}/extra/{}/mean".format(m, key))
                cells.append(num(v, d, scale))
        out.append(label + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_evidence(A):
    rows = [("silver_p_pool", "银标证据占比：候选池（\\%）", 1, 100.0),
            ("silver_p_bm25", "银标证据占比：BM25 前 16 句（\\%）", 1, 100.0),
            ("silver_p_selected", "银标证据占比：模型选出的 16 句（\\%）", 1, 100.0),
            ("auc_silver", "证据得分区分银标证据句与其余候选句的 AUC", 3, 1.0),
            ("auc_own_vs_foreign", "混入一半时，证据得分区分本声明句子与外来句子的 AUC", 3, 1.0),
            ("halfswap_foreign_sel", "混入一半时，选出的句子中外来句的占比（\\%）", 1, 100.0),
            ("auc_usable_near", "可用性得分区分“本声明的证据 / 相近声明的证据”的 AUC", 3, 1.0),
            ("auc_usable", "可用性得分区分“本声明的证据 / 随机声明的证据”的 AUC", 3, 1.0)]
    out = [r"\begin{tabular}{lcc}", r"\toprule", " & RAWFC & LIAR-RAW \\\\", r"\midrule"]
    for key, label, d, scale in rows:
        cells = [num(get(methods(A, D), "ours/extra/{}/mean".format(key)), d, scale) for D in DS]
        out.append(label + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_infer(A):
    rows = [("lora_ev", "test", "LoRA-LLM + 证据", "单遍：BM25 前 16 句（默认）", "1"),
            ("lora_ev", "test@read", "", "读完候选池，取平均", "3"),
            ("lora_ev", "test@oracle", "", "上限：按银标证据选句$^{\\S}$", "1"),
            ("ours", "test@bm25", r"\ours{}", "单遍：BM25 前 16 句", "1"),
            ("ours", "test@read", "", "读完候选池，取平均", "3"),
            ("ours", "test", "", "读—选—判（默认）", "4"),
            ("ours", "test@avg", "", "读—选—判与“取平均”再平均", "4"),
            ("ours", "test@oracle", "", "上限：按银标证据选句$^{\\S}$", "1")]
    out = [r"\begin{tabular}{llccc}", r"\toprule", r"模型 & 推理方式 & 前向次数 & RAWFC & LIAR-RAW \\", r"\midrule"]
    for i, (key, cond, name, label, q) in enumerate(rows):
        if name and i:
            out.append(r"\midrule")
        out.append("{} & {} & {} & {} \\\\".format(name, label, q, " & ".join(pm(get(methods(A, D), "{}/{}/macf1".format(key, cond))) for D in DS)))
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_ablation(A):
    D = "RAWFC"
    ab = get(A, D + "/strict/ablation")
    if not ab:
        raise KeyError("ablation")
    expl = ab.get("explain_row")
    rows = [("ours", r"\ours{}（完整）"), ("noswap", "去掉错配证据训练"), ("nosample", "去掉训练证据的重采样"),
            ("nohead", "去掉证据打分头（单遍推理）"), ("lora_ev", "三者都去掉（LoRA-LLM + 证据）")]
    if expl == "sieve":
        rows.insert(4, ("sieve", "去掉解释目标"))
    elif expl == "sievej":
        rows.append(("sievej", "加上解释目标（未采用）"))
    rows.append(("swap30", "错配概率 $\\rho$ 由 0.15 提高到 0.30"))
    out = [r"\begin{tabular}{l" + "c" * len(CONDS) + "}", r"\toprule", "变体 & " + " & ".join(c for _, c in CONDS) + r" \\", r"\midrule"]
    for key, label in rows:
        m = ab["methods"].get(key)
        if not m:
            continue
        cells = [(pm(m.get(cond)) if m.get(cond) else "--") + (star(get(ab, "vs_full/{}/{}".format(key, cond))) if key != "ours" else "")
                 for cond, _ in CONDS]
        if key in ("sievej", "swap30") and not out[-1].endswith("midrule"):
            out.append(r"\midrule")
        out.append(label + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_explain(A):
    """The verifier with and without the explanation target (seeds common to both)."""
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{RAWFC} & \multicolumn{3}{c}{LIAR-RAW} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r" & 验证 & 测试 & 随机声明 & 验证 & 测试 & 随机声明 \\", r"\midrule"]
    if not any(get(A, D + "/strict/explain") for D in DS):
        raise KeyError("explain")
    for key, label in (("sieve", "不使用解释目标"), ("sievej", "使用解释目标")):
        cells = [pm(get(A, "{}/strict/explain/methods/{}/{}".format(D, key, c))) for D in DS for c in ("val", "test", "test@swap")]
        out.append(label + " & " + " & ".join(cells) + r" \\")
    out.append(r"\midrule")
    cells = []
    for D in DS:
        for c in ("val", "test", "test@swap"):
            x = get(A, "{}/strict/explain/compare/{}".format(D, c))
            cells.append((M.signed(100 * x["diff_mean"]) + star(x)) if x else "--")
    out.append("差值（使用 $-$ 不使用） & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_protocol(A):
    out = [r"\begin{tabular}{llcccccc}", r"\toprule",
           r" & & \multicolumn{2}{c}{全部测试声明（F1）} & \multicolumn{2}{c}{含被标记报道（Acc）} & \multicolumn{2}{c}{不含被标记报道（Acc）} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
           r"数据集 & 方法 & 严格 & 全部报道 & 严格 & 全部报道 & 严格 & 全部报道 \\", r"\midrule"]
    for D in DS:
        first = True
        for key in ("zs_ev", "slm_ev", "lora_ev", "ours", "noclaim"):
            m = methods(A, D).get(key)
            leak = get(A, "{}/strict/leak/methods/{}".format(D, key), {})
            if not m:
                continue
            cells = [num(get(m, "test/macf1/mean")), num(get(m, "test@all/macf1/mean")) + star(cmp_of(A, D, key + "@all_vs_own"))]
            for sub in ("any", "none"):
                cells += [num(get(leak, sub + "/own/acc")), num(get(leak, sub + "/other/acc")) + star(get(leak, sub + "/compare"))]
            out.append("{} & {} & {} \\\\".format(D if first else "", LONG.get(key, key), " & ".join(cells)))
            first = False
        if D != DS[-1]:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_cross(A):
    """The final method trained with all reports vs trained under the strict protocol (same seeds)."""
    out = [r"\begin{tabular}{llcccc}", r"\toprule",
           r" & & \multicolumn{2}{c}{严格协议下训练} & \multicolumn{2}{c}{全部报道下训练} \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r"数据集 & 测试声明 & 严格测试 & 全部报道测试 & 严格测试 & 全部报道测试 \\", r"\midrule"]
    any_row = False
    for D in DS:
        c = get(A, "{}/cross_protocol/ours".format(D))
        if not c:
            continue
        any_row = True
        x, y, z = c["all_all_vs_strict_strict"], c["all_all_vs_strict_all"], c["all_strict_vs_strict_strict"]
        out.append("{} & 全部（F1） & {} & {} & {} & {} \\\\".format(D, num(x["b"]), num(y["b"]), num(z["a"]), num(x["a"]) + star(x)))
        for key, label in (("any", "含被标记报道（F1）"), ("none", "不含被标记报道（F1）")):
            w = c.get("all_all_vs_strict_strict@" + key)
            if w:
                out.append(" & {} & {} & -- & -- & {} \\\\".format(label, num(w["b"]), num(w["a"]) + star(w)))
        if D != DS[-1]:
            out.append(r"\midrule")
    if not any_row:
        raise KeyError("cross_protocol")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


GATE_CONDS = [("test", "原始"), ("test@halfswap", "混入一半"), ("test@nearswap", "相近声明"), ("test@swap", "随机声明")]


def tab_gate(A, q="0.05"):
    n = len(GATE_CONDS)
    out = [r"\begin{tabular}{l" + "c" * (2 * n) + "}", r"\toprule",
           r" & \multicolumn{%d}{c}{RAWFC} & \multicolumn{%d}{c}{LIAR-RAW} \\" % (n, n),
           r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (n + 1, n + 2, 2 * n + 1),
           " & " + " & ".join(c for _ in DS for _, c in GATE_CONDS) + r" \\", r"\midrule"]
    if not any(get(A, "{}/strict/gate/q/{}".format(D, q)) for D in DS):
        raise KeyError("gate")
    for key, label, kind in (("plain", r"\ours{}", "f1"), ("gated", r"\ours{} + 回退", "f1"), ("closed", "LoRA-LLM（无证据）", "f1"),
                             ("flagged", "被回退的声明比例（\\%）", "share")):
        cells = []
        for D in DS:
            for cond, _ in GATE_CONDS:
                r = get(A, "{}/strict/gate/q/{}/{}".format(D, q, cond))
                if not r:
                    cells.append("--")
                elif kind == "share":
                    cells.append(num(r[key]["mean"], 1))
                else:
                    cells.append(num(r[key]["mean"]) + (star(r.get("vs_plain")) if key == "gated" else ""))
        if key == "flagged":
            out.append(r"\midrule")
        out.append(label + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_ensemble(A):
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{RAWFC} & \multicolumn{3}{c}{LIAR-RAW} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"方法 & 种子数 & 单模型均值 & 集成 & 种子数 & 单模型均值 & 集成 \\", r"\midrule"]
    for key in ("slm_ev", "lora_closed", "lora_ev", "ours"):
        cells = []
        for D in DS:
            e = get(A, "{}/strict/ensemble/methods/{}".format(D, key))
            single = get(methods(A, D), key + "/test/macf1/mean")
            cells += ["{}".format(e["n_seeds"]), num(single), num(e["macf1"])] if e else ["--", num(single), "--"]
        out.append(LONG[key] + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_transfer(A):
    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{RAWFC $\rightarrow$ LIAR-RAW} & \multicolumn{2}{c}{LIAR-RAW $\rightarrow$ RAWFC} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"方法 & macro-F1 & Acc & macro-F1 & Acc \\", r"\midrule"]
    for key in ("lora_ev", "ours"):
        cells = []
        for D in DS:
            t = get(A, "{}/transfer/{}".format(D, key), {})
            cells += [pm(t.get("macf1")), M.pct(t["acc"]["mean"]) if t.get("acc") else "--"]
        out.append(LONG[key] + " & " + " & ".join(cells) + r" \\")
    out.append(r"\midrule")
    for key in ("lora_ev", "ours"):
        ref = [pm(get(A, "LIAR-RAW/transfer/{}/in_domain_three/macf1".format(key))),
               num(get(A, "LIAR-RAW/transfer/{}/in_domain_three/acc/mean".format(key))),
               pm(get(methods(A, "RAWFC"), key + "/test/macf1")), num(get(methods(A, "RAWFC"), key + "/test/acc/mean"))]
        out.append("参照：{}，在目标数据集上训练 & ".format(LONG[key]) + " & ".join(ref) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_llm2(A):
    D = "RAWFC"
    L2 = get(A, D + "/llm2")
    if not L2:
        raise KeyError("llm2")
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{Qwen2.5-7B} & \multicolumn{3}{c}{Llama-2-7B} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"方法 & 原始 & 相近声明 & 随机声明 & 原始 & 相近声明 & 随机声明 \\", r"\midrule"]
    seeds = get(L2, "methods/ours/seeds") or get(L2, "methods/lora_ev/seeds") or []
    for key in ("zs_closed", "zs_ev", "lora_closed", "lora_ev", "ours"):
        cells = []
        for ms_ in (methods(A, D), L2["methods"]):
            m = ms_.get(key)
            cells += [pm(get(m, c + "/macf1")) if m else "--" for c in ("test", "test@nearswap", "test@swap")]
        out.append(LONG[key] + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_cost(A):
    out = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r" & & \multicolumn{2}{c}{训练时间（分钟）} & \multicolumn{2}{c}{推理（每条声明）} & \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r"方法 & 可训练参数（M） & RAWFC & LIAR-RAW & 前向次数 & 输入 token 数 & 显存峰值（GB） \\", r"\midrule"]
    for key in ("slm_ev", "lora_closed", "lora_ev", "ours"):
        ms = [methods(A, D).get(key) for D in DS]
        m0 = next((m for m in ms if m), None)
        if not m0:
            continue
        cost = m0.get("cost", {})
        out.append("{} & {:.1f} & {} & {} & {} & {} & {} \\\\".format(
            LONG.get(key, NAME.get(key, key)), m0["n_trainable"] / 1e6,
            *["{:.0f}".format(m["train_seconds"] / 60) if m else "--" for m in ms],
            "{:.0f}".format(cost["queries"]) if cost else "1", "{:,.0f}".format(cost["tokens"]) if cost else "--",
            "{:.1f}".format(m0["max_memory_gb"]) if m0.get("max_memory_gb") else "--"))
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_slm_grid(A):
    out = [r"\begin{tabular}{llcccc}", r"\toprule",
           r" & & \multicolumn{2}{c}{RAWFC} & \multicolumn{2}{c}{LIAR-RAW} \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r"输入 & 学习率 & 验证 & 测试 & 验证 & 测试 \\", r"\midrule"]
    for kind, label in (("closed", "仅声明"), ("bm25", "声明 + 证据")):
        for lr in ("1e-5", "2e-5"):
            name = "roberta-large-{}@lr{}".format(kind, lr)
            cells = []
            for D in DS:
                g = get(A, "{}/strict/grid".format(D), {}).get(name)
                chosen = name in get(A, "{}/strict/names".format(D), {}).values()
                cells += [pm(g["val"], bold=chosen) if g else "--", pm(g["test"]) if g else "--"]
            out.append("{} & ${}\\times10^{{-5}}$ & {} \\\\".format(label, lr[0], " & ".join(cells)))
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_cases(A):
    """Worked examples (English data shown verbatim): claim, the best-scored evidence, the verdict."""
    out = []
    n = 0
    for D in DS:
        names = [M.esc(x) for x in get(A, D + "/label_text", [])]
        for c in get(A, D + "/cases", []):
            n += 1
            lab = lambda i: names[i] if names else str(i)   # noqa: E731
            out.append(r"\noindent\textbf{样例 %d}（%s；标签：%s；预测：%s，概率 %.2f）\par" % (
                n, D, lab(c["label"]), lab(c["pred"]), max(c["prob"])))
            out.append(r"{\small\begin{quote}\textbf{声明：}" + M.esc(c["claim"]) + r"\end{quote}}")
            out.append(r"{\small\begin{tabularx}{\linewidth}{@{}ccX@{}}")
            out.append(r"\toprule")
            out.append(r"得分 & 银标 & 证据句（来源） \\")
            out.append(r"\midrule")
            for e in c["evidence"]:
                out.append("%.2f & %s & %s（%s） \\\\" % (e["score"], "是" if e["silver"] else "否", M.esc(e["text"]), M.esc(e["host"] or "未知来源")))
            out.append(r"\bottomrule")
            out.append(r"\end{tabularx}}")
            if c.get("justification"):
                out.append(r"{\small\begin{quote}\textbf{模型生成的说明：}" + M.esc(c["justification"]) + r"\end{quote}}")
            out.append(r"\par\medskip")
    if not n:
        raise KeyError("cases")
    return "\n".join(out) + "\n"


def write_prompt():
    from openbook.prompts import verifier_prompt

    msgs = verifier_prompt("〈待核查的声明〉", [("来源域名 1", "〈证据句 1〉"), ("来源域名 2", "〈证据句 2〉")], ["false", "half true", "true"])
    text = msgs[0]["content"] + "\n\n" + msgs[1]["content"].replace("[2] (来源域名 2) 〈证据句 2〉", "[2] (来源域名 2) 〈证据句 2〉\n……")
    M.write(os.path.join(OUT, "ob_tables", "prompt_verifier.tex"), M.prompt_block(text))


# ----------------------------------------------------------------------------------------- figures
def fig_robust(A, path):
    """Macro-F1 as the evidence gets worse, per dataset; the closed-book verifier as reference."""
    import matplotlib.pyplot as plt

    from clara import plotting as P

    P.setup()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.75))
    conds = [("test", "原始"), ("test@droptop", "缺少\n前排"), ("test@halfswap", "混入\n一半"), ("test@nearswap", "相近\n声明"),
             ("test@swap", "随机\n声明")]
    series = [("ours", 0, "SIEVE"), ("lora_ev", 1, "LoRA-LLM + 证据"), ("slm_ev", 2, "RoBERTa-large + 证据"), ("zs_ev", 3, "零样本 LLM + 证据")]
    handles, drawn = {}, False
    for ax, D in zip(axes, DS):
        for key, slot, label in series:
            m = methods(A, D).get(key)
            if not m or any(c not in m for c, _ in conds):
                continue
            ys = [100 * cond_value(A, D, key, c) for c, _ in conds]
            es = [100 * m[c]["macf1"]["std"] for c, _ in conds]
            handles[label] = P.line(ax, range(len(conds)), ys, slot, label, yerr=es)
            drawn = True
        c = cmp_of(A, D, "ours@swap_vs_lora_closed") or cmp_of(A, D, "lora_ev@swap_vs_lora_closed")
        ref = c["b"] if c else get(methods(A, D), "lora_closed/test/macf1/mean")
        if ref is not None:
            handles["LoRA-LLM（无证据）"] = ax.axhline(100 * ref, color=P.NEUTRAL, linewidth=1.2, zorder=2)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c for _, c in conds])
        ax.set_xlim(-0.3, len(conds) - 0.7)
        ax.set_title(DS_LABEL[D], loc="left")
        ax.margins(y=0.12)
    axes[0].set_ylabel("macro-F1（%）")
    if not drawn:
        raise KeyError("no method with all perturbed conditions")
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.17), handlelength=1.8, labelcolor=P.INK2, frameon=False, columnspacing=1.3)
    fig.subplots_adjust(wspace=0.28)
    P.save(fig, path)


def fig_usable(A, path):
    """Distribution of the claim-level usability score when the pool is the claim's own, belongs
    to a similar claim, or to a random claim."""
    import matplotlib.pyplot as plt
    import numpy as np

    from clara import plotting as P

    P.setup()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), sharey=True)
    series = [("test", 0, "本声明的证据"), ("test@nearswap", 1, "相近声明的证据"), ("test@swap", 2, "随机声明的证据")]
    handles, drawn = {}, False
    for ax, D in zip(axes, DS):
        h = get(A, D + "/strict/gate/hist")
        if not h:
            continue
        edges = np.array(h["edges"])
        for cond, slot, label in series:
            share = h["share"].get(cond)
            if share is None:
                continue
            y = 100 * np.array(share)
            handles[label] = ax.stairs(y, edges, color=P.SLOTS[slot], linewidth=2.0, fill=False, zorder=3)
            ax.stairs(y, edges, color=P.SLOTS[slot], alpha=0.10, fill=True, linewidth=0, zorder=2)
            drawn = True
        ax.set_xlim(0, 1)
        ax.set_xlabel("证据可用性得分")
        ax.set_title(D, loc="left")
    axes[0].set_ylabel("声明占比（%）")
    if not drawn:
        raise KeyError("no usability histograms")
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1),
               handlelength=1.8, labelcolor=P.INK2, frameon=False, columnspacing=1.5)
    fig.subplots_adjust(wspace=0.12)
    P.save(fig, path)


def figure(name, fn, A):
    path = os.path.join(OUT, "ob_figures", name + ".pdf")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fn(A, path)
        print("wrote", os.path.relpath(path, ROOT))
    except (KeyError, TypeError) as e:
        SKIPPED.append("{}: {!r}".format(name, e))


# ----------------------------------------------------------------------------------------- numbers
def write_numbers(A):
    """Every numeric leaf as the macros of the closed-book chapter: \\ptwo{R/strict/methods/...} for
    RAWFC, \\ptwo{L/...} for LIAR-RAW, \\ptwo{D/...} for the validation-only decision record."""
    lines = ["% generated by scripts/ob_assets.py from results/ob/analysis.json - do not edit", "\\makeatletter"]
    n = 0
    trees = [(SHORT[D], A[D]) for D in DS] + ([("D", A["decision"])] if "decision" in A else [])
    for short, tree in trees:
        for key, v in M._leaves(tree, [short]):
            if any(ch in M._BAD for ch in key) or v != v or abs(v) == float("inf"):
                continue
            forms = {"pa": "{:.1f}".format(100 * v), "pb": "{:.2f}".format(100 * v), "da": M.signed(100 * v, 1),
                     "db": M.signed(100 * v, 2), "pe": "={:.3f}".format(v) if v >= 0.001 else "<0.001",
                     "fc": "{:.3f}".format(v), "fb": "{:.2f}".format(v), "fa": "{:.1f}".format(v), "ct": "{:,.0f}".format(v)}
            for tag, text in forms.items():
                lines.append("\\expandafter\\def\\csname num@{}@{}\\endcsname{{{}}}".format(tag, key, text))
            n += 1
    lines.append("\\makeatother")
    M.write(os.path.join(OUT, "ob_numbers.tex"), "\n".join(lines) + "\n")
    print("   {} numbers".format(n))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=os.path.join(ROOT, "results", "ob", "analysis.json"))
    ap.add_argument("--out", default=OUT, help="output directory (default: thesis/gen)")
    args = ap.parse_args()
    OUT = args.out
    with open(args.analysis, encoding="utf-8") as f:
        A = json.load(f)
    from openbook.data import LABEL_TEXT
    for D in DS:
        A[D]["label_text"] = LABEL_TEXT[D]
    for name, fn in [("data", tab_data), ("audit", tab_audit), ("main", tab_main), ("published", tab_published), ("other", tab_other),
                     ("sig", tab_sig), ("robust", tab_robust), ("behaviour", tab_behaviour), ("evidence", tab_evidence),
                     ("infer", tab_infer), ("ablation", tab_ablation), ("explain", tab_explain), ("protocol", tab_protocol),
                     ("cross", tab_cross), ("gate", tab_gate), ("ensemble", tab_ensemble), ("transfer", tab_transfer),
                     ("llm2", tab_llm2), ("cost", tab_cost), ("slm_grid", tab_slm_grid), ("cases", tab_cases)]:
        table(name, fn, A)
    write_prompt()
    figure("robust", fig_robust, A)
    figure("usable", fig_usable, A)
    for D in DS:
        A[D].pop("label_text", None)
    write_numbers(A)
    for s in SKIPPED:
        print("SKIPPED", s)


if __name__ == "__main__":
    main()
