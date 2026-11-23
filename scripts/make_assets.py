#!/usr/bin/env python
"""results/analysis.json -> LaTeX tables (paper/tables), figures (paper/figures), paper/numbers.tex.

Nothing here computes a result: every value is read from analysis.json.  Sections that are not
available yet are skipped with a message.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

DS = ["ARG-CN", "ARG-EN"]
DS_LABEL = {"ARG-CN": "Weibo21（中文）", "ARG-EN": "GossipCop（英文）"}
SHORT = {"ARG-CN": "CN", "ARG-EN": "EN"}
NAME = {"bert": "BERT", "lora": "LoRA-LLM", "bert_zs": "BERT+ZS", "arg_gpt": "ARG（GPT-3.5 理由）",
        "arg": "ARG", "factguard": "FactGuard", "clara": r"\method{}", "arg_d": "ARG-D",
        "factguard_d": "FactGuard-D", "clara_d": r"\student{}", "arg_swap": "ARG + 错配负例",
        "bert_claims": "BERT（只读 LLM 抽取的声明）", "bert_tc": "BERT（只读 topic-content）",
        "clara-no_relevance": r"\method{} 去掉相关性门控"}
SKIPPED = []


def both_sig(cmp, level=0.05):
    """A difference is called significant only if the paired bootstrap over test items AND the
    Welch test over seeds agree at ``level``."""
    if not cmp or "bootstrap" not in cmp:
        return False
    w = cmp.get("welch_p", float("nan"))
    return cmp["bootstrap"]["p"] < level and w == w and w < level


def fmt_p(x):
    if x != x:
        return "--"
    return "{:.3f}".format(x) if x >= 0.001 else "$<$0.001"


def pct(x, d=2):
    return "{:.{d}f}".format(100 * x, d=d)


def pm(ms, d=2, bold=False):
    core = pct(ms["mean"], d)
    if bold:
        core = r"\textbf{" + core + "}"
    return core + r"\pmstd{" + pct(ms["std"], d) + "}"


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("wrote", os.path.relpath(path, ROOT))


def table(name, fn, A):
    try:
        write(os.path.join(ROOT, "paper", "tables", name + ".tex"), fn(A))
    except KeyError as e:
        SKIPPED.append("{}: missing {}".format(name, e))


# ------------------------------------------------------------------------------------------ tables
def tab_data(A):
    rows = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r" & \multicolumn{3}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{3}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r" & 训练 & 验证 & 测试 & 训练 & 验证 & 测试 \\", r"\midrule"]

    def line(label, fn):
        return label + " & " + " & ".join(fn(A[D]["data"][sp]) for D in DS for sp in ("train", "val", "test")) + r" \\"

    rows.append(line("真新闻数", lambda d: "{:,}".format(d["n"] - d["fake"])))
    rows.append(line("假新闻数", lambda d: "{:,}".format(d["fake"])))
    rows.append(line("合计", lambda d: "{:,}".format(d["n"])))
    rows.append(line("假新闻占比（\\%）", lambda d: "{:.1f}".format(100 * d["fake"] / d["n"])))
    rows.append(line("平均字符数", lambda d: "{:.0f}".format(d["chars"])))
    rows.append(line("平均句数", lambda d: "{:.2f}".format(d["sents"])))
    rows.append(line("平均声明数 $K$", lambda d: "{:.2f}".format(d["claims"])))
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


MAIN_GROUPS = [
    ("仅 LLM，不训练", [("llm", "GPT-3.5 CS (released)", "GPT-3.5，常识视角$^\\dagger$"),
                     ("llm", "zero-shot", "Qwen2.5-7B，零样本"),
                     ("llm", "zero-shot CoT", "Qwen2.5-7B，零样本 CoT")]),
    ("仅 SLM", [("run", "bert", None)]),
    ("微调 LLM", [("run", "lora", None)]),
    ("LLM + SLM", [("run", "bert_zs", None), ("run", "arg_gpt", "ARG（GPT-3.5 理由）$^\\dagger$"),
                   ("run", "arg", None), ("run", "factguard", None), ("run", "clara", None)]),
    ("学生模型（推理时不调用 LLM）", [("run", "arg_d", None), ("run", "factguard_d", None), ("run", "clara_d", None)]),
]
MAIN_COLS = ["macf1", "acc", "f1_real", "f1_fake"]


def tab_main(A):
    shown = [key for _, rows in MAIN_GROUPS for kind, key, _ in rows if kind == "run"]
    best = {}
    for D in DS:
        for c in MAIN_COLS:
            best[(D, c)] = max(A[D]["main"][k][c] for k in shown if k in A[D]["main"])
    out = [r"\begin{tabular}{l" + "cccc" * 2 + "}", r"\toprule",
           r"\multirow{2}{*}{方法} & \multicolumn{4}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{4}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
           r" & macF1 & Acc & F1$_{\text{真}}$ & F1$_{\text{假}}$ & macF1 & Acc & F1$_{\text{真}}$ & F1$_{\text{假}}$ \\"]
    for title, rows in MAIN_GROUPS:
        body = []
        for kind, key, label in rows:
            cells = []
            for D in DS:
                if kind == "llm":
                    m = A[D]["llm_only"].get(key)
                    cells += [pct(m["test"][c]) for c in MAIN_COLS] if m else ["--"] * 4
                else:
                    m = A[D]["main"].get(key)
                    if m is None:
                        cells += ["--"] * 4
                    else:
                        cells += [pm({"mean": m[c], "std": m[c + "_std"]}, bold=abs(m[c] - best[(D, c)]) < 1e-12) for c in MAIN_COLS]
            if all(c == "--" for c in cells):
                continue
            body.append((label or NAME[key]) + " & " + " & ".join(cells) + r" \\")
        if body:
            out += [r"\midrule", r"\multicolumn{9}{l}{\emph{" + title + r"}} \\"] + body
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


ABL = [("full", r"\method{}（完整）"), None,
       ("no_claims", "去掉声明，仅整体实例"), ("no_overall", "去掉整体实例，仅声明"), None,
       ("no_args", "去掉论证文本，仅评分"), ("no_v", "去掉评分，仅论证文本"),
       ("fi_score", "论证前评分取提示内第一印象（严格单次调用）"), None,
       ("no_relevance", "去掉相关性门控与错配负例"),
       ("no_gate", "去掉判定门控（$g\\equiv1$）"), ("item_gate", "新闻级判定门控（一条新闻共用一个 $g$）"), None,
       ("agg_attention", "注意力聚合替代 noisy-OR"), ("no_global", "去掉整体文本分支"),
       ("entangled", "实例证据可见整条新闻（耦合式）")]


def tab_ablation(A):
    if not any("ablation" in A[D] for D in DS):
        raise KeyError("ablation")
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r"\multirow{2}{*}{变体} & \multicolumn{3}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{3}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r" & 原始 & 整条替换 & 单元替换 & 原始 & 整条替换 & 单元替换 \\", r"\midrule"]
    for row in ABL:
        if row is None:
            if out[-1] != r"\midrule":
                out.append(r"\midrule")
            continue
        key, label = row
        cells = []
        for D in DS:
            r = A[D].get("ablation", {}).get(key)
            if r is None:
                cells += ["--"] * 3
                continue
            cells.append(pm(r["test"]) + ("$^{*}$" if both_sig(r.get("vs_full")) else ""))
            for key in ("noise100", "unit100"):
                if key not in r:
                    cells.append("--")
                    continue
                cells.append(pm(r[key]) + ("$^{*}$" if both_sig(r.get("vs_full_" + key)) else ""))
        if any(c != "--" for c in cells):
            out.append(label + " & " + " & ".join(cells) + r" \\")
    if out[-1] == r"\midrule":
        out.pop()
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_gate(A):
    """Relevance gate: separation of genuine from foreign analysis units."""
    f3 = lambda x: "{:.3f}".format(x)
    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{2}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r" & 整条替换 & 单元替换 & 整条替换 & 单元替换 \\", r"\midrule"]
    R = {D: A[D]["gate"]["relevance"] for D in DS}
    spec = [("AUC：相关性门控 $r$ $\\rightarrow$ 是否为本新闻的分析", "auc_r"),
            ("AUC：判定门控 $g$ $\\rightarrow$ 同上", "auc_g"),
            ("本新闻的分析单元上的平均 $r$", "r_genuine"), ("被替换的分析单元上的平均 $r$", "r_foreign")]
    for label, key in spec:
        out.append(label + " & " + " & ".join(f3(R[D]["{}/{}".format(lvl, key)]) for D in DS for lvl in ("item", "unit")) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_gate_cells(A):
    """What does the verdict gate follow: the correctness of the LLM or the label of the news?"""
    f3 = lambda x: "{:.3f}".format(x)

    def g(D, k):
        return A[D]["gate"]["clara"][k]["mean"]

    out = [r"\begin{tabular}{llcccc}", r"\toprule",
           r" & & \multicolumn{2}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{2}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r"LLM 的整体判断 & 新闻标签 & 占比（\%） & 平均 $g_0$ & 占比（\%） & 平均 $g_0$ \\", r"\midrule"]
    cells = [("sayF_y1", "判假", "假（LLM 对）"), ("sayF_y0", "判假", "真（LLM 错）"),
             ("sayR_y1", "判真", "假（LLM 错）"), ("sayR_y0", "判真", "真（LLM 对）")]
    tot = {D: sum(g(D, "cells/n_" + c[0]) for c in cells) for D in DS}
    for key, say, lab in cells:
        out.append("{} & {} & ".format(say, lab) + " & ".join(
            "{:.1f} & {}".format(100 * g(D, "cells/n_" + key) / tot[D], f3(g(D, "cells/" + key))) for D in DS) + r" \\")
    out.append(r"\midrule")
    for label, key in [("AUC：$g_0$ $\\rightarrow$ LLM 的整体判断是否正确", "overall/auc_gate"),
                       ("AUC：$g_0$ $\\rightarrow$ 新闻是否为假", "overall/auc_gate_for_label"),
                       ("AUC：LLM 自身置信度 $\\rightarrow$ 判断是否正确", "overall/auc_llm_conf"),
                       ("AUC：文本分支与 LLM 是否一致 $\\rightarrow$ 判断是否正确", "overall/auc_text_agreement")]:
        out.append(r"\multicolumn{2}{l}{" + label + "} & " + " & ".join(
            r"\multicolumn{2}{c}{" + (f3(g(D, key)) if key in A[D]["gate"]["clara"] else "--") + "}" for D in DS) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_noise(A):
    """macro-F1 under replacement of the LLM analysis, as a table (the figure shows the curves)."""
    rows = [("bert", "BERT（不使用 LLM）"), ("bert_zs", "BERT+ZS"), ("factguard", "FactGuard"), ("arg", "ARG"),
            ("arg_swap", "ARG + 错配负例"), ("clara-no_relevance", r"\method{} 去掉相关性门控"), ("clara", r"\method{}")]
    cols = [("item", "0", "原始"), ("item", "40", "40\\%"), ("item", "100", "100\\%"), ("unit", "50", "50\\%"), ("unit", "100", "100\\%")]
    out = [r"\begin{tabular}{l" + "ccccc" * 2 + "}", r"\toprule",
           r" & \multicolumn{5}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{5}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
           r" & & \multicolumn{2}{c}{整条替换} & \multicolumn{2}{c}{单元替换} & & \multicolumn{2}{c}{整条替换} & \multicolumn{2}{c}{单元替换} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
           "方法 & " + " & ".join(c[2] for c in cols) + " & " + " & ".join(c[2] for c in cols) + r" \\", r"\midrule"]
    for key, label in rows:
        cells = []
        for D in DS:
            N = A[D]["noise"].get(key)
            for lvl, rate, _ in cols:
                if N is None:
                    cells.append("--")
                elif key == "bert":
                    cells.append(pct(N["item"]["0"]["mean"]) if rate == "0" else "$=$")
                else:
                    v = N[lvl].get(rate)
                    cells.append(pct(v["mean"]) if v else "--")
        if any(c != "--" for c in cells):
            out.append(label + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_useful(A):
    """ARG's rationale usefulness predictor against the correctness of the LLM judgment."""
    f3 = lambda x: "{:.3f}".format(x)
    out = [r"\begin{tabular}{llcccccc}", r"\toprule",
           r" & & \multicolumn{3}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{3}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
           r"模型 & 理由 & 可用性预测 & 融合权重 & 自身预测 & 可用性预测 & 融合权重 & 自身预测 \\", r"\midrule"]
    for base in ("arg", "arg_swap"):
        for k, kn in (("td", "文字描述"), ("cs", "常识")):
            cells = []
            for D in DS:
                U = A[D]["gate"]["usefulness"].get(base)
                cells += [f3(U[k + "/" + c]) if U and k + "/" + c in U else "--"
                          for c in ("auc_useful", "auc_weight", "auc_own_agreement")]
            if any(c != "--" for c in cells):
                out.append("{} & {} & ".format(NAME[base] if k == "td" else "", kn) + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_style_llm(A):
    f1 = lambda x: "{:.1f}".format(100 * x)
    f2 = lambda x: "{:.2f}".format(x)
    rows = [("zero-shot P(fake)", "零样本判断 $p_{\\mathrm{zs}}$"), ("overall verdict", "整体判断的怀疑度 $v_0$"),
            ("max claim suspicion", "声明怀疑度的最大值"), ("mean claim suspicion", "声明怀疑度的均值")]
    out = [r"\begin{tabular}{lcccccccc}", r"\toprule",
           r" & \multicolumn{4}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{4}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
           r" & \multicolumn{2}{c}{AUC（\%）} & \multicolumn{2}{c}{均值差（假$-$真）} & \multicolumn{2}{c}{AUC（\%）} & \multicolumn{2}{c}{均值差（假$-$真）} \\",
           r"LLM 评分 & 原文 & 改写 & 原文 & 改写 & 原文 & 改写 & 原文 & 改写 \\", r"\midrule"]
    for key, label in rows:
        cells = []
        for D in DS:
            r = A[D]["style_llm"][key]
            cells += [f1(r["auc_orig"]), f1(r["auc_style"]), f2(r["fake_orig"] - r["real_orig"]), f2(r["fake_style"] - r["real_style"])]
        out.append(label + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_style(A):
    order = ["bert", "lora", "bert_zs", "arg", "factguard", "clara", "clara_d", "bert_tc", "bert_claims"]
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{3}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"方法 & 原文 & 改写后 & $\Delta$ & 原文 & 改写后 & $\Delta$ \\", r"\midrule"]
    best = {D: max(A[D]["style"][b]["style"]["mean"] for b in order if b in A[D]["style"]) for D in DS}
    for b in order:
        cells = []
        for D in DS:
            r = A[D]["style"].get(b)
            if r is None:
                cells += ["--"] * 3
            else:
                cells += [pm(r["orig"]), pm(r["style"], bold=abs(r["style"]["mean"] - best[D]) < 1e-12),
                          "{:+.2f}".format(100 * r["delta"]["mean"])]
        if any(c != "--" for c in cells):
            if b == "bert_tc":
                out.append(r"\midrule")
            out.append(NAME[b] + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_style_recall(A):
    order = ["bert", "lora", "bert_zs", "arg", "factguard", "clara"]
    out = [r"\begin{tabular}{lcccccccc}", r"\toprule",
           r" & \multicolumn{4}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{4}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
           r" & \multicolumn{2}{c}{假新闻召回} & \multicolumn{2}{c}{真新闻召回} & \multicolumn{2}{c}{假新闻召回} & \multicolumn{2}{c}{真新闻召回} \\",
           r"方法 & 原文 & 改写 & 原文 & 改写 & 原文 & 改写 & 原文 & 改写 \\", r"\midrule"]
    for b in order:
        cells = []
        for D in DS:
            r = A[D]["style"].get(b)
            cells += [pct(r[k]["mean"], 1) for k in ("fake_recall_orig", "fake_recall_style", "real_recall_orig", "real_recall_style")] if r else ["--"] * 4
        if any(c != "--" for c in cells):
            out.append(NAME[b] + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


LOC_ROWS = [("LLM suspicion scores", "LLM 怀疑度评分（不训练）", "声明"), ("BERT + occlusion", "BERT + 遮挡归因", "句子"),
            ("LoRA-LLM + occlusion", "LoRA-LLM + 遮挡归因", "句子"), ("CLARA", r"\method{}", "声明"),
            ("CLARA-D", r"\student{}", "句子"), None,
            ("clara-entangled", r"\method{}：耦合式实例打分", "声明"), ("clara-no_gate", r"\method{}：去掉门控", "声明"),
            ("clara-no_args", r"\method{}：去掉论证文本", "声明"), ("clara-no_v", r"\method{}：去掉评分", "声明"),
            ("clara-agg_attention", r"\method{}：注意力聚合", "声明"),
            ("clara_d-no_inst", r"\student{}：去掉实例级蒸馏", "句子"), ("clara_d-no_teacher", r"\student{}：无教师", "句子")]


def tab_loc(A):
    out = [r"\begin{tabular}{llcccc}", r"\toprule",
           r" & & \multicolumn{2}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{2}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}", r"方法 & 单位 & Hit@1 & MRR & Hit@1 & MRR \\", r"\midrule"]
    out.append("随机猜测 & 句子 & " + " & ".join("{} & --".format(pct(A[D]["localisation"]["CLARA"]["chance"], 1)) for D in DS) + r" \\")
    for row in LOC_ROWS:
        if row is None:
            out.append(r"\midrule")
            continue
        key, label, unit = row
        cells = []
        for D in DS:
            r = A[D]["localisation"].get(key)
            if r is None:
                cells += ["--", "--"]
            else:
                h = pct(r["hit@1"], 1)
                if "hit@1_std" in r:
                    h += r"\pmstd{" + pct(r["hit@1_std"], 1) + "}"
                cells += [h, pct(r["mrr"], 1)]
        if any(c != "--" for c in cells):
            out.append("{} & {} & ".format(label, unit) + " & ".join(cells) + r" \\")
    if out[-1] == r"\midrule":
        out.pop()
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_tamper(A):
    order = ["bert", "lora", "bert_zs", "arg", "factguard", "clara", "clara_d"]
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{3}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"方法 & 原文判假 & 篡改后判假 & 成对正确率 & 原文判假 & 篡改后判假 & 成对正确率 \\", r"\midrule"]
    for b in order:
        cells = []
        for D in DS:
            r = A[D]["localisation"]["_sensitivity"].get(b)
            cells += [pct(r["flag_rate_original"], 1), pct(r["flag_rate_tampered"], 1), pct(r["pairwise_acc"], 1)] if r else ["--"] * 3
        if any(c != "--" for c in cells):
            out.append(NAME[b] + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_lodo(A):
    L = A["ARG-CN"]["lodo"]
    bases = ["bert", "bert_zs", "arg", "clara"]
    doms = [d for d in L if not d.startswith("_")]
    out = [r"\begin{tabular}{lrrcccc}", r"\toprule",
           "留出领域 & 训练数 & 测试数 & " + " & ".join(NAME[b] for b in bases) + r" \\", r"\midrule"]
    for d in doms:
        best = max(L[d][b]["mean"] for b in bases if b in L[d])
        cells = [pm(L[d][b], bold=abs(L[d][b]["mean"] - best) < 1e-12) if b in L[d] else "--" for b in bases]
        out.append("{} & {:,} & {:,} & ".format(d, L[d]["_n_train"], L[d]["_n_test"]) + " & ".join(cells) + r" \\")
    out.append(r"\midrule")
    avg = L["_macro_avg"]
    out.append("平均 & & & " + " & ".join(pct(avg[b]) if b in avg else "--" for b in bases) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_cross(A):
    """Relevance gate on analyses that are on topic but unfaithful in one place."""
    f3 = lambda x: "{:.3f}".format(x)
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{3}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"篡改后的文本搭配 & 被改句的声明 & 其余声明 & 整体实例 & 被改句的声明 & 其余声明 & 整体实例 \\", r"\midrule"]
    for key, label in (("tamper", "它自己的分析"), ("cross", "原文的分析（一处与文本不符）")):
        cells = []
        for D in DS:
            c = A[D]["cross"]
            cells += [f3(c[key + "/r_edited_sentence"]), f3(c[key + "/r_other_claims"]), f3(c[key + "/r_overall"])]
        out.append(label + " & " + " & ".join(cells) + r" \\")
    out.append(r"\midrule")
    out.append("AUC：$r$ $\\rightarrow$ 声明是否仍与文本相符 & " + " & ".join(
        r"\multicolumn{3}{c}{" + f3(A[D]["cross"]["cross/auc_r_unfaithful"]) + "}" for D in DS) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_distill(A):
    rows = [("bert", "BERT（同样的 3 个种子）"), ("arg_d", "ARG-D"), ("factguard_d", "FactGuard-D"),
            ("clara_d", r"\student{}"), ("clara_d-no_inst", r"\student{}：去掉实例级蒸馏"),
            ("clara_d-no_teacher", r"\student{}：无教师（仅句子级 MIL）")]
    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{2}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"模型 & 学生 & 其教师 & 学生 & 其教师 \\", r"\midrule"]
    for key, label in rows:
        cells = []
        for D in DS:
            r = A[D]["distill"].get(key)
            cells += [pm(r["test"]) if r else "--", pm(r["teacher"]) if r and "teacher" in r else "--"]
        if any(c != "--" for c in cells):
            out.append(label + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_llm2(A):
    order = ["bert_zs", "arg", "factguard", "clara"]
    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{2}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"方法 & Qwen2.5-7B & Qwen3-8B & Qwen2.5-7B & Qwen3-8B \\", r"\midrule"]
    out.append("零样本判断（不训练） & " + " & ".join(
        "{} & {}".format(pct(A[D]["llm_only"]["zero-shot"]["test"]["macf1"]),
                         pct(A[D]["llm2"]["zero_shot"]) if "zero_shot" in A[D]["llm2"] else "--") for D in DS) + r" \\")
    for b in order:
        cells = []
        for D in DS:
            r = A[D]["llm2"].get(b)
            cells += [pm(r["first"]), pm(r["second"])] if r else ["--", "--"]
        out.append(NAME[b] + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_selective(A):
    rows = [("bert", "maxprob", "BERT"), ("lora", "maxprob", "LoRA-LLM"), ("bert_zs", "maxprob", "BERT+ZS"),
            ("arg", "maxprob", "ARG"), ("arg", "selector", "ARG + 选择器"),
            ("factguard", "maxprob", "FactGuard"), ("factguard", "selector", "FactGuard + 选择器"),
            ("clara", "maxprob", r"\method{}"), ("clara", "selector", r"\method{} + 选择器")]
    cols = [("aurc", r"AURC$\downarrow$"), ("acc@10", "10\\%"), ("acc@50", "50\\%"), ("acc@90", "90\\%"), ("acc@100", "100\\%")]
    out = [r"\begin{tabular}{l" + "c" * (2 * len(cols)) + "}", r"\toprule",
           r" & \multicolumn{%d}{c}{" % len(cols) + DS_LABEL["ARG-CN"] + r"} & \multicolumn{%d}{c}{" % len(cols) + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (len(cols) + 1, len(cols) + 2, 2 * len(cols) + 1),
           r" & & \multicolumn{4}{c}{覆盖率为下列值时的准确率} & & \multicolumn{4}{c}{覆盖率为下列值时的准确率} \\",
           r"\cmidrule(lr){3-6}\cmidrule(lr){8-11}",
           "方法 & " + " & ".join(c[1] for c in cols) + " & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    for base, kind, label in rows:
        cells = []
        for D in DS:
            r = A[D]["selective"].get(base, {}).get(kind)
            cells += [pct(r[c[0]]) if r and c[0] in r else "--" for c in cols]
        if any(c != "--" for c in cells):
            out.append(label + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


COST = [("zs", "零样本判断"), ("arg", "ARG（TD + CS 理由）"), ("factguard", "FactGuard（topic-content + CS 理由）"),
        ("clara", r"\method{}（结构化调用 + 零样本评分）"), ("clara_single", r"\method{}，严格单次调用")]


def tab_cost(A):
    out = [r"\begin{tabular}{lccccccc}", r"\toprule",
           r" & & \multicolumn{3}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{3}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
           r"LLM 输出 & 调用数 & 输入 & 输出 & 秒/千条 & 输入 & 输出 & 秒/千条 \\", r"\midrule"]
    for key, label in COST:
        cells = []
        for D in DS:
            g = A[D]["cost"]["groups"][key]
            cells += ["{:.0f}".format(g["prompt"]), "{:.0f}".format(g["completion"]),
                      "{:.0f}".format(g["seconds_per_1000"]) if "seconds_per_1000" in g else "--"]
        out.append("{} & {} & ".format(label, A["ARG-CN"]["cost"]["groups"][key]["calls"]) + " & ".join(cells) + r" \\")
    if all("lora_cost" in A[D] for D in DS):
        cells = []
        for D in DS:
            cells += ["{:.0f}".format(A[D]["cost"]["tokens"]["zs"]["prompt"]), "0",
                      "{:.0f}".format(1000 * A[D]["lora_cost"]["eval_seconds_per_item"])]
        out.append(r"\midrule")
        out.append("LoRA-LLM（一次前向，不生成文本）$^\\ddagger$ & 1 & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_train_cost(A):
    order = ["bert", "bert_zs", "arg", "factguard", "clara", "lora"]
    out = [r"\begin{tabular}{lrrr}", r"\toprule", r"方法 & 可训练参数（M） & 训练秒数（CN） & 训练秒数（EN） \\", r"\midrule"]
    for b in order:
        if b not in A["ARG-CN"]["main"]:
            continue
        m = A["ARG-CN"]["main"][b]
        en = A["ARG-EN"]["main"].get(b)
        out.append("{} & {:.1f} & {:.0f} & {} \\\\".format(NAME[b], m["n_trainable"] / 1e6, m["train_seconds"],
                                                         "{:.0f}".format(en["train_seconds"]) if en else "--"))
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_grid(A):
    bases = ["bert", "bert_zs", "arg_gpt", "arg", "factguard", "clara"]
    out = [r"\begin{tabular}{llcccc}", r"\toprule",
           r" & & \multicolumn{2}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{2}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}", r"方法 & 学习率 & 验证 & 测试 & 验证 & 测试 \\", r"\midrule"]
    for b in bases:
        for lr in ("2e-05", "5e-05"):
            name = "{}@lr{}".format(b, lr)
            cells = []
            for D in DS:
                r = A[D]["grid"].get(name)
                sel = A[D]["selected"].get(b) == name
                cells += [pm(r["val"], bold=sel), pm(r["test"])] if r else ["--", "--"]
            if any(c != "--" for c in cells):
                out.append("{} & {} & ".format(NAME[b] if lr == "2e-05" else "", lr.replace("e-05", r"\!\times\!10^{-5}").join(["$", "$"])) + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_sig(A):
    def cells_for(cmp):
        if cmp is None or "bootstrap" not in cmp:
            return ["--"] * 4
        bs = cmp["bootstrap"]
        return ["{:+.2f}".format(100 * cmp["diff_mean"]), "[{:+.2f}, {:+.2f}]".format(100 * bs["ci_lo"], 100 * bs["ci_hi"]),
                fmt_p(bs["p"]), fmt_p(cmp.get("welch_p", float("nan")))]

    out = [r"\begin{tabular}{lcccccccc}", r"\toprule",
           r" & \multicolumn{4}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{4}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
           r"\method{} 对比 & $\Delta$ & 95\% CI & $p_{\text{样本}}$ & $p_{\text{种子}}$ & $\Delta$ & 95\% CI & $p_{\text{样本}}$ & $p_{\text{种子}}$ \\"]
    blocks = [("原始测试集", "significance", ["bert", "bert_zs", "arg_gpt", "arg", "arg_swap", "factguard", "lora"], "{}"),
              ("全部分析被整条替换", "noise_sig", ["arg", "arg_swap"], "noise100/{}"),
              ("全部声明/理由句被逐个替换", "noise_sig", ["arg", "arg_swap"], "unit100/{}")]
    for title, sec, bases, key in blocks:
        body = []
        for b in bases:
            cells = []
            for D in DS:
                cells += cells_for(A[D].get(sec, {}).get(key.format(b)))
            if any(c != "--" for c in cells):
                body.append(NAME[b] + " & " + " & ".join(cells) + r" \\")
        if body:
            out += [r"\midrule", r"\multicolumn{9}{l}{\emph{" + title + r"}} \\"] + body
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_live(A):
    L = A["ARG-CN"]["live"]
    out = [r"\begin{tabular}{lccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{macro-F1（\%）} & \multicolumn{2}{c}{秒/次} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
           r"方法 & 缓存+混合精度 & 缓存+全精度 & 每步重算 & 缓存+混合精度 & 每步重算 \\", r"\midrule"]
    for name, r in sorted(L.items()):
        base = name.split("@")[0]
        out.append("{} & {} & {} & {} & {} & {:.0f} \\\\".format(
            NAME.get(base, base), pm(r["cached"]) if "cached" in r else "--", pm(r["fp32"]) if "fp32" in r else "--",
            pm(r["live"]), "{:.0f}".format(r["cached_seconds"]) if "cached_seconds" in r else "--", r["live_seconds"]))
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_loc_type(A):
    label = {"negation": "否定", "number": "数字", "time": "时间", "entity": "实体", "outcome": "结果/程度"}
    out = [r"\begin{tabular}{lrccrcc}", r"\toprule",
           r" & \multicolumn{3}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{3}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"篡改类型 & 条数 & \method{} & BERT+遮挡 & 条数 & \method{} & BERT+遮挡 \\", r"\midrule"]
    types = [t for t in ("entity", "outcome", "number", "time", "negation") if any(t in A[D]["localisation"]["_by_type"] for D in DS)]
    for t in types:
        cells = []
        for D in DS:
            r = A[D]["localisation"]["_by_type"].get(t)
            if r is None:
                cells += ["--"] * 3
            else:
                cells += ["{:.0f}".format(r["_n_items"]), pct(r["CLARA"]["hit@1"], 1),
                          pct(r["BERT + occlusion"]["hit@1"], 1) if "BERT + occlusion" in r else "--"]
        out.append(label[t] + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------ prompts, cases
_ESC = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "%": r"\%", "_": r"\_", "&": r"\&", "#": r"\#",
        "$": r"\$", "^": r"\^{}", "~": r"\~{}", "<": r"\textless{}", ">": r"\textgreater{}"}


def esc(text):
    return "".join(_ESC.get(ch, ch) for ch in text)


def prompt_block(text):
    """A prompt as a LaTeX quote: one paragraph per line, leading spaces kept as indentation."""
    lines = []
    for raw in text.strip("\n").split("\n"):
        if not raw.strip():
            lines.append(r"\vspace{3pt}")
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((r"\hspace*{%dem}" % indent if indent else "") + esc(raw.strip()) + r"\par")
    return "{\\small\\setlength{\\parindent}{0pt}\\begin{quote}\n" + "\n".join(lines) + "\n\\end{quote}}\n"


def write_prompts():
    from clara.llm import tasks as T

    zh_msg = "〈按句编号的消息文本：[1] …… [2] ……〉"
    en_msg = "<the message with numbered sentences: [1] ... [2] ...>"
    b = T.build_clara("占位。", "zh", 0)
    text = b["system"] + "\n\n" + b["user"].split("【消息】")[0] + "【消息】\n" + zh_msg
    write(os.path.join(ROOT, "paper", "tables", "prompt_clara_zh.tex"), prompt_block(text))
    b = T.build_clara("Placeholder.", "en", 0)
    text = b["system"] + "\n\n" + b["user"].split("Message:")[0] + "Message:\n" + en_msg
    write(os.path.join(ROOT, "paper", "tables", "prompt_clara_en.tex"), prompt_block(text))
    text = T.STYLE_SYSTEM["zh"] + "\n\n" + T.STYLE_USER["zh"].format(
        style="〈目标文风：" + T.STYLE_NAME["zh"]["authoritative"] + "；或：" + T.STYLE_NAME["zh"]["sensational"] + "〉", content="〈原文〉")
    write(os.path.join(ROOT, "paper", "tables", "prompt_style_zh.tex"), prompt_block(text))
    text = T.TAMPER_SYSTEM["zh"] + "\n\n" + T.TAMPER_USER["zh"].format(content="〈消息全文〉", sentence="〈被选中的句子〉")
    write(os.path.join(ROOT, "paper", "tables", "prompt_tamper_zh.tex"), prompt_block(text))


TYPE_ZH = {"fact": "事实", "number": "数字", "quote": "引述", "opinion": "观点", "overall": "整体"}


def tab_cases(A):
    cases = A["ARG-CN"]["cases"]
    out = []
    for n, c in enumerate(cases, 1):
        out.append(r"\noindent\textbf{样例 %d}（篡改类型：%s；\method{} 预测 $\hat y=%.2f$）\par" % (
            n, {"negation": "否定", "number": "数字", "time": "时间", "entity": "实体", "outcome": "结果/程度"}[c["change_type"]], c["prob"]))
        out.append(r"{\small\begin{quote}")
        for j, sent in enumerate(c["sents"]):
            t = esc(sent.strip())
            if j == c["tamper_sent"]:
                t = r"\textbf{〔被篡改〕}" + t + r"\quad\textbf{〔原句〕}" + esc(c["orig_sentence"])
            out.append("[%d] %s\\par" % (j + 1, t))
        out.append(r"\end{quote}}")
        out.append(r"{\small\begin{tabularx}{\linewidth}{@{}clXccc@{}}")
        out.append(r"\toprule")
        out.append(r"句 & 类型 & 声明与两方论证 & $v_k$ & $g_k$ & $p_k$ \\")
        out.append(r"\midrule")
        for r in c["instances"]:
            body = r"\textbf{" + esc(r["claim"]) + "}"
            if r["support"] or r["refute"]:
                body += r"\newline $+$ " + esc(r["support"]) + r"\newline $-$ " + esc(r["refute"])
            sent = "--" if r["type"] == "overall" else str(r["sent"] + 1)
            out.append("%s & %s & %s & %.2f & %.2f & %.2f \\\\" % (sent, TYPE_ZH[r["type"]], body, r["v"], r["g"], r["p"]))
        out.append(r"\bottomrule")
        out.append(r"\end{tabularx}}")
        out.append(r"\par\medskip")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------ figures
# categorical slot of every entity: fixed, never reassigned between figures
# (validated with the palette checker for the sets that share a plot: 0-1-2-3-4 and 0-1-6)
ENT = {"clara": 0, "arg": 1, "factguard": 2, "bert_zs": 3, "arg_swap": 4, "clara_d": 5, "lora": 6, "bert": None}
FIG_NAME = {"clara": "CLARA", "arg": "ARG", "factguard": "FactGuard", "bert_zs": "BERT+ZS", "lora": "LoRA-LLM",
            "clara_d": "CLARA-D", "bert": "BERT", "arg_swap": "ARG+错配负例"}


def _end_labels(ax, x_end, ends, x_text, ha="left"):
    """Direct labels at the line ends, spread vertically with thin leaders when they would collide."""
    from clara import plotting as P

    if not ends:
        return
    lo, hi = ax.get_ylim()
    pos = P.spread_labels([e[0] for e in ends], 0.085 * (hi - lo))
    for (y, name), py in zip(ends, pos):
        ax.annotate(name, (x_end, y), xytext=(x_text, py), textcoords="data", va="center", ha=ha,
                    fontsize=7.5, color=P.INK2, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=P.AXIS, linewidth=0.6, shrinkA=0, shrinkB=2))


def fig_noise(A, path):
    """Rows: item-level / unit-level replacement.  Columns: datasets."""
    import matplotlib.pyplot as plt

    from clara import plotting as P

    P.setup()
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.7))
    # FactGuard drops by 8-13 points under replacement (see the table); it is left out of the figure
    # so that the vertical axis can resolve the differences between the other methods
    levels = [("item", "整条替换：被替换的测试样本比例（%）", ("clara", "arg", "arg_swap", "bert_zs")),
              ("unit", "单元替换：每个声明/理由句被替换的概率（%）", ("clara", "arg", "arg_swap"))]
    handles = {}
    for row, (lvl, xlabel, bases) in enumerate(levels):
        for col, D in enumerate(DS):
            ax, N = axes[row][col], A[D]["noise"]
            ends, xmax = [], 100
            for base in bases:
                if base not in N or len(N[base][lvl]) < 2:
                    continue
                xs = sorted(int(k) for k in N[base][lvl])
                ys = [100 * N[base][lvl][str(x)]["mean"] for x in xs]
                es = [100 * N[base][lvl][str(x)]["std"] for x in xs]
                handles[FIG_NAME[base]] = P.line(ax, xs, ys, ENT[base], FIG_NAME[base], yerr=es)
                ends.append((ys[-1], FIG_NAME[base]))
            if "bert" in N:
                y = 100 * N["bert"]["item"]["0"]["mean"]
                handles["BERT（不使用 LLM）"] = ax.axhline(y, color=P.NEUTRAL, linewidth=1.2, zorder=2)
                ends.append((y, "BERT"))
            ax.set_xlim(-4, xmax + 4)
            ax.set_xticks([0, 20, 40, 60, 80, 100] if lvl == "item" else [0, 25, 50, 100])
            _end_labels(ax, xmax, ends, xmax + 9)
            ax.set_xlabel(xlabel, fontsize=8)
            if row == 0:
                ax.set_title(DS_LABEL[D], loc="left")
            if col == 0:
                ax.set_ylabel("macro-F1（%）")
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.03), handlelength=1.8, labelcolor=P.INK2, frameon=False, columnspacing=1.3)
    fig.subplots_adjust(wspace=0.55, hspace=0.5, right=0.88, top=0.9)
    P.save(fig, path)


def fig_selective(A, path):
    import matplotlib.pyplot as plt

    from clara import plotting as P

    P.setup()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    series = [("clara", "selector", "CLARA+选择器"), ("clara", "maxprob", "CLARA"), ("arg", "maxprob", "ARG"),
              ("lora", "maxprob", "LoRA-LLM"), ("bert", "maxprob", "BERT")]
    style = {"CLARA+选择器": dict(color=P.SLOTS[0], linewidth=2.0), "CLARA": dict(color=P.SLOTS[0], linewidth=1.2, alpha=0.55),
             "ARG": dict(color=P.SLOTS[1], linewidth=2.0), "LoRA-LLM": dict(color=P.SLOTS[ENT["lora"]], linewidth=2.0),
             "BERT": dict(color=P.NEUTRAL, linewidth=1.4)}
    handles = {}
    for ax, D in zip(axes, DS):
        C = A[D]["selective"]["_curves"]
        ends = []
        for base, kind, label in series:
            c = C.get("{}/{}".format(base, kind))
            if c is None:
                continue
            x, y = 100 * np.array(c["coverage"]), 100 * np.array(c["accuracy"])
            handles[label] = ax.plot(x, y, label=label, zorder=3, **style[label])[0]
            ends.append((y[0], label))
        ax.set_xlim(100, 10)
        _end_labels(ax, 10, ends, 4.5)
        ax.set_xlabel("覆盖率（%）；越往右拒答越多", fontsize=8)
        ax.set_title(DS_LABEL[D], loc="left")
    axes[0].set_ylabel("保留样本上的准确率（%）")
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.07), handlelength=1.8, labelcolor=P.INK2, frameon=False, columnspacing=1.3)
    fig.subplots_adjust(wspace=0.62, right=0.86)
    P.save(fig, path)


def fig_sensitivity(A, path):
    import matplotlib.pyplot as plt

    from clara import plotting as P

    P.setup()
    panels = [(D, "claims_k", "声明数上限 $K$（" + SHORT[D] + "）") for D in DS if "claims_k" in A[D]]
    for key, label in (("swap_rate", r"错配负例比例 $\rho$"), ("lambda_rel", r"$\lambda_{\mathrm{rel}}$"),
                       ("lambda_mil", r"$\lambda_{\mathrm{mil}}$"), ("lambda_gate", r"$\lambda_{\mathrm{gate}}$")):
        if key in A["ARG-CN"]:
            panels.append(("ARG-CN", key, label + "（CN）"))
    if not panels:
        raise KeyError("sensitivity")
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5, 2.05 * nrow), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (D, key, label) in zip(axes.flat, panels):
        S = A[D][key]
        xs = sorted(S, key=float)
        ys = [100 * S[x]["test"]["mean"] for x in xs]
        es = [100 * S[x]["test"]["std"] for x in xs]
        pos = list(range(len(xs)))
        P.line(ax, pos, ys, 0, "CLARA", yerr=es)
        for i, x in enumerate(xs):
            if S[x].get("default"):
                ax.annotate("默认", (i, ys[i]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7.5, color=P.INK2)
        if "main" in A[D] and "bert" in A[D]["main"]:
            yb = 100 * A[D]["main"]["bert"]["macf1"]
            ax.axhline(yb, color=P.NEUTRAL, linewidth=1.2, zorder=2)
            ax.annotate("BERT", (pos[-1], yb), xytext=(4, -8), textcoords="offset points", ha="right", fontsize=7.5, color=P.INK2)
        ax.set_xticks(pos)
        ax.set_xticklabels(["{:g}".format(float(x)) for x in xs])
        ax.set_xlabel(label, fontsize=8)
        ax.margins(x=0.15, y=0.35)
    for r in range(nrow):
        axes[r][0].set_ylabel("macro-F1（%）")
    fig.subplots_adjust(wspace=0.42, hspace=0.75)
    P.save(fig, path)


def figure(name, fn, A):
    try:
        fn(A, os.path.join(ROOT, "paper", "figures", name + ".pdf"))
        print("wrote paper/figures/{}.pdf".format(name))
    except KeyError as e:
        SKIPPED.append("{}: missing {}".format(name, e))


# ------------------------------------------------------------------------------------------ numbers
SKIP_KEYS = {"cases", "_curves"}
_BAD = set("\\{}%#~&$^")


def _leaves(obj, path):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_KEYS:
                continue
            yield from _leaves(v, path + [str(k)])
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield "/".join(path), float(obj)
    elif isinstance(obj, list) and path[-1].endswith("seeds"):
        yield "/".join(path[:-1] + [path[-1] + "_count"]), float(len(obj))


def write_numbers(A):
    """Every numeric leaf of analysis.json as a set of TeX macros, addressed by its path:

        \\pone{CN/noise/clara/item/100/mean}   percentage, one decimal        (75.5)
        \\ptwo{...}  percentage, two decimals    \\dpone{...} / \\dptwo{...}  signed percentage
        \\fthree{...} / \\ftwo{...} / \\fone{...}  plain number    \\cnt{...}  integer with separators

    A path that does not exist prints a visible marker instead of nothing.
    """
    lines = ["% generated by scripts/make_assets.py from results/analysis.json - do not edit",
             "\\makeatletter",
             "\\newcommand{\\num@get}[2]{\\ifcsname num@#1@#2\\endcsname\\csname num@#1@#2\\endcsname"
             "\\else\\textbf{??\\detokenize{#2}??}\\fi}",
             "\\newcommand{\\pone}[1]{\\num@get{pa}{#1}}", "\\newcommand{\\ptwo}[1]{\\num@get{pb}{#1}}",
             "\\newcommand{\\dpone}[1]{\\num@get{da}{#1}}", "\\newcommand{\\dptwo}[1]{\\num@get{db}{#1}}",
             "\\newcommand{\\fthree}[1]{\\num@get{fc}{#1}}", "\\newcommand{\\ftwo}[1]{\\num@get{fb}{#1}}",
             "\\newcommand{\\fone}[1]{\\num@get{fa}{#1}}", "\\newcommand{\\cnt}[1]{\\num@get{ct}{#1}}"]
    n = 0
    for D in DS:
        for key, v in _leaves(A[D], [SHORT[D]]):
            if any(ch in _BAD for ch in key) or v != v or abs(v) == float("inf"):
                continue
            forms = {"pa": "{:.1f}".format(100 * v), "pb": "{:.2f}".format(100 * v),
                     "da": "{:+.1f}".format(100 * v), "db": "{:+.2f}".format(100 * v),
                     "fc": "{:.3f}".format(v), "fb": "{:.2f}".format(v), "fa": "{:.1f}".format(v),
                     "ct": "{:,.0f}".format(v)}
            for tag, text in forms.items():
                lines.append("\\expandafter\\def\\csname num@{}@{}\\endcsname{{{}}}".format(tag, key, text))
            n += 1
    lines.append("\\makeatother")
    write(os.path.join(ROOT, "paper", "numbers.tex"), "\n".join(lines) + "\n")
    print("   {} numbers".format(n))


def main():
    with open(os.path.join(ROOT, "results", "analysis.json"), encoding="utf-8") as f:
        A = json.load(f)
    for name, fn in [("data", tab_data), ("main", tab_main), ("ablation", tab_ablation), ("gate", tab_gate),
                     ("gate_cells", tab_gate_cells), ("noise", tab_noise), ("useful", tab_useful),
                     ("style_llm", tab_style_llm), ("style", tab_style), ("style_recall", tab_style_recall),
                     ("loc", tab_loc), ("loc_type", tab_loc_type), ("tamper", tab_tamper), ("lodo", tab_lodo),
                     ("selective", tab_selective), ("cost", tab_cost), ("train_cost", tab_train_cost),
                     ("grid", tab_grid), ("sig", tab_sig), ("live", tab_live), ("cases", tab_cases),
                     ("llm2", tab_llm2), ("distill", tab_distill), ("cross", tab_cross)]:
        table(name, fn, A)
    write_prompts()
    figure("noise", fig_noise, A)
    figure("selective", fig_selective, A)
    figure("sensitivity", fig_sensitivity, A)
    write_numbers(A)
    for s in SKIPPED:
        print("SKIPPED", s)


if __name__ == "__main__":
    main()
