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
        "factguard_d": "FactGuard-D", "clara_d": r"\student{}"}
SKIPPED = []


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
    best = {}
    for D in DS:
        for c in MAIN_COLS:
            vals = [v[c] for v in A[D]["main"].values()]
            best[(D, c)] = max(vals)
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
       ("refute_only", "仅质疑论证"), ("no_args", "去掉论证文本，仅评分"), ("no_v", "去掉评分，仅论证文本"),
       ("fi_score", "整体评分取提示内第一印象（严格单次调用）"), None,
       ("no_gate", "去掉门控（$g\\equiv1$）"), ("item_gate", "新闻级门控（一条新闻共用一个 $g$）"),
       ("no_gate_fake", "门控监督不含假新闻声明"), None,
       ("agg_attention", "注意力聚合替代 noisy-OR"), ("no_global", "去掉整体文本分支"),
       ("entangled", "实例打分可见整条新闻（耦合式）")]


def tab_ablation(A):
    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r"\multirow{2}{*}{变体} & \multicolumn{2}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{2}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r" & macF1 & 噪声 100\% & macF1 & 噪声 100\% \\", r"\midrule"]
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
                cells += ["--", "--"]
                continue
            star = ""
            if "vs_full" in r and r["vs_full"].get("bootstrap", {}).get("p", 1.0) < 0.05:
                star = "$^{*}$"
            cells.append(pm(r["test"]) + star)
            cells.append(pm(r["noise100"]) if "noise100" in r else "--")
        if any(c != "--" for c in cells):
            out.append(label + " & " + " & ".join(cells) + r" \\")
    if out[-1] == r"\midrule":
        out.pop()
    out += [r"\bottomrule", r"\end{tabular}"]
    if not any("ablation" in A[D] for D in DS):
        raise KeyError("ablation")
    return "\n".join(out) + "\n"


def tab_gate(A):
    def g(D, k):
        return A[D]["gate"]["clara"][k]["mean"]

    out = [r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{2}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r" & 整体实例 & 真新闻声明 & 整体实例 & 真新闻声明 \\", r"\midrule"]
    spec = [("LLM 表态比例（\\%）", "decided_rate", pct), ("表态时 LLM 站对边的比例（\\%）", "llm_acc", pct),
            ("AUC：门控 $g$ $\\rightarrow$ LLM 是否站对边", "auc_gate", lambda x: "{:.3f}".format(x)),
            ("AUC：LLM 自身置信度 $\\rightarrow$ 同上", "auc_llm_conf", lambda x: "{:.3f}".format(x)),
            ("LLM 站对边时的平均 $g$", "gate_when_right", lambda x: "{:.3f}".format(x)),
            ("LLM 站错边时的平均 $g$", "gate_when_wrong", lambda x: "{:.3f}".format(x))]
    for label, key, fmt in spec:
        out.append(label + " & " + " & ".join(fmt(g(D, "{}/{}".format(part, key))) for D in DS for part in ("overall", "real_claims")) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_gate_noise(A):
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{3}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"方法 & 未替换 & 被替换 & AUC & 未替换 & 被替换 & AUC \\", r"\midrule"]
    for base in ("arg", "factguard", "clara"):
        cells = []
        for D in DS:
            r = A[D]["gate"]["noise"].get(base)
            cells += ["{:.3f}".format(r[k]) for k in ("clean", "corrupted", "auc")] if r else ["--"] * 3
        out.append(NAME[base] + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_style(A):
    order = ["bert", "lora", "bert_zs", "arg", "factguard", "clara", "clara_d"]
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
    bases = ["bert", "arg", "factguard", "clara"]
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


def tab_selective(A):
    rows = [("bert", "maxprob", "BERT"), ("lora", "maxprob", "LoRA-LLM"), ("bert_zs", "maxprob", "BERT+ZS"),
            ("arg", "maxprob", "ARG"), ("arg", "selector", "ARG + 选择器"),
            ("factguard", "maxprob", "FactGuard"), ("factguard", "selector", "FactGuard + 选择器"),
            ("clara", "maxprob", r"\method{}"), ("clara", "selector", r"\method{} + 选择器")]
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{3}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"方法 & AURC$\downarrow$ & Acc@90 & Acc@70 & AURC$\downarrow$ & Acc@90 & Acc@70 \\", r"\midrule"]
    for base, kind, label in rows:
        cells = []
        for D in DS:
            r = A[D]["selective"].get(base, {}).get(kind)
            cells += [pct(r["aurc"]), pct(r["acc@90"]), pct(r["acc@70"])] if r else ["--"] * 3
        if any(c != "--" for c in cells):
            out.append(label + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


COST = [("零样本判断", ["zs"]), ("ARG（TD + CS 理由）", ["td", "cs"]),
        ("FactGuard（topic-content + CS 理由）", ["tc", "cs"]),
        (r"\method{}（结构化调用 + 零样本评分）", ["clara", "zs"]),
        (r"\method{}，严格单次调用", ["clara"])]


def tab_cost(A):
    out = [r"\begin{tabular}{lccccccc}", r"\toprule",
           r" & & \multicolumn{3}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{3}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
           r"LLM 输出 & 调用数 & 输入 & 输出 & 秒/千条 & 输入 & 输出 & 秒/千条 \\", r"\midrule"]
    for label, tasks in COST:
        cells = []
        for D in DS:
            t, s = A[D]["cost"]["tokens"], A[D]["cost"]["gen_seconds_per_item"]
            cells += ["{:.0f}".format(sum(t[k]["prompt"] for k in tasks)), "{:.0f}".format(sum(t[k]["completion"] for k in tasks)),
                      "{:.0f}".format(1000 * sum(s[k] for k in tasks)) if all(k in s for k in tasks) else "--"]
        out.append("{} & {} & ".format(label, len(tasks)) + " & ".join(cells) + r" \\")
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
    order = ["bert", "bert_zs", "arg_gpt", "arg", "factguard", "lora"]
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{" + DS_LABEL["ARG-CN"] + r"} & \multicolumn{3}{c}{" + DS_LABEL["ARG-EN"] + r"} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"\method{} 对比 & $\Delta$macF1 & 95\% CI & $p$ & $\Delta$macF1 & 95\% CI & $p$ \\", r"\midrule"]
    for b in order:
        cells = []
        for D in DS:
            r = A[D]["significance"].get(b)
            if r is None or "bootstrap" not in r:
                cells += ["--"] * 3
            else:
                bs = r["bootstrap"]
                cells += ["{:+.2f}".format(100 * r["diff_mean"]), "[{:+.2f}, {:+.2f}]".format(100 * bs["ci_lo"], 100 * bs["ci_hi"]),
                          "{:.3f}".format(bs["p"]) if bs["p"] >= 0.001 else "$<$0.001"]
        if any(c != "--" for c in cells):
            out.append(NAME[b] + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_live(A):
    L = A["ARG-CN"]["live"]
    out = [r"\begin{tabular}{lcccc}", r"\toprule", r"方法 & 缓存（macF1） & 每步重算（macF1） & 缓存（秒/次） & 重算（秒/次） \\", r"\midrule"]
    for name, r in sorted(L.items()):
        base = name.split("@")[0]
        out.append("{} & {} & {} & {:.0f} & {:.0f} \\\\".format(
            NAME.get(base, base), pm(r["cached"]) if "cached" in r else "--", pm(r["live"]),
            r.get("cached_seconds", float("nan")), r["live_seconds"]))
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
                t = r"\underline{" + t + r"}（原句：" + esc(c["orig_sentence"]) + "）"
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
ENT = {"clara": 0, "arg": 1, "factguard": 2, "bert_zs": 3, "lora": 4, "clara_d": 5, "bert": None}
FIG_NAME = {"clara": "CLARA", "arg": "ARG", "factguard": "FactGuard", "bert_zs": "BERT+ZS", "lora": "LoRA-LLM",
            "clara_d": "CLARA-D", "bert": "BERT"}


def fig_noise(A, path):
    import matplotlib.pyplot as plt

    from clara import plotting as P

    P.setup()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.55), sharey=False)
    for ax, D in zip(axes, DS):
        N = A[D]["noise"]
        ends = []
        for base in ("clara", "arg", "factguard", "bert_zs"):
            if base not in N:
                continue
            xs = sorted(int(k) for k in N[base])
            ys = [100 * N[base][str(x)]["mean"] for x in xs]
            es = [100 * N[base][str(x)]["std"] for x in xs]
            P.line(ax, xs, ys, ENT[base], FIG_NAME[base], yerr=es)
            ends.append((ys[-1], FIG_NAME[base]))
        if "bert" in N:
            y = 100 * N["bert"]["0"]["mean"]
            ax.axhline(y, color=P.NEUTRAL, linewidth=1.2, zorder=2)
            ends.append((y, "BERT"))
        lo, hi = ax.get_ylim()
        pos = P.spread_labels([e[0] for e in ends], 0.075 * (hi - lo))
        for (y, name), py in zip(ends, pos):
            ax.annotate(name, (100, y), xytext=(104, py), textcoords="data", va="center", ha="left",
                        fontsize=8, color=P.INK2, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=P.AXIS, linewidth=0.6, shrinkA=0, shrinkB=2))
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("LLM 输出被替换的测试样本比例（%）")
        ax.set_title(DS_LABEL[D], loc="left")
    axes[0].set_ylabel("macro-F1（%）")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.06), handlelength=1.8,
               labelcolor=P.INK2, frameon=False, columnspacing=1.6)
    fig.subplots_adjust(wspace=0.42, right=0.9)
    P.save(fig, path)


def fig_selective(A, path):
    import matplotlib.pyplot as plt

    from clara import plotting as P

    P.setup()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.55))
    series = [("clara", "selector", "CLARA + 选择器"), ("arg", "maxprob", "ARG"), ("factguard", "maxprob", "FactGuard"),
              ("lora", "maxprob", "LoRA-LLM"), ("bert", "maxprob", "BERT")]
    for ax, D in zip(axes, DS):
        C = A[D]["selective"]["_curves"]
        ends = []
        for base, kind, label in series:
            c = C.get("{}/{}".format(base, kind))
            if c is None:
                continue
            x, y = 100 * np.array(c["coverage"]), 100 * np.array(c["accuracy"])
            color = P.NEUTRAL if ENT[base] is None else P.SLOTS[ENT[base]]
            ax.plot(x, y, color=color, linewidth=1.4 if ENT[base] is None else 2.0, label=label, zorder=3)
            ends.append((y[0], label))
        lo, hi = ax.get_ylim()
        pos = P.spread_labels([e[0] for e in ends], 0.075 * (hi - lo))
        for (y, name), py in zip(ends, pos):
            ax.annotate(name, (50, y), xytext=(47.5, py), textcoords="data", va="center", ha="right",
                        fontsize=8, color=P.INK2, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=P.AXIS, linewidth=0.6, shrinkA=0, shrinkB=2))
        ax.set_xlim(50, 100)
        ax.invert_xaxis()
        ax.set_xlabel("覆盖率（%），向右拒答更多")
        ax.set_title(DS_LABEL[D], loc="left")
    axes[0].set_ylabel("保留样本上的准确率（%）")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.06), handlelength=1.8,
               labelcolor=P.INK2, frameon=False, columnspacing=1.4)
    fig.subplots_adjust(wspace=0.5, left=0.19)
    P.save(fig, path)


def fig_sensitivity(A, path):
    import matplotlib.pyplot as plt

    from clara import plotting as P

    P.setup()
    panels = []
    for D in DS:
        if "claims_k" in A[D]:
            panels.append((D, "claims_k", "声明数上限 $K$"))
    for key, label in (("lambda_mil", r"$\lambda_{\mathrm{mil}}$"), ("lambda_gate", r"$\lambda_{\mathrm{gate}}$")):
        if key in A["ARG-CN"]:
            panels.append(("ARG-CN", key, label))
    if not panels:
        raise KeyError("sensitivity")
    fig, axes = plt.subplots(1, len(panels), figsize=(6.3, 2.2), squeeze=False)
    for ax, (D, key, label) in zip(axes[0], panels):
        S = A[D][key]
        xs = sorted(S, key=float)
        ys = [100 * S[x]["test"]["mean"] for x in xs]
        es = [100 * S[x]["test"]["std"] for x in xs]
        pos = list(range(len(xs)))
        P.line(ax, pos, ys, 0, "CLARA", yerr=es)
        for i, x in enumerate(xs):
            if S[x].get("default"):
                ax.annotate("默认", (i, ys[i]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7.5, color=P.INK2)
        if "main" in A[D] and "bert" in A[D]["main"]:
            ax.axhline(100 * A[D]["main"]["bert"]["macf1"], color=P.NEUTRAL, linewidth=1.2, zorder=2)
            ax.annotate("BERT", (len(xs) - 1, 100 * A[D]["main"]["bert"]["macf1"]), xytext=(0, -9),
                        textcoords="offset points", ha="right", fontsize=7.5, color=P.INK2)
        ax.set_xticks(pos)
        ax.set_xticklabels(["{:g}".format(float(x)) for x in xs])
        ax.set_xlabel(label)
        ax.set_title(SHORT[D], loc="left")
        ax.margins(x=0.12, y=0.25)
    axes[0][0].set_ylabel("macro-F1（%）")
    fig.subplots_adjust(wspace=0.45)
    P.save(fig, path)


def figure(name, fn, A):
    try:
        fn(A, os.path.join(ROOT, "paper", "figures", name + ".pdf"))
        print("wrote paper/figures/{}.pdf".format(name))
    except KeyError as e:
        SKIPPED.append("{}: missing {}".format(name, e))


# ------------------------------------------------------------------------------------------ macros
def numbers(A):
    """Inline numbers used in the prose, as LaTeX macros (letters only in names)."""
    N = {}

    def put(name, value):
        assert name.isalpha(), name
        N[name] = value

    for D in DS:
        S = SHORT[D]
        d = A[D].get("data")
        if d:
            put(S + "nTrain", "{:,}".format(d["train"]["n"]))
            put(S + "nVal", "{:,}".format(d["val"]["n"]))
            put(S + "nTest", "{:,}".format(d["test"]["n"]))
        for base, m in A[D].get("main", {}).items():
            key = S + "".join(w.capitalize() for w in base.split("_"))
            put(key + "F", pct(m["macf1"]))
            put(key + "Fstd", pct(m["macf1_std"]))
            put(key + "Acc", pct(m["acc"]))
            put(key + "Auc", pct(m["auc"]))
            put(key + "Ece", pct(m["ece"]))
            put(key + "N", str(m["n"]))
        c = A[D].get("cost")
        if c:
            put(S + "parseRate", "{:.2f}".format(100 * (1 - c["clara_second_attempt_rate"])))
            put(S + "claraIn", "{:.0f}".format(c["tokens"]["clara"]["prompt"]))
            put(S + "claraOut", "{:.0f}".format(c["tokens"]["clara"]["completion"]))
        pt = A[D].get("perturb")
        if pt:
            put(S + "styleN", "{:,}".format(pt["style_n"]))
            put(S + "styleRew", "{:,}".format(pt["style_rewritten"]))
            put(S + "tamperN", "{:,}".format(pt["tamper_n"]))
            put(S + "tamperSents", "{:.1f}".format(pt["tamper_mean_sents"]))
            if "tamper_candidates" in pt:
                put(S + "tamperCand", "{:,}".format(pt["tamper_candidates"]))
        fp = A[D].get("fact_preservation")
        if fp:
            put(S + "digitKept", "{:.1f}".format(100 * fp["kept_rate"]))
        for key, row in A[D].get("llm_only", {}).items():
            if key in ("zero-shot", "zero-shot CoT", "GPT-3.5 CS (released)", "GPT-3.5 TD (released)"):
                nm = {"zero-shot": "Zs", "zero-shot CoT": "Cot", "GPT-3.5 CS (released)": "GptCs", "GPT-3.5 TD (released)": "GptTd"}[key]
                put(S + nm + "F", pct(row["test"]["macf1"]))
    return N


def main():
    with open(os.path.join(ROOT, "results", "analysis.json"), encoding="utf-8") as f:
        A = json.load(f)
    for name, fn in [("data", tab_data), ("main", tab_main), ("ablation", tab_ablation), ("gate", tab_gate),
                     ("gate_noise", tab_gate_noise), ("style", tab_style), ("style_recall", tab_style_recall),
                     ("loc", tab_loc), ("loc_type", tab_loc_type), ("tamper", tab_tamper), ("lodo", tab_lodo),
                     ("selective", tab_selective), ("cost", tab_cost), ("train_cost", tab_train_cost),
                     ("grid", tab_grid), ("sig", tab_sig), ("live", tab_live), ("cases", tab_cases)]:
        table(name, fn, A)
    write_prompts()
    figure("noise", fig_noise, A)
    figure("selective", fig_selective, A)
    figure("sensitivity", fig_sensitivity, A)
    N = numbers(A)
    lines = ["% generated by scripts/make_assets.py from results/analysis.json - do not edit"]
    lines += ["\\newcommand{{\\{}}}{{{}}}".format(k, v) for k, v in sorted(N.items())]
    write(os.path.join(ROOT, "paper", "numbers.tex"), "\n".join(lines) + "\n")
    for s in SKIPPED:
        print("SKIPPED", s)


if __name__ == "__main__":
    main()
