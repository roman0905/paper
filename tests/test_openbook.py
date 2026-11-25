"""Smoke tests of the open-book pipeline on synthetic data (CPU, tiny randomly initialised LM).

They check mechanics - data normalisation, leak tagging, ranking, prompt layout, the position of
the evidence read-outs, training and two-pass inference - not model quality.
"""
import os

import numpy as np
import pytest
import torch

from openbook import data as D
from openbook import pipeline as P
from openbook import retrieval as R
from openbook import views as V
from openbook.metrics import classification_metrics, paired_bootstrap
from openbook.prompts import verifier_prompt

TOKENIZER = os.environ.get("CLARA_LLM_TOKENIZER", "/media/dell/DATA/models/Qwen2.5-7B-Instruct")
TOPICS = ["river bridge collapse", "school budget vote", "vaccine trial result", "city bus strike",
          "solar farm permit", "hospital merger plan"]


def raw_claim(i, split):
    topic = TOPICS[i % len(TOPICS)]
    label = ["false", "half", "true"][i % 3]
    reports = []
    for r in range(4):
        sents = []
        for s in range(5):
            on_topic = (r + s) % 2 == 0
            text = ("Officials said the {} was {} according to record {} of case {} .".format(topic, label, 10 * r + s, i)
                    if on_topic else "Unrelated weather note number {} about clouds and wind on day {} .".format(10 * r + s, i))
            sents.append({"sent": text, "is_evidence": int(on_topic and s < 3)})
        link = "https://news{}.example.com/a{}".format(r, i)
        if r == 3 and i % 4 == 0:
            link = "https://www.snopes.com/fact-check/item{}".format(i)
        reports.append({"link": link, "content": "", "domain": "", "tokenized": sents})
    return {"event_id": "{}{}".format(split, i), "claim": "The {} happened as reported, item {} .".format(topic, i),
            "label": label, "original_label": label, "explain": "It is {} because of the record .".format(label), "reports": reports}


@pytest.fixture(scope="module")
def corpus():
    items = []
    for sp, n in (("train", 24), ("val", 9), ("test", 9)):
        items += [D.normalise(raw_claim(i, sp), "RAWFC", sp) for i in range(n)]
    return D.Corpus("RAWFC", items)


@pytest.fixture(scope="module")
def ranking(corpus):
    return {p: {it["id"]: R.bm25_ranking(it, p) for it in corpus.items} for p in D.PROTOCOLS}


def test_normalise_and_flags(corpus):
    it = corpus.split("train")[0]
    assert it["label"] == 0 and len(it["reports"]) == 4
    assert it["reports"][0]["sents"][0].endswith("of case 0.")          # spacing before punctuation undone
    assert it["reports"][3]["flags"] == ["source"] and it["reports"][0]["flags"] == []
    assert len(D.reports_under(it, "strict")) == 3 and len(D.reports_under(it, "all")) == 4
    assert D.report_flags("https://github.com/x/liar/blob/master/test.tsv", ["a row"]) == ["dump"]
    assert D.report_flags("https://www.usatoday.com/story/news/factcheck/2020/x", ["plain text"]) == ["fc_url"]
    assert D.report_flags("https://example.org/a", ["PolitiFact rated the statement Pants on Fire"]) == ["fc_text"]
    assert D.report_flags("https://example.org/a", ["an ordinary sentence about a debate"]) == []


def test_pool_and_ranking(corpus, ranking):
    it = corpus.split("train")[0]
    assert len(R.pool(it, "all")) == 20 and len(R.pool(it, "strict")) == 15
    r = ranking["strict"][it["id"]]
    assert all(ri != 3 for ri, _, _ in r)                              # the flagged report is not a candidate
    assert [s for _, _, s in r] == sorted([s for _, _, s in r], reverse=True)
    top = [R.sentence(it, (a, b)) for a, b, _ in r[:4]]
    assert all("river bridge collapse" in t for t in top)              # on-topic sentences first
    q = R.ranking_quality(corpus.split("train"), ranking["strict"], ks=(4,))
    assert q["p@4"] > 0.4 and q["lift@4"] > 1.0
    scores = R.bm25_scores(["bridge"], [["bridge", "fell"], ["sunny", "day"], []])
    assert scores[0] > 0 and scores[1] == 0 and scores[2] == 0


def test_views(corpus, ranking):
    items, r = corpus.split("train"), ranking["strict"]
    std = V.standard(items, r, 6)
    assert all(len(ex["evidence"]) == 6 == len(ex["tags"]) == len(ex["cands"]) for ex in std)
    ex = std[0]
    assert ex["cands"] == sorted(ex["cands"], key=lambda c: ([c2[0] for c2 in ex["cands"]].index(c[0]), c[1]))
    sw = V.swapped(items, r, 6, seed=0)
    assert all(a["evidence"] != b["evidence"] and a["label"] == b["label"] for a, b in zip(std, sw))
    assert all(sum(ex["tags"]) == 0 for ex in sw)
    rng = np.random.RandomState(0)
    tr = V.training(items, r, 6, rng, top=12, swap_rate=0.5)
    kinds = [ex["kind"] for ex in tr]
    assert 0 < kinds.count("swap") < len(tr)
    assert all(sum(ex["tags"]) == 0 for ex in tr if ex["kind"] == "swap")
    picks = [tuple(V.sample_picks(15, 6, 12, rng)) for _ in range(20)]
    assert len(set(picks)) > 1 and all(len(p) == 6 and max(p) < 12 for p in picks)
    ch = V.chunked(items[:2], r, 6, 12)
    assert len(ch) == 4 and sorted(c for ex in ch[:2] for c in ex["cands"]) == sorted((a, b) for a, b, _ in r[items[0]["id"]][:12])
    assert V.closed(items)[0]["evidence"] is None
    orc = V.oracle(items, r, 6, 12)
    assert all(sum(o["tags"]) >= sum(x["tags"]) for o, x in zip(orc, std)) and sum(sum(o["tags"]) for o in orc) > sum(sum(x["tags"]) for x in std) - 1
    assert all(set(o["cands"]) <= {(a, b) for a, b, _ in r[it["id"]][:12]} for o, it in zip(orc, items))
    hidden = V.withhold_claim(std)
    assert all(h["claim"] != ex["claim"] and h["evidence"] == ex["evidence"] and h["label"] == ex["label"] for h, ex in zip(hidden, std))
    assert len({h["claim"] for h in hidden}) == 1 and std[0]["claim"].startswith("The ")        # the originals are untouched


def test_perturbations(corpus, ranking):
    items, r = corpus.split("test"), ranking["strict"]
    sw_items, sw_rank = V.perturb(items, r, "swap", 5, 10, seed=0)
    assert [it["id"] for it in sw_items] == [it["id"] for it in items] and [it["label"] for it in sw_items] == [it["label"] for it in items]
    for it, orig in zip(sw_items, items):
        pool = sw_rank[it["id"]]
        assert len(pool) == 10 and all(V.is_foreign(it, (a, b)) for a, b, _ in pool)
        assert all(it["reports"][a]["ev"][b] == 0 for a, b, _ in pool)                 # foreign sentences carry no tag
        assert it["reports"][:len(orig["reports"])] == orig["reports"] and "n_reports_own" not in orig
    mx_items, mx_rank = V.perturb(items, r, "mix", 5, 10, seed=0)
    for it in mx_items:
        flags = [V.is_foreign(it, (a, b)) for a, b, _ in mx_rank[it["id"]]]
        assert sum(flags) == 5 and flags[:4] == [False, True, False, True]            # interleaved by rank
    ex = V.standard(mx_items, mx_rank, 6)[0]
    assert len(ex["evidence"]) == 6 and sum(V.is_foreign(mx_items[0], c) for c in ex["cands"]) == 3
    dt_items, dt_rank = V.perturb(items, r, "droptop", 5, 10)
    assert all(dt_rank[it["id"]] == r[it["id"]][5:] for it in items) and dt_items == list(items)


def test_prompt():
    msgs = verifier_prompt("C", [("a.com", "one two"), ("", "three")], ["false", "true"])
    assert "[1] (a.com) one two\n[2] (unknown source) three" in msgs[1]["content"] and "A. false\nB. true" in msgs[1]["content"]
    assert "Evidence" not in verifier_prompt("C", None, ["false", "true"])[1]["content"]
    assert "(no evidence was retrieved)" in verifier_prompt("C", [], ["false", "true"])[1]["content"]


def test_metrics():
    y = np.array([0, 1, 2, 0, 1, 2])
    good = np.eye(3)[y]
    m = classification_metrics(y, good)
    assert m["macf1"] == 1.0 and m["acc"] == 1.0 and m["mae"] == 0.0 and m["f1_h"] == 1.0
    skew = classification_metrics(y, np.eye(3)[np.array([0, 0, 2, 0, 0, 2])])
    assert abs(skew["f1_h"] - 2 * skew["macp"] * skew["macr"] / (skew["macp"] + skew["macr"])) < 1e-12 and skew["f1_h"] != skew["macf1"]
    bad = np.eye(3)[np.array([1, 1, 1, 1, 1, 1])]
    assert abs(classification_metrics(y, bad)["macf1"] - (2 * (1 / 3) / (1 + 1 / 3)) / 3) < 1e-9
    b = paired_bootstrap(y, [good.argmax(1)], [bad.argmax(1)], 3, n_boot=300)
    assert b["diff"] > 0.5 and b["p"] < 0.05 and b["ci_lo"] > 0
    same = paired_bootstrap(y, [good.argmax(1)], [good.argmax(1)], 3, n_boot=200)
    assert same["diff"] == 0.0 and same["p"] == 1.0


@pytest.fixture(scope="module")
def tiny():
    """A two-layer Qwen2 with random weights behind the real tokenizer."""
    if not os.path.isdir(TOKENIZER):
        pytest.skip("tokenizer not available")
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)
    return tok, Qwen2ForCausalLM(cfg)


def make_verifier(tiny, **kw):
    from openbook.verifier import Verifier

    tok, model = tiny
    import copy
    return Verifier("unused", ["false", "half true", "true"], lora_rank=4, max_tokens=kw.pop("max_tokens", 400), device="cpu",
                    model=copy.deepcopy(model), tokenizer=tok, lora_targets=["q_proj", "v_proj"], **kw)


def test_evidence_positions(tiny, corpus, ranking):
    v = make_verifier(tiny, evidence_head=True)
    exs = V.standard(corpus.split("train")[:3], ranking["strict"], 5)
    for ex, e in zip(exs, v.encode(exs)):
        assert e["n_ev"] == 5 and len(e["ev_pos"]) == 5 and e["ev_pos"] == sorted(e["ev_pos"])
        for (host, sent), p in zip(ex["evidence"], e["ev_pos"]):
            upto = v.tok.decode(e["ids"][:p + 1])
            assert upto.rstrip("\n").endswith(sent), (sent, upto[-60:])      # read-out sits on the sentence end
    # truncation drops whole sentences from the end and keeps the question
    v.max_tokens = 230
    e = v.encode(exs[:1])[0]
    assert 0 < e["n_ev"] < 5 and len(e["ids"]) <= 230
    assert v.tok.decode(e["ids"]).rstrip().endswith("assistant")
    assert make_verifier(tiny).encode(V.closed(corpus.split("train")[:1]))[0]["ev_pos"] == []


def test_training_and_two_pass(tiny, corpus, ranking):
    from openbook.verifier import fit

    v = make_verifier(tiny, evidence_head=True)
    train, val, r = corpus.split("train"), corpus.split("val"), ranking["strict"]
    yv = np.array([it["label"] for it in val])
    rng = np.random.RandomState(0)
    before = {k: x.clone() for k, x in v.state().items()}
    calls = []

    def score():
        out = P.read_select_verify(v, val, r, 5, 10, 3, batch=4)
        calls.append(1)
        return classification_metrics(yv, out["prob"])["macf1"]

    s = fit(v, lambda ep: V.training(train, r, 5, rng, top=10, swap_rate=0.25), score, epochs=2, lr=1e-3,
            micro_batch=4, accum=2, seed=0, lambda_ev=1.0, log=lambda m: None)
    assert len(s["history"]) == 2 == len(calls) and s["best_epoch"] in (0, 1)
    assert s["history"][1]["loss_ev"] < s["history"][0]["loss_ev"]                # the evidence head learns something
    assert any(not torch.equal(before[k], x) for k, x in v.state().items())
    out = P.read_select_verify(v, val, r, 5, 10, 3, batch=4)
    assert out["prob"].shape == (9, 3) and np.allclose(out["prob"].sum(1), 1.0, atol=1e-5)
    assert out["prob_read"].shape == (9, 3) and out["queries"] == 3.0
    for it, sel in zip(val, out["selected"]):
        sc = out["scores"][it["id"]]
        assert len(sel) == 5 and len(sc) == 10
        assert min(sc[c] for c in sel) >= max(x for c, x in sc.items() if c not in sel) - 1e-9   # the best five
    one = P.single_pass(v, V.standard(val, r, 5), 3, batch=4)
    assert all(len(sc) == 5 for sc in one["scores"]) and 0 <= P.silver_precision(val, out["selected"]) <= 1
    assert P.read_ensemble(v, val, r, 5, 10, 3, batch=4).shape == (9, 3)
    # diagnostics run for both inference procedures
    for select in (True, False):
        run = (lambda items, rk: P.read_select_verify(v, items, rk, 5, 10, 3, batch=4)) if select else \
              (lambda items, rk: P.single_pass(v, V.standard(items, rk, 5), 3, batch=4))
        pert = {}
        for kind in ("swap", "halfswap"):
            items_p, rank_p = V.perturb(val, r, {"halfswap": "mix"}.get(kind, kind), 5, 10, seed=0)
            pert[kind] = (items_p, run(items_p, rank_p))
        d = P.evidence_diagnostics(val, r, 5, run(val, r), pert, select)
        assert all(0.0 <= d[k] <= 1.0 for k in ("auc_silver", "silver_p_pool", "silver_p_bm25", "auc_own_vs_foreign", "auc_usable"))
        assert ("halfswap_foreign_sel" in d) == select == ("silver_p_selected" in d)
        assert len(P.score_maps(val, run(val, r))[0]) == (10 if select else 5)
    # restoring a state reproduces predictions
    st = v.state()
    p1 = P.single_pass(v, V.standard(val, r, 5), 3, batch=4)["prob"]
    v.load_state(st)
    assert np.allclose(p1, P.single_pass(v, V.standard(val, r, 5), 3, batch=4)["prob"], atol=1e-6)


def test_soft_targets_and_no_head(tiny, corpus, ranking):
    from openbook.verifier import fit

    v = make_verifier(tiny)
    train = V.standard(corpus.split("train")[:8], ranking["all"], 4)
    for ex in train:
        ex["soft"] = [0.2, 0.3, 0.5]
    s = fit(v, lambda ep: train, lambda: 0.0, epochs=1, lr=1e-3, micro_batch=4, accum=1, seed=0, log=lambda m: None)
    assert np.isfinite(s["history"][0]["loss"]) and s["history"][0]["loss_ev"] == 0.0
    out = P.single_pass(v, train, 3, batch=4)
    assert all(len(x) == 0 for x in out["scores"]) and P.score_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
