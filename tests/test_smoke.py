"""End-to-end smoke tests on synthetic data (CPU, a few seconds per method).

They check that every method and every CLARA ablation switch trains, evaluates and produces
well-formed outputs; they say nothing about accuracy.  Skipped if the local encoder is missing.
"""
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clara.data import Dataset  # noqa: E402
from clara.experiment import ENCODER, Session  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isdir(ENCODER["zh"]), reason="local encoder not available")

WORDS = ["北京", "上海", "专家", "表示", "研究", "发现", "网友", "爆料", "官方", "辟谣", "疫苗", "地震", "股市", "学校"]


def _text(rng, n):
    return "".join(rng.choice(WORDS) for _ in range(n)) + "。"


def _item(rng, i, split):
    y = i % 2
    k = rng.randint(1, 3)
    sents = [_text(rng, 5) for _ in range(k + 1)]

    def inst():
        v = min(1.0, max(0.0, rng.gauss(0.65 if y else 0.35, 0.2)))
        return {"support": _text(rng, 6), "refute": _text(rng, 6), "suspicion": 3, "v": v,
                "dist": [0.1, 0.2, 0.4, 0.2, 0.1]}

    claims = [dict(inst(), claim=_text(rng, 4), type=rng.choice(["fact", "number", "quote", "opinion"]),
                   sent=rng.randint(1, len(sents))) for _ in range(k)]
    overall = dict(inst(), v0=rng.random(), first_impression=3)
    rat = lambda: {"rationale": _text(rng, 8), "pred": rng.choice([0, 1, 2])}
    return {
        "id": i, "split": split, "label": y, "content": "".join(sents), "sents": sents,
        "gpt": {"td": dict(rat(), acc=rng.randint(0, 1)), "cs": dict(rat(), acc=rng.randint(0, 1))},
        "td": rat(), "cs": rat(), "tc": "Topic：" + _text(rng, 4),
        "zs": {"pred": y, "p_fake": None if i % 7 == 0 else min(0.99, max(0.01, rng.gauss(0.6 if y else 0.4, 0.2)))},
        "clara": None if i == 5 else {"claims": claims, "overall": overall, "soft": True, "order": i % 2},
    }


@pytest.fixture(scope="module")
def session():
    rng = random.Random(0)
    items = [_item(rng, i, "train") for i in range(48)]
    items += [_item(rng, 100 + i, "val") for i in range(16)]
    items += [_item(rng, 200 + i, "test") for i in range(16)]
    return Session(Dataset("toy", "zh", items, "synthetic"), device="cpu", log=lambda s: None)


FAST = {"epochs": 2, "batch_size": 16}


def test_cache_matches_live():
    """Cached hidden states equal the on-the-fly frozen forward (eval mode), across blocks.

    The cache is stored in float16, so agreement is up to its quantisation (relative 2^-11)."""
    from clara.encoder import FrozenPrefix

    rng = random.Random(1)
    a = [_text(rng, rng.randint(2, 12)) for _ in range(9)]
    b = [_text(rng, rng.randint(2, 12)) for _ in range(9)]
    outs = []
    for live in (False, True):
        st = FrozenPrefix(ENCODER["zh"], device="cpu", live=live)
        i1 = st.register(a, max_len=32)
        st.build()
        i2 = st.register(a, b, max_len=40)                 # second block: text pairs
        i3 = st.register(b, max_len=24, pad_full=True)     # padded to max_len
        st.build()
        ids = torch.as_tensor(np.concatenate([i1[:3], i2[:3], i3[:3], i1[3:5]]))
        outs.append(st.fetch(ids))
    (h0, m0), (h1, m1) = outs
    assert torch.equal(m0, m1) and h0.shape == h1.shape
    assert torch.allclose(h0, h1, rtol=2e-3, atol=2e-3), float((h0 - h1).abs().max())


def _check(res, n_test=16):
    m = res["metrics"]["test"]
    assert 0.0 <= m["macf1"] <= 1.0 and np.isfinite(m["ece"])
    assert res["n_items"]["test"] == n_test
    p = res["_preds"]["test"]["prob"]
    assert p.shape == (n_test,) and np.all(np.isfinite(p)) and p.min() >= 0 and p.max() <= 1


@pytest.mark.parametrize("method,model", [
    ("bert", {}),
    ("arg", {"rationale_source": "gpt"}),
    ("arg", {"rationale_source": "llm"}),
    ("factguard", {}),
    ("bert", {"use_zs": True}),
    ("bert", {"text_source": "claims"}),
    ("bert", {"text_source": "tc"}),
    ("factguard", {"use_zs": True}),
    ("arg_d", {}),
    ("factguard_d", {}),
])
def test_baselines(session, method, model):
    res = session.run(method, seed=1, model=model, train=FAST, teacher_train=FAST, keep=True)
    _check(res)


@pytest.mark.parametrize("model", [
    {},
    {"use_claims": False},
    {"use_overall": False},
    {"stance": "refute"}, {"stance": "support"}, {"stance": "none"},
    {"use_v": False}, {"use_v0": False}, {"v0_source": "fi"},
    {"gate": "none"}, {"gate": "instance"}, {"gate_fake": False},
    {"agg": "attention"}, {"agg": "max"}, {"agg": "mean"},
    {"use_global": False},
    {"max_claims": 1},
    {"factorised": False},
    {"factorised": False, "stance": "none"},
    {"stance": "none", "use_v": False},
    {"relevance": False},
    {"swap_rate": 0.5},
])
def test_clara_variants(session, model):
    res = session.run("clara", seed=1, model=dict(model, warmup=1), train=FAST, keep=True)
    _check(res)
    pr = res["_preds"]["test"]
    data = res["_data"]["test"]
    n_inst = int(data["off"][-1])
    assert pr["inst_p"].shape == (n_inst,) and pr["inst_g"].shape == (n_inst,)
    assert np.all((pr["inst_g"] >= 0) & (pr["inst_g"] <= 1))
    if model.get("agg", "noisy_or") == "noisy_or":
        # the bag probability must equal the noisy-OR of its instance probabilities
        off = data["off"].numpy()
        nor = np.array([1 - np.prod(1 - pr["inst_p"][off[i]:off[i + 1]]) for i in range(16)])
        assert np.allclose(nor, pr["p_bag"], atol=1e-4)


@pytest.mark.parametrize("model", [{}, {"kd_inst": 0.0}, {"no_teacher": True}, {"init_from_teacher": False}])
def test_student(session, model):
    res = session.run("clara_d", seed=1, model=model, train=FAST, teacher_train=FAST, keep=True)
    _check(res)
    data, pr = res["_data"], res["_preds"]["test"]
    # one holistic unit plus one unit per sentence
    n_units = sum(1 + len(it["sents"]) for it in session.ds.split("test"))
    assert pr["inst_p"].shape == (n_units,)
    if model.get("no_teacher"):
        assert "teacher" not in res and float(data["train"]["w_item"].sum()) == 0
    else:
        assert "teacher" in res
        assert float(data["train"]["w_item"].sum()) == 48 and float(data["train"]["w_inst"].sum()) > 0
        assert float(data["test"]["w_item"].sum()) == 0      # targets exist for training items only


def test_null_instances_fall_back_to_text(session):
    """With every relevance gate closed the prediction is exactly the text branch."""
    res = session.run("clara", seed=1, train=FAST, keep=True)
    m, data = res["_method"], res["_data"]["test"]
    idx = torch.arange(16)
    m.eval()
    last = m.mlp_r.net[-1]
    with torch.no_grad():
        w_old, b_old = last.weight.clone(), last.bias.clone()
        last.weight.zero_()
        last.bias.fill_(-50.0)                   # r -> 0 for every instance
        out = m(data, idx)
        last.weight.copy_(w_old)
        last.bias.copy_(b_old)
    assert torch.allclose(out["r"], torch.zeros_like(out["r"]), atol=1e-6)
    assert torch.allclose(out["evidence"], torch.zeros_like(out["evidence"]), atol=1e-3)
    assert torch.allclose(out["prob"], torch.sigmoid(out["s_x"] + m.bias), atol=1e-3)


def test_swap_training_targets(session):
    """Mismatch negatives: swapped items take the instances of another item and relevance target 0."""
    res = session.run("clara", seed=1, model={"swap_rate": 0.5}, train=FAST, keep=True)
    m, data = res["_method"], res["_data"]["train"]
    idx = torch.arange(32)
    m.train()
    torch.manual_seed(0)
    out = m(data, idx)
    sw = out["swapped"]
    assert 0 < int(sw.sum()) < 32
    off = data["off"]
    own_first = off[idx]                                   # first instance index of each item's own bag
    first = torch.full((32,), 10 ** 9, dtype=torch.long)
    first.scatter_reduce_(0, out["item"], out["inst"], reduce="amin", include_self=True)
    assert torch.all(first[~sw] == own_first[~sw])         # genuine items keep their own analysis
    assert torch.all(first[sw] != own_first[sw])           # swapped items do not
    loss, logs = m.loss(out, data, idx, epoch=3)
    assert torch.isfinite(loss) and "rel" in logs
    m.eval()
    with torch.no_grad():
        assert not m(data, idx)["swapped"].any()           # never at evaluation time


def test_unit_noise_eval(session):
    from clara import analysis
    from clara.report import load_preds
    import tempfile

    tmp = tempfile.mkdtemp()
    evals = [{"name": "noise40", "transform": "replace", "rate": 0.4, "seed": 0},
             {"name": "unit50", "transform": "units", "rate": 0.5, "seed": 0}]
    runs = {}
    for name, method, model in [("clara", "clara", {}), ("arg_swap", "arg", {"rationale_source": "llm", "swap_rate": 0.3})]:
        runs[name] = {1: session.run(method, seed=1, model=model, train=FAST, evals=evals,
                                      pred_path=os.path.join(tmp, "outputs", "runs", "toy", name, "s1.npz"))}
        assert runs[name][1]["n_items"]["unit50"] == 16
    orig = analysis.load_preds
    analysis.load_preds = lambda tag, name, seed: load_preds(tag, name, seed, root=tmp)
    try:
        rel = analysis.relevance_under_noise(session.ds, "toy", "clara", runs["clara"], "zh")
    finally:
        analysis.load_preds = orig
    for k in ("item/auc_r", "unit/auc_r", "unit/r_genuine", "unit/r_foreign"):
        assert 0.0 <= rel[k] <= 1.0, k
    z = load_preds("toy", "arg_swap", 1, root=tmp)
    assert z["test/useful"].shape == (16, 2)


def test_gate_targets(session):
    """In real news the gate target of a claim is 'the LLM did not suspect it'."""
    res = session.run("clara", seed=1, model={"delta": 0.05}, train=FAST, keep=True)
    m, data = res["_method"], res["_data"]["test"]
    idx = torch.arange(16)
    m.eval()
    with torch.no_grad():
        out = m(data, idx)
        target, weight = m.gate_targets(out, data, idx)
    y = data["label"][idx][out["item"]]
    v = data["v"][out["inst"]]
    is_claim = data["type"][out["inst"]] != 4
    sel = is_claim & (y == 0)
    assert torch.equal(target[sel], (v[sel] < 0.45).float())
    assert torch.all((target >= 0) & (target <= 1))
    assert torch.all(weight[is_claim & (y == 1)] <= m.cfg["eta"] + 1e-6)
    # no supervision where the LLM is undecided
    undecided = is_claim & ((v - 0.5).abs() <= 0.05)
    assert torch.all(weight[undecided] == 0)


def test_replace_noise_eval(session):
    res = session.run("clara", seed=1, train=FAST, keep=True,
                      evals=[{"name": "noise50", "transform": "replace", "rate": 0.5, "seed": 0}])
    assert "noise50" in res["metrics"] and res["n_items"]["noise50"] == 16


def test_path_evals_and_analysis(session, tmp_path, monkeypatch):
    """Perturbed test sets given as dataset files, and the analyses that consume the predictions."""
    import json

    from clara import analysis
    from clara.report import load_preds

    rng = random.Random(3)
    tam = []
    for i in range(12):
        it = _item(rng, 20_000_000 + 200 + i, "test")
        it["label"] = 1
        it["orig_id"] = 200 + i
        it["tamper_sent"] = rng.randrange(len(it["sents"]))
        tam.append(it)
    tam_path = str(tmp_path / "tamper.json")
    json.dump({"name": "toy-tamper", "lang": "zh", "llm": "synthetic", "items": tam},
              open(tam_path, "w", encoding="utf-8"), ensure_ascii=False)
    occ = [{"id": it["id"] * 100 + j, "split": "test", "label": 1, "orig_id": it["id"], "occl_sent": j,
            "content": "".join(s for k, s in enumerate(it["sents"]) if k != j),
            "sents": [s for k, s in enumerate(it["sents"]) if k != j]}
           for it in tam for j in range(len(it["sents"]))]
    occ_path = str(tmp_path / "occl.json")
    json.dump({"name": "toy-occl", "lang": "zh", "llm": None, "items": occ},
              open(occ_path, "w", encoding="utf-8"), ensure_ascii=False)

    noise = {"name": "noise40", "transform": "replace", "rate": 0.4, "seed": 0}
    specs = {"bert": ("bert", {}, [{"name": "tamper", "path": tam_path}, {"name": "occl", "path": occ_path}]),
             "clara": ("clara", {}, [{"name": "tamper", "path": tam_path}, noise]),
             "arg": ("arg", {"rationale_source": "llm"}, [noise]),
             "clara_d": ("clara_d", {}, [{"name": "tamper", "path": tam_path}])}
    runs = {}
    for name, (method, model, evals) in specs.items():
        runs[name] = {}
        for seed in (1, 2):
            res = session.run(method, seed=seed, model=model, train=FAST, teacher_train=FAST, evals=evals,
                              pred_path=str(tmp_path / "outputs" / "runs" / "toy" / name / "s{}.npz".format(seed)))
            assert res["n_items"]["tamper"] == 12 if "tamper" in res["n_items"] else True
            runs[name][seed] = res
    monkeypatch.setattr(analysis, "load_preds", lambda tag, name, seed: load_preds(tag, name, seed, root=str(tmp_path)))

    ds_t = Dataset.load(tam_path)
    loc = analysis.localisation(ds_t, "toy", {"BERT + occlusion": ("bert", runs["bert"]),
                                               "CLARA": ("clara", runs["clara"]),
                                               "CLARA-D": ("clara_d", runs["clara_d"])})
    assert loc["BERT + occlusion"]["kind"] == "occlusion" and loc["CLARA"]["kind"] == "claims"
    assert loc["CLARA-D"]["kind"] == "sentences"
    for k in ("LLM suspicion scores", "BERT + occlusion", "CLARA", "CLARA-D"):
        assert 0.0 <= loc[k]["hit@1"] <= 1.0 and loc[k]["n"] == 12
    assert 0.0 < loc["CLARA"]["chance"] < 1.0
    half = {it["id"] for it in tam[:6]}
    sub = analysis.localisation(ds_t, "toy", {"CLARA": ("clara", runs["clara"]), "BERT + occlusion": ("bert", runs["bert"]),
                                               "CLARA-D": ("clara_d", runs["clara_d"])}, only=half)
    assert sub["_n_items"] == 6 and all(sub[k]["n"] == 6 for k in ("CLARA", "BERT + occlusion", "CLARA-D"))

    det = analysis.tamper_detection(ds_t, "toy", "clara", runs["clara"])
    assert 0.0 <= det["pairwise_acc"] <= 1.0
    gate = analysis.gate_analysis(session.ds, "toy", "clara", runs["clara"])
    assert "overall/auc_gate" in gate and "real_claims/llm_acc" in gate
    gn = analysis.gate_under_noise("toy", "clara", runs["clara"], "clara", rates=(40,))
    assert 0.0 <= gn["clean"] <= 1.0 and 0.0 <= gn["corrupted"] <= 1.0
    gn = analysis.gate_under_noise("toy", "arg", runs["arg"], "pair", rates=(40,))
    assert 0.0 <= gn["clean"] <= 1.0
    sel = analysis.selective_summary("toy", "clara", runs["clara"])
    assert "maxprob" in sel and "selector" in sel and 0.0 <= sel["selector"]["aurc"] <= 1.0
    cmp_ = analysis.compare("toy", runs["clara"], "clara", runs["bert"], "bert")
    assert "bootstrap" in cmp_ and 0.0 <= cmp_["bootstrap"]["p"] <= 1.0
    grid, curve = analysis.risk_coverage_curve("toy", "clara", runs["clara"], kind="selector")
    assert len(grid) == len(curve) and np.all(np.isfinite(curve))
    assert "alpha" in runs["clara"][1]["scalars"]
