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
        "td": rat(), "cs": rat(), "tc": "Topic：" + _text(rng, 4), "zs": {"pred": y, "p_fake": 0.5},
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
    {"use_v": False}, {"use_v0": False},
    {"gate": "none"}, {"gate": "instance"}, {"gate_fake": False},
    {"agg": "attention"}, {"agg": "max"}, {"agg": "mean"},
    {"use_global": False},
    {"max_claims": 1},
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


def test_replace_noise_eval(session):
    res = session.run("clara", seed=1, train=FAST, keep=True,
                      evals=[{"name": "noise50", "transform": "replace", "rate": 0.5, "seed": 0}])
    assert "noise50" in res["metrics"] and res["n_items"]["noise50"] == 16
