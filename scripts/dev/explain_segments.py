#!/usr/bin/env python
"""Which part of a fact-checker's justification says most about the verdict?  TRAINING split only.

For every training claim the justification is cut in several ways to the length of the explanation
target (128 tokens of the verifier's tokenizer); a bag-of-n-grams classifier predicts the label
from each cut (5-fold cross-validation inside the training split).  The result motivated using the
*conclusion* of the justification as the explanation target; nothing here touches validation or
test data, or any model output.

    python -I scripts/dev/explain_segments.py        # writes results/ob/explain_segments.json
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from openbook.data import Corpus  # noqa: E402
from openbook.verifier import conclusion_sentences  # noqa: E402

N = 128
TOKENIZER = os.environ.get("CLARA_LLM_TOKENIZER", "/media/dell/DATA/models/Qwen2.5-7B-Instruct")


def conclusion(tok, text):
    """The explanation target of openbook.verifier.Verifier.target_ids, as text."""
    sents = conclusion_sentences(text)
    body = []
    for s in reversed(sents):
        ids = tok(" " + s, add_special_tokens=False)["input_ids"]
        if len(body) + len(ids) > N:
            break
        body = ids + body
    if not body and sents:
        body = tok(" " + sents[-1], add_special_tokens=False)["input_ids"][-N:]
    return tok.decode(body)


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    out = {"tokens": N, "datasets": {}}
    for D in ("RAWFC", "LIAR-RAW"):
        items = Corpus.load(os.path.join(ROOT, "outputs", "ob", D + ".json")).split("train")
        y = np.array([it["label"] for it in items])
        ids = [tok(" " + " ".join(conclusion_sentences(it["explain"])), add_special_tokens=False)["input_ids"] for it in items]
        length = np.array([len(x) for x in ids])
        cuts = {"claim": [it["claim"] for it in items],
                "head": [tok.decode(x[:N]) for x in ids],
                "tail": [tok.decode(x[-N:]) for x in ids],
                "conclusion": [conclusion(tok, it["explain"]) for it in items],
                "full": [tok.decode(x) for x in ids]}
        row = {"n": len(items), "median_tokens": float(np.median(length)), "fits": float(np.mean(length <= N)), "macf1": {}}
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        for name, texts in cuts.items():
            clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                LogisticRegression(C=4.0, max_iter=3000))
            pred = cross_val_predict(clf, texts, y, cv=cv)
            row["macf1"][name] = float(f1_score(y, pred, average="macro"))
        row["conclusion_tokens"] = float(np.mean([len(tok(t, add_special_tokens=False)["input_ids"]) for t in cuts["conclusion"]]))
        out["datasets"][D] = row
        print(D, {k: round(100 * v, 1) for k, v in row["macf1"].items()}, "| median tokens", row["median_tokens"],
              "| fits", round(row["fits"], 3), "| mean target tokens", round(row["conclusion_tokens"], 1), flush=True)
    path = os.path.join(ROOT, "results", "ob", "explain_segments.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
