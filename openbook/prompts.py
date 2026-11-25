"""Prompts of the evidence-grounded verifier.

The verifier answers with a single option letter; its class distribution is read from the logits
of the option letters at the first answer position, so no text is generated at inference time.
"""
from typing import List, Optional, Sequence, Tuple

LETTERS = "ABCDEFGH"
#: what the verifier is shown instead of the claim in the evidence-only control
NO_CLAIM = "(the claim is withheld)"

SYSTEM = ("You are an experienced fact-checker. You are given a claim and sentences retrieved from news reports. "
          "Judge how truthful the claim is. Rely on the evidence where it addresses the claim; "
          "the evidence may be incomplete, noisy or irrelevant.")
SYSTEM_CLOSED = ("You are an experienced fact-checker. You are given a claim without any evidence. "
                 "Judge how truthful the claim is from the claim itself and your own knowledge.")


def format_evidence(evidence: Sequence[Tuple[str, str]], max_words: int = 60) -> str:
    """``evidence`` is a list of (source host, sentence); long sentences are cut."""
    lines = []
    for i, (host, text) in enumerate(evidence, 1):
        w = text.split()
        if len(w) > max_words:
            text = " ".join(w[:max_words]) + " ..."
        lines.append("[{}] ({}) {}".format(i, host or "unknown source", text))
    return "\n".join(lines)


def verifier_prompt(claim: str, evidence: Optional[Sequence[Tuple[str, str]]], options: Sequence[str],
                    max_words: int = 60) -> List[dict]:
    """Chat messages of one verification query.  ``evidence=None`` gives the closed-book variant
    (no evidence section at all); an empty list means that retrieval returned nothing."""
    opts = "\n".join("{}. {}".format(LETTERS[i], o) for i, o in enumerate(options))
    if evidence is None:
        body = "Claim: {}\n\nHow truthful is the claim?\n{}\nAnswer with a single letter.".format(claim, opts)
        return [{"role": "system", "content": SYSTEM_CLOSED}, {"role": "user", "content": body}]
    ev = format_evidence(evidence, max_words) if evidence else "(no evidence was retrieved)"
    body = ("Claim: {}\n\nEvidence:\n{}\n\nHow truthful is the claim?\n{}\nAnswer with a single letter."
            .format(claim, ev, opts))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]
