"""LLM verifier: a causal LM that answers a verification query with one option letter.

The class distribution is the softmax over the logits of the option letters at the first answer
position.  With LoRA adapters the model is fine-tuned by cross-entropy on that distribution;
without training the same read-out gives the zero-shot baseline, so that the two differ in
nothing but the fine-tuning.

Optionally the verifier carries an *evidence head*: a linear read-out of the hidden state at the
last token of every evidence sentence that scores how useful the sentence is for the verdict.
Because the model is causal and the claim precedes the evidence in the prompt, that state has seen
the claim and the sentence, so the scores come out of the same forward pass as the verdict.

Optionally the verifier is trained with an *explanation target*: in training, the answer letter is
followed by the conclusion of the fact-checker's justification of the claim (its last sentences),
and the language-model loss on those tokens is added.  The verdict is still read at the first answer position, which
cannot attend to anything that follows it, so inference is unchanged and never sees or produces
the justification unless it is asked to (:meth:`Verifier.justify`).
"""
import bisect
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from openbook.prompts import LETTERS, verifier_prompt

Example = Dict[str, Any]
Encoded = Dict[str, Any]          # {"ids": [int], "ev_pos": [int], "n_ev": int}

_URL = re.compile(r"https?://\S+")
_SENT_END = re.compile(r"(?<=[.!?\"”’)])\s+(?=[A-Z0-9\"“‘(\[])")


def conclusion_sentences(text: str) -> List[str]:
    """Sentences of a justification text, links removed (a light splitter: the split points only
    decide where a training target may start)."""
    text = " ".join(_URL.sub(" ", text or "").split())
    return [s for s in _SENT_END.split(text) if s.strip()]


class Verifier:
    def __init__(self, model_path: str, options: Sequence[str], lora_rank: int = 16, max_tokens: int = 1024,
                 device: str = "cuda", model=None, tokenizer=None, lora_targets: Optional[Sequence[str]] = None,
                 grad_checkpoint: bool = False, evidence_head: bool = False, explain_tokens: int = 0):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.options, self.max_tokens, self.device = list(options), max_tokens, device
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_path)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        # models with an optional reasoning phase (Qwen3) must answer directly: the class
        # distribution is read at the first position after the prompt
        self.chat_kwargs = {"enable_thinking": False} if "enable_thinking" in (self.tok.chat_template or "") else {}
        if model is None:
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map={"": device})
        self.amp = device.startswith("cuda")
        if lora_rank > 0:
            from peft import LoraConfig, get_peft_model

            targets = list(lora_targets or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05,
                                                     bias="none", task_type="CAUSAL_LM", target_modules=targets))
            self.backbone = model.get_base_model()
            if grad_checkpoint:
                # trade compute for memory: activations of a layer are recomputed in the backward
                # pass (non-reentrant, so the adapters get gradients although the inputs need none)
                self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.backbone = model
            for p in model.parameters():
                p.requires_grad_(False)
        self.model = model
        ids = [self.tok.encode(LETTERS[i], add_special_tokens=False) for i in range(len(self.options))]
        assert all(len(x) == 1 for x in ids), "option letters must be single tokens: {}".format(ids)
        self.option_ids = [x[0] for x in ids]
        # rows of the (frozen) output embedding that belong to the option letters: [n_options, H]
        self.head = self.backbone.get_output_embeddings().weight[self.option_ids].detach().float()
        self.adapter_params = [p for p in self.model.parameters() if p.requires_grad]
        self.ev_head = None
        if evidence_head:
            self.ev_head = torch.nn.Linear(self.head.shape[1], 1).to(device).float()
            torch.nn.init.zeros_(self.ev_head.weight)
            torch.nn.init.constant_(self.ev_head.bias, -1.0)
        self.params = self.adapter_params + (list(self.ev_head.parameters()) if self.ev_head is not None else [])
        # explanation target: "<letter>\nJustification: <conclusion of the justification, at most explain_tokens tokens>"
        self.explain_tokens = explain_tokens
        self.explain_prefix = self.tok("\nJustification:", add_special_tokens=False)["input_ids"]
        self.lm_weight = self.backbone.get_output_embeddings().weight          # frozen, full vocabulary

    # ------------------------------------------------------------------------------- encoding
    def _render(self, ex: Example, n_evidence: Optional[int]):
        ev = ex["evidence"]
        if ev is not None and n_evidence is not None:
            ev = ev[:n_evidence]
        msgs = verifier_prompt(ex["claim"], ev, self.options)
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **self.chat_kwargs)
        return text, msgs[-1]["content"], ev

    def _encode_one(self, ex: Example, n: Optional[int]) -> Encoded:
        text, user, ev = self._render(ex, n)
        if not ev or self.ev_head is None:
            return {"ids": self.tok(text, add_special_tokens=False)["input_ids"], "ev_pos": [], "n_ev": len(ev or [])}
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        starts = [s for s, _ in enc["offset_mapping"]]
        base = text.index(user)
        block = user.index("Evidence:\n") + len("Evidence:\n")
        lines = user[block:user.index("\n\nHow truthful", block)].split("\n")
        assert len(lines) == len(ev), (len(lines), len(ev))
        pos, cursor = [], base + block
        for line in lines:
            cursor += len(line)                              # one past the last character of the line
            pos.append(bisect.bisect_right(starts, cursor - 1) - 1)
            cursor += 1                                      # the newline
        return {"ids": enc["input_ids"], "ev_pos": pos, "n_ev": len(ev)}

    def target_ids(self, ex: Example) -> List[int]:
        """Tokens that follow the answer letter in training: the justification prefix, the
        conclusion of the justification - its last complete sentences that fit into
        ``explain_tokens`` tokens (the last tokens of the final sentence if even that is longer) -
        and the end-of-turn token."""
        if not self.explain_tokens:
            return []
        sents = conclusion_sentences(ex.get("explain") or "")
        if not sents:
            return []
        body: List[int] = []
        for s in reversed(sents):
            ids = self.tok(" " + s, add_special_tokens=False)["input_ids"]
            if len(body) + len(ids) > self.explain_tokens:
                break
            body = ids + body
        if not body:
            body = self.tok(" " + sents[-1], add_special_tokens=False)["input_ids"][-self.explain_tokens:]
        return self.explain_prefix + body + [self.tok.eos_token_id]

    def target_text(self, ex: Example) -> str:
        """The justification segment used as training target, as text (reference for evaluation)."""
        ids = self.target_ids(ex)
        return self.tok.decode(ids[len(self.explain_prefix):-1]).strip() if ids else ""

    def encode(self, examples: Sequence[Example], targets: bool = False) -> List[Encoded]:
        """Token ids of every query (and the position of the last token of every evidence sentence);
        evidence is dropped from the end until a query fits into ``max_tokens`` - the claim and the
        question are never cut.

        ``targets=True`` (training with an explanation target) appends the gold answer letter and
        the explanation tokens of examples that have an explanation; ``n_prompt`` is then the
        length of the query itself, whose last token is where the verdict is read."""
        out = []
        for ex in examples:
            n = len(ex["evidence"]) if ex["evidence"] is not None else None
            while True:
                e = self._encode_one(ex, n)
                if len(e["ids"]) <= self.max_tokens or not n:
                    break
                n = max(0, n - max(1, n // 8))
            if targets:
                tgt = self.target_ids(ex)
                if tgt:
                    e["n_prompt"], e["n_target"] = len(e["ids"]), len(tgt)
                    e["ids"] = e["ids"] + [self.option_ids[ex["label"]]] + tgt
            out.append(e)
        return out

    def _batch(self, encoded: Sequence[Encoded]):
        L = max(len(e["ids"]) for e in encoded)
        ids = torch.full((len(encoded), L), self.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(encoded), L), dtype=torch.long)
        rows, cols = [], []
        for i, e in enumerate(encoded):
            off = L - len(e["ids"])
            ids[i, off:] = torch.tensor(e["ids"])
            att[i, off:] = 1
            rows += [i] * len(e["ev_pos"])
            cols += [off + p for p in e["ev_pos"]]
        return ids.to(self.device), att.to(self.device), rows, cols

    def forward(self, encoded: Sequence[Encoded]):
        """Returns (option logits [B, n_options] at the answer position; the logits of all evidence
        sentences of the batch, flat and in order, or ``None`` without an evidence head; the mean
        language-model loss over the explanation tokens of the batch, or ``None`` if there are none)."""
        ids, att, rows, cols = self._batch(encoded)
        body = getattr(self.backbone, self.backbone.base_model_prefix)
        if self.amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = body(input_ids=ids, attention_mask=att).last_hidden_state
        else:
            h = body(input_ids=ids, attention_mask=att).last_hidden_state
        L = ids.shape[1]
        # the verdict is read at the last token of the query: the last position, unless training
        # targets follow the query
        ans = [L - len(e["ids"]) + e.get("n_prompt", len(e["ids"])) - 1 for e in encoded]
        logits = h[list(range(len(encoded))), ans].float() @ self.head.T
        ev = None
        if self.ev_head is not None and rows:
            ev = self.ev_head(h[rows, cols].float()).squeeze(-1)
        lm = None
        t_rows, t_cols, t_ids = [], [], []
        for i, e in enumerate(encoded):
            n = e.get("n_target", 0)
            if n:
                first = ans[i] + 1                       # position of the (teacher-forced) answer letter
                t_rows += [i] * n
                t_cols += list(range(first, first + n))  # the state at position p predicts the token at p + 1
                t_ids += e["ids"][e["n_prompt"] + 1:e["n_prompt"] + 1 + n]
        if t_ids:
            lm_logits = h[t_rows, t_cols].to(self.lm_weight.dtype) @ self.lm_weight.T
            lm = F.cross_entropy(lm_logits.float(), torch.tensor(t_ids, device=lm_logits.device))
        return logits, ev, lm

    @torch.no_grad()
    def predict(self, encoded: Sequence[Encoded], batch_size: int = 16, token_budget: int = 12000):
        """Option probabilities [n, n_options] and, per query, the evidence scores (probabilities;
        empty arrays without an evidence head).  Batches are formed by length and capped by a
        token budget so that long queries do not exhaust the memory."""
        self.model.eval()
        order = np.argsort([len(e["ids"]) for e in encoded])
        prob = np.zeros((len(encoded), len(self.options)), dtype=np.float32)
        scores: List[np.ndarray] = [np.zeros(0, dtype=np.float32) for _ in encoded]
        s = 0
        while s < len(order):
            longest = len(encoded[order[min(s + batch_size, len(order)) - 1]]["ids"])
            b = max(1, min(batch_size, token_budget // max(longest, 1)))
            idx = order[s:s + b]
            logits, ev, _ = self.forward([encoded[i] for i in idx])
            prob[idx] = torch.softmax(logits, -1).cpu().numpy()
            if ev is not None:
                ev = torch.sigmoid(ev).cpu().numpy()
                k = 0
                for i in idx:
                    n = len(encoded[i]["ev_pos"])
                    scores[i] = ev[k:k + n]
                    k += n
            s += b
        return prob, scores

    @torch.no_grad()
    def justify(self, encoded: Sequence[Encoded], labels: Sequence[int], max_new_tokens: int = 120,
                batch_size: int = 8) -> List[str]:
        """Greedy continuation of "<query><answer letter>\\nJustification:" for verifiers trained
        with an explanation target.  ``labels`` are the answers to justify (e.g. the predictions)."""
        self.model.eval()
        seqs = [e["ids"][:e.get("n_prompt", len(e["ids"]))] + [self.option_ids[int(y)]] + self.explain_prefix
                for e, y in zip(encoded, labels)]
        out: List[str] = [""] * len(seqs)
        order = np.argsort([len(x) for x in seqs])
        s = 0
        while s < len(order):
            # the key/value cache grows with batch x length (and is large without grouped-query
            # attention), so long queries are generated in smaller batches
            longest = len(seqs[order[min(s + batch_size, len(order)) - 1]]) + max_new_tokens
            idx = order[s:s + max(1, min(batch_size, 8000 // longest))]
            s += len(idx)
            ids, att, _, _ = self._batch([{"ids": seqs[i], "ev_pos": []} for i in idx])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                gen = self.model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id)
            for i, g in zip(idx, gen[:, ids.shape[1]:]):
                out[i] = self.tok.decode(g, skip_special_tokens=True).strip()
        return out

    # --------------------------------------------------------------------------------- state
    def state(self) -> Dict[str, torch.Tensor]:
        st = {k: v.detach().cpu().clone() for k, v in self.model.named_parameters() if v.requires_grad}
        if self.ev_head is not None:
            st.update({"ev_head." + k: v.detach().cpu().clone() for k, v in self.ev_head.state_dict().items()})
        return st

    def load_state(self, state: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for k, v in self.model.named_parameters():
                if v.requires_grad:
                    v.copy_(state[k].to(v.device))
        if self.ev_head is not None:
            self.ev_head.load_state_dict({k[len("ev_head."):]: v for k, v in state.items() if k.startswith("ev_head.")})


def class_probs(option_prob: np.ndarray, n_classes: int) -> np.ndarray:
    """Distribution over the veracity classes, i.e. the option probabilities without any extra
    option, renormalised."""
    p = option_prob[:, :n_classes].astype(np.float64)
    return p / np.maximum(p.sum(1, keepdims=True), 1e-12)


def fit(v: Verifier, train_fn: Callable[[int], Sequence[Example]], evaluate: Callable[[], float],
        epochs: int = 4, lr: float = 1e-4, micro_batch: int = 2, accum: int = 8, seed: int = 1,
        lambda_ev: float = 0.0, lr_head: float = 1e-3, lambda_exp: float = 0.0,
        log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Fine-tune the adapters (and the evidence head).

    ``train_fn(epoch)`` returns the training examples of an epoch, so that evidence can be
    re-sampled; an example may carry ``"tags"`` (0/1 per evidence sentence) for the evidence loss,
    ``"soft"`` (a target distribution over the options) to replace its one-hot label and
    ``"explain"`` (a justification text) for the explanation loss of weight ``lambda_exp``.
    ``evaluate()`` returns the validation score used for model selection (higher is better); it is
    called after every epoch and the best state is restored at the end."""
    rng = np.random.RandomState(seed)
    first = list(train_fn(0))
    total = math.ceil(len(first) / (micro_batch * accum)) * epochs
    warmup = max(1, int(0.06 * total))
    groups = [{"params": v.adapter_params, "lr": lr}]
    if v.ev_head is not None:
        groups.append({"params": list(v.ev_head.parameters()), "lr": lr_head})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: (st + 1) / warmup if st < warmup else max(0.0, (total - st) / max(1, total - warmup)))
    best, best_state, best_epoch, history = -1.0, None, -1, []
    t0 = time.time()
    for epoch in range(epochs):
        examples = first if epoch == 0 else list(train_fn(epoch))
        enc = v.encode(examples, targets=lambda_exp > 0)
        n_opt = len(v.options)
        target = torch.zeros(len(examples), n_opt)
        for i, ex in enumerate(examples):
            if ex.get("soft") is not None:
                target[i] = torch.tensor(ex["soft"], dtype=torch.float32)
            else:
                target[i, ex["label"]] = 1.0
        # shuffle, then sort by length inside chunks so that a micro-batch holds similar lengths
        perm = rng.permutation(len(enc))
        chunk = micro_batch * 32
        batches = []
        for s in range(0, len(perm), chunk):
            part = sorted(perm[s:s + chunk], key=lambda i: len(enc[i]["ids"]))
            batches += [part[j:j + micro_batch] for j in range(0, len(part), micro_batch)]
        batches = [batches[i] for i in rng.permutation(len(batches))]
        v.model.train()
        run, run_ev, run_exp, nb = 0.0, 0.0, 0.0, 0
        opt.zero_grad(set_to_none=True)
        for b, idx in enumerate(batches):
            logits, ev, lm = v.forward([enc[i] for i in idx])
            loss = -(target[idx].to(logits.device) * F.log_softmax(logits, -1)).sum(-1).mean()
            run += float(loss)
            if ev is not None and lambda_ev > 0:
                tags = [t for i in idx for t in (examples[i].get("tags") or [0] * enc[i]["n_ev"])[:enc[i]["n_ev"]]]
                l_ev = F.binary_cross_entropy_with_logits(ev, torch.tensor(tags, dtype=torch.float32, device=ev.device))
                run_ev += float(l_ev)
                loss = loss + lambda_ev * l_ev
            if lm is not None and lambda_exp > 0:
                run_exp += float(lm)
                loss = loss + lambda_exp * lm
            (loss / accum).backward()
            nb += 1
            if (b + 1) % accum == 0 or b + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(v.params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if nb % 400 == 0:
                log("  ep {} step {}/{} loss {:.4f} ev {:.4f} exp {:.4f} {:.0f}s".format(
                    epoch, nb, len(batches), run / nb, run_ev / nb, run_exp / nb, time.time() - t0))
        score = evaluate()
        history.append({"epoch": epoch, "val": score, "loss": run / max(nb, 1), "loss_ev": run_ev / max(nb, 1),
                        "loss_exp": run_exp / max(nb, 1), "n_train": len(examples)})
        log("epoch {} loss {:.4f} ev {:.4f} exp {:.4f} val {:.4f} ({:.0f}s)".format(
            epoch, run / max(nb, 1), run_ev / max(nb, 1), run_exp / max(nb, 1), score, time.time() - t0))
        if score > best:
            best, best_epoch, best_state = score, epoch, v.state()
    if best_state is not None:
        v.load_state(best_state)
    return {"best_epoch": best_epoch, "best_val": best, "epochs_run": epochs, "history": history,
            "train_seconds": time.time() - t0}
