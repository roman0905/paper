"""LLM verifier: a causal LM that answers a verification query with one option letter.

The class distribution is the softmax over the logits of the option letters at the first answer
position.  With LoRA adapters the model is fine-tuned by cross-entropy on that distribution;
without training the same read-out gives the zero-shot baseline, so that the two differ in
nothing but the fine-tuning.

Optionally the verifier carries an *evidence head*: a linear read-out of the hidden state at the
last token of every evidence sentence that scores how useful the sentence is for the verdict.
Because the model is causal and the claim precedes the evidence in the prompt, that state has seen
the claim and the sentence, so the scores come out of the same forward pass as the verdict.
"""
import bisect
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from openbook.prompts import LETTERS, verifier_prompt

Example = Dict[str, Any]
Encoded = Dict[str, Any]          # {"ids": [int], "ev_pos": [int], "n_ev": int}


class Verifier:
    def __init__(self, model_path: str, options: Sequence[str], lora_rank: int = 16, max_tokens: int = 1024,
                 device: str = "cuda", model=None, tokenizer=None, lora_targets: Optional[Sequence[str]] = None,
                 grad_checkpoint: bool = False, evidence_head: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.options, self.max_tokens, self.device = list(options), max_tokens, device
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_path)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
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

    # ------------------------------------------------------------------------------- encoding
    def _render(self, ex: Example, n_evidence: Optional[int]):
        ev = ex["evidence"]
        if ev is not None and n_evidence is not None:
            ev = ev[:n_evidence]
        msgs = verifier_prompt(ex["claim"], ev, self.options)
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), msgs[-1]["content"], ev

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

    def encode(self, examples: Sequence[Example]) -> List[Encoded]:
        """Token ids of every query (and the position of the last token of every evidence sentence);
        evidence is dropped from the end until a query fits into ``max_tokens`` - the claim and the
        question are never cut."""
        out = []
        for ex in examples:
            n = len(ex["evidence"]) if ex["evidence"] is not None else None
            while True:
                e = self._encode_one(ex, n)
                if len(e["ids"]) <= self.max_tokens or not n:
                    break
                n = max(0, n - max(1, n // 8))
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
        """Option logits [B, n_options] at the answer position and, with an evidence head, the
        logits of all evidence sentences of the batch (flat, in order) - otherwise ``None``."""
        ids, att, rows, cols = self._batch(encoded)
        body = getattr(self.backbone, self.backbone.base_model_prefix)
        if self.amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = body(input_ids=ids, attention_mask=att).last_hidden_state
        else:
            h = body(input_ids=ids, attention_mask=att).last_hidden_state
        logits = h[:, -1, :].float() @ self.head.T
        ev = None
        if self.ev_head is not None and rows:
            ev = self.ev_head(h[rows, cols].float()).squeeze(-1)
        return logits, ev

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
            logits, ev = self.forward([encoded[i] for i in idx])
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
        lambda_ev: float = 0.0, lr_head: float = 1e-3, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Fine-tune the adapters (and the evidence head).

    ``train_fn(epoch)`` returns the training examples of an epoch, so that evidence can be
    re-sampled; an example may carry ``"tags"`` (0/1 per evidence sentence) for the evidence loss
    and ``"soft"`` (a target distribution over the options) to replace its one-hot label.
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
        enc = v.encode(examples)
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
        run, run_ev, nb = 0.0, 0.0, 0
        opt.zero_grad(set_to_none=True)
        for b, idx in enumerate(batches):
            logits, ev = v.forward([enc[i] for i in idx])
            loss = -(target[idx].to(logits.device) * F.log_softmax(logits, -1)).sum(-1).mean()
            run += float(loss)
            if ev is not None and lambda_ev > 0:
                tags = [t for i in idx for t in (examples[i].get("tags") or [0] * enc[i]["n_ev"])[:enc[i]["n_ev"]]]
                l_ev = F.binary_cross_entropy_with_logits(ev, torch.tensor(tags, dtype=torch.float32, device=ev.device))
                run_ev += float(l_ev)
                loss = loss + lambda_ev * l_ev
            (loss / accum).backward()
            nb += 1
            if (b + 1) % accum == 0 or b + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(v.params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if nb % 400 == 0:
                log("  ep {} step {}/{} loss {:.4f} ev {:.4f} {:.0f}s".format(
                    epoch, nb, len(batches), run / nb, run_ev / nb, time.time() - t0))
        score = evaluate()
        history.append({"epoch": epoch, "val": score, "loss": run / max(nb, 1), "loss_ev": run_ev / max(nb, 1),
                        "n_train": len(examples)})
        log("epoch {} loss {:.4f} ev {:.4f} val {:.4f} ({:.0f}s)".format(
            epoch, run / max(nb, 1), run_ev / max(nb, 1), score, time.time() - t0))
        if score > best:
            best, best_epoch, best_state = score, epoch, v.state()
    if best_state is not None:
        v.load_state(best_state)
    return {"best_epoch": best_epoch, "best_val": best, "epochs_run": epochs, "history": history,
            "train_seconds": time.time() - t0}
