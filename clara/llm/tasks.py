"""Prompt templates, JSON schemas and output parsers for every offline LLM task.

This module is pure Python (no vLLM / torch import) so that it can be imported from both the
generation environment and the training environment.

Tasks
-----
clara        single-call claim decomposition with dual-stance arguments plus a holistic
             ("overall") instance (ours, Module 1)
td / cs      perspective-specific zero-shot CoT rationales, prompt wording follows ARG
             (Hu et al., AAAI 2024, Table 8); used to rebuild the ARG / FactGuard inputs with the
             same LLM that CLARA uses
tc           topic-content extraction, prompt taken from the FactGuard repository
zs / cot     plain zero-shot and zero-shot CoT veracity prediction (LLM-only baselines)
"""
import json
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from clara.text import number_sentences, split_sentences

CLAIM_TYPES = ["fact", "number", "quote", "opinion"]
SUSP_LEVELS = [1, 2, 3, 4, 5]

# --------------------------------------------------------------------------------------
# CLARA: structured dual-stance claim analysis
# --------------------------------------------------------------------------------------

CLARA_SYSTEM = {
    "zh": "你是一名严谨的事实核查分析员。你无法联网检索，只能依据消息文本本身以及你已有的常识和背景知识进行分析。",
    "en": "You are a rigorous fact-checking analyst. You cannot search the web; rely only on the "
          "message itself and on your own commonsense and background knowledge.",
}

CLARA_USER = {
    "zh": """下面是一条已按句编号的新闻或社交媒体消息。请对它做“声明分解 + 正反论证”分析，只输出一个 JSON 对象。

【步骤】
0. 先不做分析，凭第一印象填写 first_impression：整条消息是虚假或误导性信息的可能性，1=几乎肯定属实，2=较可能属实，3=无法判断，4=较可能不实，5=几乎肯定不实。
1. 找出消息中最核心、可被核查的原子声明（claim），按重要性排序，共 {kmin}–{k} 条，尽量覆盖消息中彼此不同的事实陈述。
   - 一条声明只陈述一件事，并且要自包含：补全主语、时间、地点，不要出现“他”“该公司”“此事”等指代。
   - 保留归属：原文若是“某人/某机构称……”，声明里必须保留“某人称”，不要改写成既成事实。
   - 不要抽取情绪宣泄、呼吁转发、提问、投票等非事实内容。
2. 对每条声明写出两条有针对性的理由，各不超过 60 字：
   - {first_key}：{first_desc}；
   - {second_key}：{second_desc}。
   理由要结合常识、背景知识、逻辑和文本内部的一致性，针对这条声明本身来写，避免“细节具体”“缺乏来源”这类对任何消息都适用的套话；不要写“所以这是真/假新闻”之类的结论。
3. 写完两条理由后，权衡哪一方更站得住脚，再给出 suspicion（该声明不属实的可能性）：
   1–5，含义同第 0 步。
4. type：fact=事件或事实陈述，number=数字或统计，quote=引述他人说法，opinion=预测、评价或观点。
5. sent：这条声明主要出自哪一句（填句子编号）。
6. 最后填写 overall（整体判断）：综合上面各条声明的分析，并考虑标题与正文是否一致、信息来源是否明确可靠、是否煽动情绪或夸大其词、叙述是否自洽，对整条消息做出评估。同样先写 {first_key}（{first_all}）和 {second_key}（{second_all}），再给出 suspicion（整条消息是虚假或误导性信息的可能性，1–5，含义同上）。

【消息】
{numbered}""",
    "en": """Below is a news article or social media message whose sentences have been numbered. Perform a "claim decomposition + two-sided argumentation" analysis and output a single JSON object only.

Steps:
0. Before any analysis, fill in first_impression from your first impression: how likely the whole message is false or misleading, 1 = almost certainly true, 2 = more likely true, 3 = cannot tell, 4 = more likely untrue, 5 = almost certainly untrue.
1. Extract the most central, checkable atomic claims of the message, ordered by importance: between {kmin} and {k} claims, covering the distinct factual assertions of the message.
   - Each claim states exactly one thing and must be self-contained: fill in the subject, time and place; do not use pronouns or references such as "he", "the company", "this".
   - Keep the attribution: if the text says "X said/claimed ...", the claim must keep "X said"; do not rewrite it as an established fact.
   - Do not extract emotional venting, calls to share, questions, or other non-factual content.
2. For each claim write two targeted arguments, each at most 40 words:
   - {first_key}: {first_desc};
   - {second_key}: {second_desc}.
   Ground the arguments in commonsense, background knowledge, logic and the internal consistency of the text, and make them specific to this claim; avoid boilerplate that would fit any message (e.g. "it gives specific details", "no source is given"). Do not state a verdict such as "so this news is real/fake".
3. Only after writing both arguments, weigh which side is stronger and give suspicion (how likely the claim is NOT true):
   1-5, same scale as in step 0.
4. type: fact = an event or factual statement, number = a figure or statistic, quote = a reported statement of someone, opinion = a prediction, evaluation or opinion.
5. sent: the number of the sentence this claim mainly comes from.
6. Finally fill in overall (overall judgement): combine your analysis of the claims above with whether headline and body agree, whether sources are explicit and reliable, whether the message is sensational or exaggerated, and whether the narrative is self-consistent, and assess the message as a whole. Again write {first_key} ({first_all}) and {second_key} ({second_all}) first, then give suspicion (how likely the whole message is false or misleading, 1-5, same scale as above).

Message:
{numbered}""",
}

_ALL_DESC = {
    "zh": {"support": "认为整条消息可信的最强理由", "refute": "认为整条消息不可信的最强理由"},
    "en": {"support": "the strongest argument that the message is credible",
           "refute": "the strongest argument that the message is not credible"},
}

_ARG_DESC = {
    "zh": {"support": "认为这条声明可信的最强理由", "refute": "认为这条声明可疑的最强理由"},
    "en": {"support": "the strongest argument that this claim is credible",
           "refute": "the strongest argument that this claim is doubtful"},
}

# character budgets used as JSON-schema maxLength (generous upper bounds, not targets)
_MAXLEN = {"zh": {"claim": 100, "arg": 120}, "en": {"claim": 320, "arg": 420}}


def arg_order(item_id: Any) -> int:
    """0 -> support is written first, 1 -> refute is written first.

    The order is balanced deterministically over item ids so that a possible recency bias of the
    LLM (the stance written last dominating the suspicion score) cannot correlate with the label.
    """
    try:
        return int(item_id) % 2
    except (TypeError, ValueError):
        return sum(ord(c) for c in str(item_id)) % 2


def min_claims(n_sents: int, max_claims: int = 6) -> int:
    """Lower bound on the number of claims, so that multi-sentence messages yield a real bag."""
    if n_sents <= 2:
        k = 1
    elif n_sents <= 4:
        k = 2
    else:
        k = 3
    return min(k, max_claims)


def clara_schema(n_sents: int, order: int, lang: str, max_claims: int = 6,
                 use_maxlen: bool = True) -> Dict[str, Any]:
    keys = ["support", "refute"] if order == 0 else ["refute", "support"]
    ml = _MAXLEN[lang]

    def s(n: int) -> Dict[str, Any]:
        d = {"type": "string", "minLength": 2}
        if use_maxlen:
            d["maxLength"] = n
        return d

    claim_props = OrderedDict()
    claim_props["claim"] = s(ml["claim"])
    claim_props["type"] = {"type": "string", "enum": CLAIM_TYPES}
    claim_props["sent"] = {"type": "integer", "enum": list(range(1, max(1, n_sents) + 1))}
    for k in keys:
        claim_props[k] = s(ml["arg"])
    claim_props["suspicion"] = {"type": "integer", "enum": SUSP_LEVELS}

    frame_props = OrderedDict()
    for k in keys:
        frame_props[k] = s(ml["arg"])
    frame_props["suspicion"] = {"type": "integer", "enum": SUSP_LEVELS}

    props = OrderedDict()
    props["first_impression"] = {"type": "integer", "enum": SUSP_LEVELS}
    props["claims"] = {
        "type": "array",
        "minItems": min_claims(n_sents, max_claims),
        "maxItems": max_claims,
        "items": {"type": "object", "properties": claim_props,
                  "required": list(claim_props.keys()), "additionalProperties": False},
    }
    props["overall"] = {"type": "object", "properties": frame_props,
                        "required": list(frame_props.keys()), "additionalProperties": False}
    return {"type": "object", "properties": props,
            "required": ["first_impression", "claims", "overall"],
            "additionalProperties": False}


def build_clara(content: str, lang: str, item_id: Any, max_claims: int = 6,
                use_maxlen: bool = True) -> Dict[str, Any]:
    sents = split_sentences(content, lang)
    order = arg_order(item_id)
    keys = ["support", "refute"] if order == 0 else ["refute", "support"]
    user = CLARA_USER[lang].format(
        k=max_claims, kmin=min_claims(len(sents), max_claims), numbered=number_sentences(sents),
        first_key=keys[0], first_desc=_ARG_DESC[lang][keys[0]],
        second_key=keys[1], second_desc=_ARG_DESC[lang][keys[1]],
        first_all=_ALL_DESC[lang][keys[0]], second_all=_ALL_DESC[lang][keys[1]],
    )
    return {
        "system": CLARA_SYSTEM[lang],
        "user": user,
        "prefix": "",
        "schema": clara_schema(len(sents), order, lang, max_claims, use_maxlen),
        "meta": {"sents": sents, "order": order, "max_claims": max_claims},
    }


_SUSP_RE = re.compile(r'"(?:suspicion|first_impression)"\s*:\s*([1-5])$')


def suspicion_distributions(tokens: List[Tuple[str, Dict[str, float]]]) -> List[List[float]]:
    """Recover the LLM's distribution over the five suspicion levels at every score position.

    ``tokens`` is the generated sequence as ``(decoded_token, {alternative_decoded: logprob})``.
    Returns one 5-way probability vector per score position (``"first_impression": d`` and every
    ``"suspicion": d``), in generation order.
    Several alternatives can map to the same digit (e.g. ``"4"`` and ``" 4"``); their mass is summed.
    """
    import math

    out: List[List[float]] = []
    acc = ""
    for tok, alts in tokens:
        cur = acc + tok
        if _SUSP_RE.search(cur) and not _SUSP_RE.search(acc):
            mass = [0.0] * 5
            for alt, lp in alts.items():
                m = _SUSP_RE.search(acc + alt)
                if m:
                    mass[int(m.group(1)) - 1] += math.exp(lp)
            z = sum(mass)
            if z <= 0:
                m = _SUSP_RE.search(cur)
                mass = [0.0] * 5
                mass[int(m.group(1)) - 1] = 1.0
                z = 1.0
            out.append([x / z for x in mass])
        acc = cur
    return out


def expected_suspicion(dist: List[float]) -> float:
    """Map a 5-way distribution over levels 1..5 to a score in [0, 1]."""
    return sum((i * p) for i, p in enumerate(dist)) / 4.0


def parse_clara(text: str, meta: Dict[str, Any],
                dists: Optional[List[List[float]]] = None) -> Optional[Dict[str, Any]]:
    """Validate the JSON and attach soft suspicion scores. Returns ``None`` if unusable."""
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("claims"), list):
        return None
    frame = obj.get("overall")
    if not isinstance(frame, dict):
        return None
    n_sents = len(meta["sents"])
    claims = []
    for c in obj["claims"]:
        if not isinstance(c, dict):
            return None
        try:
            claim = str(c["claim"]).strip()
            sup, ref = str(c["support"]).strip(), str(c["refute"]).strip()
            susp = int(c["suspicion"])
            sent = int(c["sent"])
            ctype = str(c["type"])
        except (KeyError, ValueError, TypeError):
            return None
        if not claim or susp not in SUSP_LEVELS:
            return None
        claims.append({
            "claim": claim, "type": ctype if ctype in CLAIM_TYPES else "fact",
            "sent": min(max(sent, 1), n_sents), "support": sup, "refute": ref, "suspicion": susp,
        })
    if not claims:
        return None
    try:
        fr = {"support": str(frame["support"]).strip(), "refute": str(frame["refute"]).strip(),
              "suspicion": int(frame["suspicion"])}
    except (KeyError, ValueError, TypeError):
        return None
    if fr["suspicion"] not in SUSP_LEVELS:
        return None

    try:
        first = int(obj["first_impression"])
    except (KeyError, ValueError, TypeError):
        return None
    if first not in SUSP_LEVELS:
        return None

    # soft scores: logit expectation when the positions line up, otherwise the hard integer.
    # generation order: first_impression, one suspicion per claim, overall suspicion
    soft_ok = dists is not None and len(dists) == len(claims) + 2
    if soft_ok:
        fr["dist0"] = [round(x, 5) for x in dists[0]]
        fr["v0"] = round(expected_suspicion(dists[0]), 5)
        dists = dists[1:]
    else:
        fr["v0"] = (first - 1) / 4.0
    fr["first_impression"] = first
    for i, c in enumerate(claims):
        if soft_ok:
            c["dist"] = [round(x, 5) for x in dists[i]]
            c["v"] = round(expected_suspicion(dists[i]), 5)
        else:
            c["v"] = (c["suspicion"] - 1) / 4.0
    if soft_ok:
        fr["dist"] = [round(x, 5) for x in dists[-1]]
        fr["v"] = round(expected_suspicion(dists[-1]), 5)
    else:
        fr["v"] = (fr["suspicion"] - 1) / 4.0
    return {"claims": claims, "overall": fr, "soft": bool(soft_ok)}


# --------------------------------------------------------------------------------------
# ARG-style perspective-specific rationales (td / cs) and LLM-only baselines (zs / cot)
# --------------------------------------------------------------------------------------

ARG_QUESTION = {
    "zh": "给定以下消息，预测其真实性。如果它更可能是真实消息，返回1；否则返回0。"
          "请不要给出“无法确定”之类模糊的判断：{news}",
    "en": "Given the following message, predict its veracity. If it is more likely to be a real "
          "message, return 1; otherwise, return 0. Please refrain from providing ambiguous "
          "assessments such as undetermined: {news}",
}
_ARG_TAIL = {
    "zh": "\n请用不超过150字进行分析，并在最后一句写明“返回1”或“返回0”。",
    "en": "\nAnalyse in at most 120 words and state \"Return 1\" or \"Return 0\" in the final sentence.",
}
ARG_PREFIX = {
    "td": {"zh": "让我们从文字描述的角度思考。", "en": "Let's think from the perspective of textual description."},
    "cs": {"zh": "让我们从常识的角度思考。", "en": "Let's think from the perspective of commonsense."},
    "cot": {"zh": "让我们一步一步地思考。", "en": "Let's think step by step."},
}
_GENERIC_SYSTEM = {"zh": "你是一个乐于助人的助手。", "en": "You are a helpful assistant."}


def build_arg(content: str, lang: str, perspective: str) -> Dict[str, Any]:
    return {"system": _GENERIC_SYSTEM[lang],
            "user": ARG_QUESTION[lang].format(news=content) + _ARG_TAIL[lang],
            "prefix": ARG_PREFIX[perspective][lang], "schema": None, "meta": {}}


def build_zs(content: str, lang: str) -> Dict[str, Any]:
    tail = {"zh": "\n只回答一个数字。", "en": "\nAnswer with a single digit only."}[lang]
    return {"system": _GENERIC_SYSTEM[lang], "user": ARG_QUESTION[lang].format(news=content) + tail,
            "prefix": "", "schema": None, "meta": {}}


_RET_RE = re.compile(r"(?:返回|return(?:s|ing)?|answer(?:\s*\(arabic numerals?\))?\s*(?:is|:)|答案(?:是|为|：|:))"
                     r"\s*[:：“\"'「]?\s*([01])", re.I)
_LEAD_RE = re.compile(r"^\W*([01])\b")


def parse_binary(text: str) -> str:
    """Extract the 0/1 answer (1 = real, 0 = fake, following ARG's prompt). Returns real/fake/other."""
    text = (text or "").strip()
    m = _RET_RE.findall(text)
    digit = m[-1] if m else None
    if digit is None:
        lead = _LEAD_RE.match(text)
        if lead:
            digit = lead.group(1)
    if digit is None:
        return "other"
    return "real" if digit == "1" else "fake"


def binary_soft(tokens: List[Tuple[str, Dict[str, float]]]) -> Optional[float]:
    """P(fake) from the logits of the first generated token of a 0/1 answer (1 = real, 0 = fake)."""
    import math

    for _, alts in tokens[:2]:
        m0 = sum(math.exp(lp) for a, lp in alts.items() if a.strip() == "0")
        m1 = sum(math.exp(lp) for a, lp in alts.items() if a.strip() == "1")
        if m0 + m1 > 0:
            return m0 / (m0 + m1)
    return None


def clean_rationale(text: str, prefix: str) -> str:
    """Rationale text fed to the SLM: the assistant prefix plus the generated analysis."""
    return (prefix + (text or "")).strip()


# --------------------------------------------------------------------------------------
# FactGuard topic-content extraction (prompt text from the official FactGuard repository,
# Data-Preprocessing/openai_client_{zh,en}.py)
# --------------------------------------------------------------------------------------

TC_SYSTEM = {"zh": "你是一个中文新闻主题与摘要提取助手。",
             "en": "You are an assistant that extracts structured information from text."}

TC_USER = {
    "zh": """
你是一个中文新闻信息抽取助手，请阅读下面的新闻文本，从中抽取两个要素：
1. **Topic（主题）**：用一句简洁的话概括新闻的核心事件或主要主题。
2. **Content（内容摘要）**：用简洁流畅的中文总结新闻的主要事实、背景或观点。

请严格按照以下格式输出：
Topic：...
Content：...

---
示例：
输入文本：
【媒体融合链来了中国搜索赋能版权保护】媒体融合链，版权价值链。由新华社中国搜索研发的媒体融合链区块链版权平台，近日正式上线。作为国家版权局版权保护新技术研究推广站点授牌单位，中国搜索此举旨在打造融合、协同、创新的版权新生态，助力提升媒体版权价值。

输出：
Topic：新华社中国搜索研发的区块链版权平台上线，赋能版权保护。
Content：由新华社中国搜索研发的媒体融合链区块链版权平台近日正式上线，作为国家版权局版权保护新技术研究推广站点授牌单位，中国搜索通过此举，旨在打造融合、协同、创新的版权新生态，助力提升媒体版权价值。

---
现在请处理以下新闻文本：
{content}
""",
    "en": """
You are an information extraction assistant. Your task is to read a given passage and extract two elements:
1. **Topic** – A short phrase summarizing the main event or focus.
2. **Content** – A concise description summarizing what happened.

Please follow the format strictly:
Topic: ...
Content: ...

---

### Example

Input Text:
No matter what people thought of "The Bachelorette" Season 13 finale on Monday, August 7, they must admit Rachel Lindsay's proposal dress was elegant. However, she didn't get to keep it. What viewers didn't know is how much her dress weighed or how much her engagement ring cost. 30-pound dress Viewers probably heard Rachel exclaim that the dress was heavy when Bryan Abasolo picked her up after he proposed to her. He quickly put her down before they made their way downhill from the castle. You may like Meghan Markle has no objection if Prince Harry shoots pheasant on Boxing Day Rachel did not exaggerate. The designer admitted the dress did weigh 30 pounds because of the beads. The gown Rachel wore appears in Randi Rahm.

Output:
Topic: The Bachelorette Season 13 finale and Rachel Lindsay's proposal dress
Content: The finale of Season 13 of The Bachelorette aired on August 7. Rachel Lindsay's proposal dress was elegant, but it weighed 30 pounds due to the beads. The designer, Randi Rahm, confirmed the dress's weight. The engagement ring's cost was not mentioned. After the proposal, Bryan Abasolo picked Rachel up, and she mentioned the dress's weight. He quickly put her down before they walked downhill from the castle. The dress was from Randi Rahm's collection.

---

Now, extract the topic and content for the following text:

Input Text:
{content}

Output:
Topic:
Content:
""",
}


def build_tc(content: str, lang: str) -> Dict[str, Any]:
    return {"system": TC_SYSTEM[lang], "user": TC_USER[lang].format(content=content),
            "prefix": "", "schema": None, "meta": {}}


# --------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------

TASKS = {
    # name: (max_tokens, temperature, needs_logprobs)
    "clara": (1200, 0.0, True),
    "td": (320, 0.0, False),
    "cs": (320, 0.0, False),
    "cot": (400, 0.0, False),
    "zs": (4, 0.0, True),
    "tc": (512, 0.2, False),
}


def build(task: str, content: str, lang: str, item_id: Any, **kw: Any) -> Dict[str, Any]:
    if task == "clara":
        return build_clara(content, lang, item_id, **kw)
    if task in ("td", "cs", "cot"):
        return build_arg(content, lang, task)
    if task == "zs":
        return build_zs(content, lang)
    if task == "tc":
        return build_tc(content, lang)
    raise KeyError(task)
