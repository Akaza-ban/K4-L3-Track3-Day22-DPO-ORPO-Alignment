"""Verifiable math reward for GRPO (NB7).

Vietnamese writes "1.440" for 1440 and "2,5" for 2.5, the opposite of English.
A model answering in Vietnamese may use either convention, so a number is read
every way that is plausible and counts as correct if any reading matches.
"""

from __future__ import annotations

import contextlib
import re

ANSWER_MARKER = "Đáp số:"
NUM = re.compile(r"-?\d[\d.,]*")
_GROUPED = {sep: re.compile(rf"-?\d{{1,3}}(\{sep}\d{{3}})+") for sep in ".,"}


def readings(token: str) -> set[float]:
    """All plausible numeric values of a token such as '1.440', '2,5', '1,234.5'."""
    token = token.rstrip(".,")
    out: set[float] = set()
    if "." in token and "," in token:
        # Both separators: the later one is the decimal point.
        dec = "." if token.rfind(".") > token.rfind(",") else ","
        grp = "," if dec == "." else "."
        candidates = [token.replace(grp, "").replace(dec, ".")]
    else:
        sep = "." if "." in token else ("," if "," in token else "")
        candidates = [token]
        if sep:
            if _GROUPED[sep].fullmatch(token):
                candidates.append(token.replace(sep, ""))  # thousands grouping
            if token.count(sep) == 1:
                candidates.append(token.replace(sep, "."))  # decimal separator
    for c in candidates:
        with contextlib.suppress(ValueError):
            out.add(float(c))
    return out


def extract_answer(text: str) -> str | None:
    """Last number after the final 'Đáp số:' marker, else the last number in the text."""
    tail = text.rsplit(ANSWER_MARKER, 1)[-1]
    nums = NUM.findall(tail)
    return nums[-1] if nums else None


def is_correct(pred: str | None, ref: str) -> bool:
    if pred is None:
        return False
    # The reference is canonical ("0.125", "1440"), so read it strictly; only the
    # model's free-form answer gets the lenient Vietnamese/English readings.
    try:
        refs = {float(str(ref).strip())}
    except ValueError:
        refs = readings(str(ref))
    return any(abs(p - r) < 1e-6 for p in readings(pred) for r in refs)


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    """TRL GRPO reward: 2.0 when the final number matches the reference."""
    return [2.0 if is_correct(extract_answer(c[0]["content"]), a) else 0.0 for c, a in zip(completions, answer)]


def format_reward(completions, **kwargs) -> list[float]:
    """TRL GRPO reward: 0.5 when the answer ends with an 'Đáp số: <number>' line."""
    return [0.5 if re.search(rf"{ANSWER_MARKER}\s*-?\d", c[0]["content"]) else 0.0 for c in completions]
