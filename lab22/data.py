"""Preference-data loading, held-out splitting, and length diagnostics.

The pure helpers (`message_text`, `to_conversational`, `split_by_prompt`,
`length_stats`) run without a GPU and are covered by `scripts/test_lab22.py`.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any


def message_text(value: Any) -> str:
    """Return the assistant text from a string or a list of chat messages.

    UltraFeedback-style rows store `chosen`/`rejected` as the whole
    conversation; the assistant reply is the last message.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and value:
        last = value[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
    raise ValueError(f"Cannot extract a response from {type(value).__name__}: {value!r:.80}")


def prompt_text(row: dict) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    # Some sets only carry the prompt as the first user turn of `chosen`.
    chosen = row.get("chosen")
    if isinstance(chosen, Sequence) and chosen and isinstance(chosen[0], dict) and chosen[0].get("role") == "user":
        return str(chosen[0]["content"])
    raise ValueError("Row has no usable prompt")


def to_conversational(row: dict) -> dict:
    """Map a raw row to TRL's conversational preference format.

    TRL applies the tokenizer's chat template itself, so the data never
    contains template tokens and cannot be double-templated.
    """
    return {
        "prompt": [{"role": "user", "content": prompt_text(row).strip()}],
        "chosen": [{"role": "assistant", "content": message_text(row["chosen"]).strip()}],
        "rejected": [{"role": "assistant", "content": message_text(row["rejected"]).strip()}],
    }


def is_usable_pair(pair: dict, min_chars: int = 2) -> bool:
    chosen = pair["chosen"][0]["content"]
    rejected = pair["rejected"][0]["content"]
    prompt = pair["prompt"][0]["content"]
    return (
        len(prompt) >= min_chars
        and len(chosen) >= min_chars
        and len(rejected) >= min_chars
        and chosen != rejected
    )


def normalize_prompt(text: str) -> str:
    return " ".join(text.lower().split())


def split_by_prompt(
    pairs: Iterable[dict], n_train: int, n_eval: int, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Shuffle, then split so that no prompt appears in both train and eval.

    Pairs that share a prompt are kept together on one side of the split.
    The original lab used the last 50 rows of the training slice as the
    eval set, so every eval prompt had been trained on.
    """
    groups: dict[str, list[dict]] = {}
    for pair in pairs:
        groups.setdefault(normalize_prompt(pair["prompt"][0]["content"]), []).append(pair)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    eval_rows: list[dict] = []
    train_rows: list[dict] = []
    for key in keys:
        if len(eval_rows) < n_eval:
            eval_rows.extend(groups[key])
        elif len(train_rows) < n_train:
            train_rows.extend(groups[key])
        else:
            break
    return train_rows[:n_train], eval_rows[:n_eval]


def assert_disjoint(train_rows: list[dict], eval_rows: list[dict]) -> None:
    train_prompts = {normalize_prompt(r["prompt"][0]["content"]) for r in train_rows}
    leaked = [r for r in eval_rows if normalize_prompt(r["prompt"][0]["content"]) in train_prompts]
    if leaked:
        raise AssertionError(f"{len(leaked)} eval prompts also appear in train")


def length_stats(rows: list[dict], count: Callable[[str], int] = len) -> dict[str, float]:
    """Length bias of the chosen side. `count` can be a token counter."""
    if not rows:
        return {"n": 0}
    chosen = [count(r["chosen"][0]["content"]) for r in rows]
    rejected = [count(r["rejected"][0]["content"]) for r in rows]
    longer = sum(c > r for c, r in zip(chosen, rejected))
    return {
        "n": len(rows),
        "chosen_median": float(statistics.median(chosen)),
        "rejected_median": float(statistics.median(rejected)),
        "chosen_longer_frac": longer / len(rows),
    }


def fits_budget(pair: dict, tokenizer: Any, max_len: int, template_kwargs: dict | None = None) -> bool:
    """True when prompt + chosen and prompt + rejected both fit in `max_len` tokens.

    Filtering is better than silent truncation here: a truncated chosen
    answer can end mid-sentence and teach the opposite of what was labelled.
    Both sides are tokenized because character length is a poor proxy for
    token length across Vietnamese, code, and numbers.
    """
    kwargs = template_kwargs or {}
    for side in ("chosen", "rejected"):
        text = tokenizer.apply_chat_template(pair["prompt"] + pair[side], tokenize=False, **kwargs)
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) > max_len:
            return False
    return True


def load_preference_pairs(
    dataset_id: str,
    tokenizer: Any,
    max_len: int,
    n_train: int,
    n_eval: int,
    language: str | None = None,
    seed: int = 42,
    template_kwargs: dict | None = None,
):
    """Load, filter, and split a preference set into TRL-ready `Dataset`s."""
    from datasets import Dataset, load_dataset

    raw = load_dataset(dataset_id, split="train")
    if language and "language" in raw.column_names:
        raw = raw.filter(lambda r: r["language"] == language)
    raw = raw.shuffle(seed=seed)

    pairs = []
    for row in raw:
        try:
            pair = to_conversational(row)
        except (KeyError, ValueError):
            continue
        if is_usable_pair(pair) and fits_budget(pair, tokenizer, max_len, template_kwargs):
            pairs.append(pair)
        if len(pairs) >= 3 * (n_train + n_eval):
            break

    train_rows, eval_rows = split_by_prompt(pairs, n_train, n_eval, seed)
    assert_disjoint(train_rows, eval_rows)
    if len(train_rows) < n_train or len(eval_rows) < n_eval:
        print(
            f"WARNING: only {len(train_rows)} train / {len(eval_rows)} eval pairs fit "
            f"max_len={max_len}; requested {n_train} / {n_eval}."
        )
    # TRL (DPO and ORPO) reads `chat_template_kwargs` per example, so training
    # renders the same template (e.g. enable_thinking=False) as generation.
    if template_kwargs:
        train_rows = [{**r, "chat_template_kwargs": dict(template_kwargs)} for r in train_rows]
        eval_rows = [{**r, "chat_template_kwargs": dict(template_kwargs)} for r in eval_rows]
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


SPLIT_FILE = "split.json"


def split_fingerprint(pref_dir: Path) -> dict[str, str]:
    """SHA-256 of the train/eval Parquets, saved next to every adapter trained on them."""
    return {name: hashlib.sha256((pref_dir / f"{name}.parquet").read_bytes()).hexdigest() for name in ("train", "eval")}


def save_split_fingerprint(pref_dir: Path, adapter_dir: Path) -> None:
    (adapter_dir / SPLIT_FILE).write_text(json.dumps(split_fingerprint(pref_dir), indent=2))


def split_mismatch(pref_dir: Path, adapter_dir: Path) -> str | None:
    """Why the adapter's training split differs from the current one, or None.

    NB2 overwrites the Parquets, so rerunning it (e.g. with another SEED) after
    NB3 could move trained prompts into "held-out" without this check.
    """
    path = adapter_dir / SPLIT_FILE
    if not path.exists():
        return f"{path} is missing: retrain the adapter (NB3)"
    if json.loads(path.read_text()) != split_fingerprint(pref_dir):
        return "data/pref changed after this adapter was trained: rerun NB3 (or restore the NB2 split)"
    return None
