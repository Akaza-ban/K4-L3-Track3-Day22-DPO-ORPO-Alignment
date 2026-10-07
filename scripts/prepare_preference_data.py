#!/usr/bin/env python3
"""Build data/pref/{train,eval}.parquet from the command line (same as NB2).

Usage:
    python scripts/prepare_preference_data.py
    PREF_DATASET=argilla/ultrafeedback-binarized-preferences-cleaned PREF_LANGUAGE= \
        python scripts/prepare_preference_data.py

Settings come from lab22/config.py and the matching environment variables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    from transformers import AutoTokenizer

    from lab22 import config as C
    from lab22 import data as D

    C.ensure_dirs()
    print(C.summary())
    tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    train_ds, eval_ds = D.load_preference_pairs(
        C.PREF_DATASET,
        tokenizer,
        max_len=C.MAX_LEN,
        n_train=C.PREF_TRAIN,
        n_eval=C.PREF_EVAL,
        language=C.PREF_LANGUAGE or None,
        seed=C.SEED,
        template_kwargs=C.CHAT_TEMPLATE_KWARGS,
    )
    D.assert_disjoint(list(train_ds), list(eval_ds))
    stats = D.length_stats(list(train_ds), count=lambda t: len(tokenizer(t, add_special_tokens=False)["input_ids"]))
    train_ds.to_parquet(str(C.PREF_DIR / "train.parquet"))
    eval_ds.to_parquet(str(C.PREF_DIR / "eval.parquet"))
    (C.PREF_DIR / "stats.json").write_text(
        json.dumps({"dataset": C.PREF_DATASET, "language": C.PREF_LANGUAGE, **stats}, ensure_ascii=False, indent=2)
    )
    print(f"train={len(train_ds)} eval={len(eval_ds)} chosen longer in {stats['chosen_longer_frac']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
