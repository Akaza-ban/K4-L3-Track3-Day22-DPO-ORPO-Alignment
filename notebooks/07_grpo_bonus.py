# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB7 — GRPO với reward kiểm chứng được (RLVR) (BONUS, +8)
#
# DPO học từ *cặp* preference có sẵn (offline). GRPO (DeepSeekMath, 2024; dùng trong
# DeepSeek-R1) sinh **G câu trả lời cho mỗi prompt**, chấm bằng hàm reward, rồi đẩy
# xác suất các câu có reward cao hơn trung bình nhóm. Không cần reward model hay
# critic: với toán, reward là "đáp số đúng hay sai" (RLVR, *verifiable rewards*).
#
# Notebook này chạy một vòng GRPO rất nhỏ trên GSM8K để thấy cơ chế, **không** để
# đạt điểm cao. T4: ~40 phút cho 60 bước với G=4.
#
# | | DPO (NB3) | GRPO (NB7) |
# |---|---|---|
# | Dữ liệu | cặp chosen/rejected cố định | chỉ cần prompt + cách chấm |
# | Sinh trong lúc train | không | có (on-policy) |
# | Reference / KL | bắt buộc (β) | tuỳ chọn; TRL mặc định `beta=0.0` |
# | Chi phí | 2 forward / cặp | G lần sinh / prompt |

# %%
import re
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lab22" / "config.py").exists())
sys.path.insert(0, str(ROOT))

import unsloth  # noqa: F401
import torch

from lab22 import config as C
from lab22 import modeling as MD

assert torch.cuda.is_available()
assert C.SFT_MERGED.exists(), "Run NB1 first"
C.ensure_dirs()

BIG = C.COMPUTE_TIER == "BIGGPU"
N_TRAIN, N_TEST, MAX_STEPS, G = (2000, 200, 200, 8) if BIG else (400, 100, 60, 4)

# %% [markdown]
# ## 1. Dữ liệu + hàm reward

# %%
from datasets import load_dataset

INSTRUCTION = "Giải bài toán sau. Suy luận ngắn gọn, rồi kết thúc bằng dòng 'Đáp số: <số>'.\n\n"


def gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def to_row(r):
    return {"prompt": [{"role": "user", "content": INSTRUCTION + r["question"]}], "answer": gold(r["answer"])}


gsm = load_dataset("openai/gsm8k", "main")
train_ds = gsm["train"].shuffle(seed=C.SEED).select(range(N_TRAIN)).map(to_row, remove_columns=["question"])
test_ds = gsm["test"].select(range(N_TEST)).map(to_row, remove_columns=["question"])

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_number(text: str) -> str | None:
    tail = text.split("Đáp số:")[-1] if "Đáp số:" in text else text
    nums = NUM.findall(tail)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def same_number(pred: str | None, ref: str) -> bool:
    try:
        return pred is not None and abs(float(pred) - float(ref)) < 1e-6
    except ValueError:
        return False


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    return [2.0 if same_number(extract_number(c[0]["content"]), a) else 0.0 for c, a in zip(completions, answer)]


def format_reward(completions, **kwargs) -> list[float]:
    return [0.5 if re.search(r"Đáp số:\s*-?\d", c[0]["content"]) else 0.0 for c in completions]


assert same_number(extract_number("... vậy\nĐáp số: 1,250"), "1250")
assert correctness_reward([[{"content": "Đáp số: 7"}]], ["8"]) == [0.0]

# %% [markdown]
# ## 2. Accuracy trước khi train

# %%
def accuracy(model, tokenizer) -> float:
    prompts = [r["prompt"][0]["content"] for r in test_ds]
    outs = MD.generate(model, tokenizer, prompts, max_new_tokens=320)
    return sum(same_number(extract_number(o), r["answer"]) for o, r in zip(outs, test_ds)) / len(test_ds)


model, tokenizer = MD.load_model(C.SFT_MERGED)
acc_before = accuracy(model, tokenizer)
print(f"GSM8K[:{N_TEST}] accuracy before GRPO: {acc_before:.3f}")
model = MD.add_lora(model)

# %% [markdown]
# ## 3. GRPO
#
# `per_device_train_batch_size × grad_accum` phải chia hết cho `num_generations` (G):
# mỗi batch chứa trọn các nhóm G câu trả lời của cùng một prompt.

# %%
from trl import GRPOConfig, GRPOTrainer

args = GRPOConfig(
    output_dir=str(C.ADAPTERS / "grpo-checkpoints"),
    per_device_train_batch_size=G,
    gradient_accumulation_steps=1,
    num_generations=G,
    max_completion_length=320,
    max_steps=MAX_STEPS,
    learning_rate=5e-6,
    warmup_steps=0.1,
    lr_scheduler_type="cosine",
    temperature=1.0,
    chat_template_kwargs=C.CHAT_TEMPLATE_KWARGS,
    logging_steps=5,
    save_strategy="no",
    optim="adamw_8bit",
    seed=C.SEED,
    report_to="none",
    **MD.precision_flags(),
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[correctness_reward, format_reward],
    args=args,
    train_dataset=train_ds,
    processing_class=tokenizer,
)
trainer.train()

# %% [markdown]
# ## 4. Reward curve + accuracy sau train (deliverable `08-grpo-reward.png`)

# %%
import json

import matplotlib.pyplot as plt
import pandas as pd

logs = pd.DataFrame([r for r in trainer.state.log_history if "reward" in r])
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(logs["step"], logs["reward"], color="#2e548a", label="mean reward")
if "reward_std" in logs:
    ax.fill_between(logs["step"], logs["reward"] - logs["reward_std"], logs["reward"] + logs["reward_std"], alpha=0.2)
ax.set_xlabel("step")
ax.set_ylabel("reward (max 2.5)")
ax.set_title(f"GRPO · GSM8K · G={G}")
ax.grid(True, alpha=0.3)
fig.savefig(C.SCREENSHOTS / "08-grpo-reward.png", dpi=120, bbox_inches="tight")
plt.show()

acc_after = accuracy(trainer.model, tokenizer)
trainer.model.save_pretrained(str(C.GRPO_ADAPTER))
result = {"n_test": N_TEST, "steps": MAX_STEPS, "num_generations": G, "acc_before": acc_before, "acc_after": acc_after}
(C.GRPO_ADAPTER / "grpo_metrics.json").write_text(json.dumps(result, indent=2))
print(result)

# %% [markdown]
# ## 5. Câu hỏi
#
# 1. Reward tăng nhanh nhất ở thành phần nào: format hay correctness? Đó có phải "reward hacking" không?
# 2. Với N_TEST=100, chênh lệch accuracy bao nhiêu mới vượt nhiễu? (sai số chuẩn ≈ √(p(1−p)/n)).
# 3. So `acc_before` với GSM8K của SFT ở NB6: prompt và cách chấm khác nhau, nên con số nào
#    đáng tin hơn cho câu hỏi "GRPO có giúp toán không"?
