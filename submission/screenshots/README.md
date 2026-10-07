# Screenshots

Các notebook **tự lưu** ảnh vào thư mục này. `make verify` kiểm tra 4 ảnh core.

## Core (bắt buộc)

| File | Notebook | Nội dung |
|---|---|---|
| `02-sft-loss.png` | NB1 | Loss SFT giảm dần |
| `02b-pref-length.png` | NB2 | Phân bố độ dài chosen vs rejected (thiên vị độ dài của dữ liệu) |
| `03-dpo-reward-curves.png` | NB3 | `rewards/chosen` và `rewards/rejected` **riêng biệt**, train + held-out, cùng margin |
| `04-side-by-side-table.png` | NB4 | 8 prompt cố định, SFT vs SFT+DPO |

Chỉ có "margin tăng" thì chưa đủ: rubric yêu cầu thấy chosen và rejected riêng.

## Bonus

| File | Notebook |
|---|---|
| `03b-variants.png` | NB3b — DPO / RPO / DPO-norm / LD-DPO / ORPO |
| `06-gguf-smoke.png` | NB5 — **chụp tay** cell llama-cpp (tên file `Q4_K_M` + câu trả lời) |
| `07-benchmark-comparison.png` | NB6 — IFEval / GSM8K / Global-MMLU-vi có error bar |
| `08-grpo-reward.png` | NB7 — reward GRPO theo step |
| `bonus-beta-sweep.png` | `make beta-sweep` |

## Lưu ý

- Không để lộ API key trong ảnh. Key chỉ nằm trong `.env` hoặc Colab Secrets.
- Ảnh chụp tay thì crop sát nội dung.
