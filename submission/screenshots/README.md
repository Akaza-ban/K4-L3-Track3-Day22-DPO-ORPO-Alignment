# Ảnh nộp bài

Các notebook **tự lưu** ảnh vào thư mục này. `make verify` kiểm tra 4 ảnh bắt buộc.

## Bắt buộc

| File | Notebook | Nội dung |
|---|---|---|
| `02-sft-loss.png` | NB1 | Loss SFT giảm dần |
| `02b-pref-length.png` | NB2 | Phân bố độ dài `chosen` so với `rejected` (thiên vị độ dài của dữ liệu) |
| `03-dpo-reward-curves.png` | NB3 | `rewards/chosen` và `rewards/rejected` **riêng biệt**, trên tập huấn luyện và held-out, kèm margin |
| `04-side-by-side-table.png` | NB4 | 8 câu hỏi cố định, SFT so với SFT+DPO |

Chỉ có "margin tăng" thì chưa đủ: thang điểm yêu cầu thấy riêng `chosen` và `rejected`.

## Bonus

| File | Notebook |
|---|---|
| `03b-variants.png` | NB3b — DPO / RPO / DPO-norm / LD-DPO / ORPO |
| `06-gguf-smoke.png` | NB5 — **chụp màn hình thủ công** cell llama-cpp (tên file `Q4_K_M` + câu trả lời) |
| `07-benchmark-comparison.png` | NB6 — IFEval / GSM8K / Global-MMLU-vi có thanh sai số |
| `08-grpo-reward.png` | NB7 — reward GRPO theo từng bước |
| `bonus-beta-sweep.png` | `make beta-sweep` |

## Lưu ý

- Không để lộ khoá API trong ảnh. Khoá chỉ nằm trong `.env` hoặc Colab Secrets.
- Ảnh chụp thủ công thì cắt sát nội dung.
