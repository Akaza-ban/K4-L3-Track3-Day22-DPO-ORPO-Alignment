# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** _<Họ Tên>_
**Cohort:** _<A20-K4 / ...>_
**Tier đã chạy:** _<T4 | BIGGPU | both>_
**Date:** _<YYYY-MM-DD>_

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Setup

| Item | Value |
|---|---|
| GPU / VRAM | _<e.g., Colab T4 16 GB>_ |
| Base model | _<e.g., unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit>_ |
| SFT data | _<bkai-foundation-models/vi-alpaca · N mẫu · epochs>_ |
| Preference data | _<sailor2/sea-ultrafeedback-onpolicy (vi) · N train / N held-out>_ |
| Chosen dài hơn rejected (NB2) | _<e.g., 65%>_ |
| DPO: β / lr / epochs | _<0.1 / 5e-6 / 1>_ |
| Judge | _<rm:model hoặc provider:model; sanity accuracy>_ |
| Chi phí | _<$0 Colab free / ...>_ |

---

## 2. Kết quả DPO

| Metric | Giá trị |
|---|---:|
| Thời gian train NB3 | _<...>_ |
| VRAM peak | _<...>_ |
| Train reward gap cuối (chosen − rejected) | _<...>_ |
| Held-out reward accuracy | _<...>_ |
| Held-out margin | _<...>_ |
| Chẩn đoán tự động (`diagnosis`) | _<INTENDED / LIKELIHOOD DISPLACEMENT / FAILURE / AMBIGUOUS>_ |
| Độ dài TB câu trả lời SFT → DPO (NB4) | _<... → ... ký tự>_ |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

_Mô tả riêng `rewards/chosen` và `rewards/rejected` trên **train và held-out**. Chosen tăng hay giảm?
Margin tăng vì chosen tăng hay vì rejected giảm nhanh hơn (likelihood displacement)? Held-out có đi
cùng hướng với train không, hay chỉ train tăng (overfit)? Chẩn đoán tự động có khớp với điều bạn
thấy không?_

_Trả lời ở đây._

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate cặp dài gần bằng | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | | | | | | | |
| helpfulness (4) | | | | | | | |
| safety (4) | | | | | | | |

Judge: ______ · sanity accuracy: ______ · `score_length_spearman` (RM) hoặc position consistency (API): ______

_CI có chứa 0.5 không? Judge có đáng tin trên tiếng Việt không (sanity set)? DPO thắng vì câu trả lời tốt
hơn hay vì dài hơn? Hai RM trong hội đồng (`per_judge`) có cho win rate gần nhau không? Nếu judge Qwen3 cho DPO thắng
cao hơn hẳn judge Llama, điều đó nói gì về preference leakage?
Chọn 2 ví dụ cụ thể (1 helpfulness, 1 safety) và giải thích._

_Trả lời ở đây._

---

## 5. β trade-off (bonus `make beta-sweep`)

| β | Held-out margin | Held-out accuracy | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

_Nếu không chạy: viết giả thuyết 3 câu về điều bạn dự đoán sẽ thấy._

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, lr, lát dữ liệu, judge, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

_Trả lời ở đây._

---

## 7. Benchmark (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Benchmark | Limit / subtask | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có alignment tax trên GSM8K không? Benchmark có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Held-out accuracy | Held-out margin | Độ dài TB | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Accuracy trước / sau (n test) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (format hay correctness)? Chênh lệch có vượt nhiễu không?_

---

## Bonus checklist

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Cross-judge, hai họ model (+4)
- [ ] HF Hub push + model card (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_
