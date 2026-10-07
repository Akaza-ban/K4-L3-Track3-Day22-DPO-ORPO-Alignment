# Day 22 — DPO/ORPO Alignment Lab (Track 3)

Lab cho **AICB-P2T3 · Ngày 22 · DPO/ORPO Alignment — From SFT to Preference Learning**.
Viết loss DPO từ đầu → build SFT-mini → train DPO trên dữ liệu preference **tiếng Việt** →
so sánh SFT vs SFT+DPO trên prompt **held-out** → (bonus) biến thể DPO/ORPO, GGUF, lm-eval, GRPO.

> Bản này là K4 đã sửa lỗi và cập nhật stack (10/2026). Danh sách thay đổi: [`CHANGELOG.md`](CHANGELOG.md).
> **Chưa chạy end-to-end trên GPU**: thời gian và số liệu trong README là ước tính.

---

## Hai tier

| Tier | Compute | Base model | SFT | Preference (train / held-out) | Khi nào dùng |
|---|---|---|---|---|---|
| **T4 (default)** | Colab T4 16 GB / GPU ≥ 12 GB | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` | 1k VN Alpaca | 800 / 100 cặp tiếng Việt | Hầu hết học viên |
| **BigGPU** | A100 / L4 / H100 | `unsloth/Qwen3-8B-unsloth-bnb-4bit` (thinking tắt) | 2k VN Alpaca | 3500 / 200 | Có GPU lớn, muốn số liệu ổn định hơn |

Đổi tier bằng `COMPUTE_TIER` trong `.env`. Mọi tham số nằm ở [`lab22/config.py`](lab22/config.py) và
đều override được bằng biến môi trường.

**Model thay thế:** Gemma 4 E4B (instruction-tuned, ~4B hiệu dụng) cũng vừa T4. Đặt `BASE_MODEL` tới bản
Unsloth 4-bit của nó (kiểm tra đúng id trên Hugging Face trước). Gemma dùng chat template khác ChatML, nên
`train_on_responses_only` ở NB1 cần đổi marker; Qwen3 là đường mặc định đã được kiểm tra mã nguồn.

**VRAM:** với LoRA, TRL không nạp model reference thứ hai. Ở lab này reference log-prob được **tính trước**
(`precompute_ref_log_probs=True`) bằng chính model SFT, nên lúc train chỉ có policy + chosen/rejected trong
batch. DPO tốn hơn SFT chủ yếu vì mỗi bước xử lý hai câu trả lời, không phải vì hai bản trọng số.

---

## Quick start

**Colab (không cài gì):** mở `colab/Lab22_DPO_T4.ipynb` (hoặc `Lab22_DPO_BigGPU.ipynb`) →
Runtime → Change runtime type → T4 GPU → Run all. Notebook Colab được sinh tự động từ
`notebooks/` + `lab22/` bằng `make colab`, nên luôn khớp với mã nguồn.

**Laptop / server (GPU ≥ 12 GB):**

```bash
bash setup-laptop.sh    # venv + deps + CUDA probe
make smoke              # import + GPU + sources
make pipeline           # NB0 → NB4 (core)
make verify             # gatekeeper trước khi nộp
```

Yêu cầu: Python 3.10–3.13, NVIDIA GPU, driver CUDA 12.x. `make test` chạy test CPU (không cần GPU).

### Lệnh `make`

```
make nb0 / sft / data / dpo / eval     core: NB0, NB1, NB2, NB3, NB4
make variants / deploy / bench / grpo  bonus: NB3b, NB5, NB6, NB7
make pipeline | pipeline-full          core | core + bonus
make beta-sweep                        β ∈ {0.05, 0.1, 0.5} + biểu đồ held-out
make colab                             sinh lại colab/*.ipynb
make test | verify | clean
```

---

## Notebooks

| Notebook | Nội dung | Pass khi |
|---|---|---|
| `00_dpo_loss_from_scratch` (CPU) | Tự viết `sequence_logps` và DPO loss, kiểm tra loss = log 2 lúc khởi tạo, thấy likelihood displacement trên ví dụ đồ chơi, so IPO/RPO/SimPO/ORPO | các `assert` qua; trả lời câu hỏi cuối notebook |
| `01_sft_mini` | Unsloth + LoRA, loss chỉ trên câu trả lời; lưu adapter **và** model merged 16-bit `models/sft-merged` (= reference của DPO) | loss giảm; `models/sft-merged/` tồn tại |
| `02_preference_data` | `sailor2/sea-ultrafeedback-onpolicy` lọc tiếng Việt → format hội thoại của TRL; tách train/held-out **theo prompt**; đo thiên vị độ dài | không prompt nào nằm ở cả hai phía; `02b-pref-length.png` |
| `03_dpo_train` | `DPOTrainer` trên `models/sft-merged` + LoRA mới, lr 5e-6, β 0.1, đánh giá trên held-out; chẩn đoán đường reward | `03-dpo-reward-curves.png`; `dpo_metrics.json` có chẩn đoán |
| `03b_dpo_variants` (bonus) | Cùng dữ liệu, chỉ đổi loss: DPO, RPO, DPO chuẩn hoá độ dài, LD-DPO, ORPO | `03b-variants.png` |
| `04_compare_and_eval` | 8 prompt cố định + ≥ 50 prompt held-out; **chấm tự động** bằng hội đồng 2 reward model local khác họ (không cần API key, có bộ sanity tiếng Việt, giảm preference leakage) hoặc judge API hai chiều; CI 95%, "câu dài thắng", win rate cặp dài gần bằng | `judge_summary.json` |
| `05_merge_deploy_gguf` (bonus) | Load SFT+**DPO** ở 16-bit → GGUF Q4_K_M → so câu trả lời HF vs GGUF | `deploy_meta.json` |
| `06_benchmark` (bonus) | lm-eval có chat template: IFEval, GSM8K, Global-MMLU-vi; limit tính theo subtask; có stderr | `07-benchmark-comparison.png` |
| `07_grpo_bonus` (bonus) | GRPO với reward kiểm chứng được: bài toán tiếng Việt `vuongtsc/vi-gsm8k-agentic`, chấm đáp số (đọc được `1.440`, `2,5`) | `08-grpo-reward.png` |

Notebook là file Jupytext `.py` (dễ review). Logic dùng chung nằm trong package [`lab22/`](lab22/)
(`config`, `data`, `judge`, `dpo_math`, `math_reward`, `modeling`) và có test CPU trong `scripts/test_lab22.py`.

---

## Deliverables

1. Notebook đã chạy (giữ output) cho NB0–NB4.
2. Ảnh trong `submission/screenshots/` (notebook tự lưu): `02-sft-loss`, `02b-pref-length`,
   `03-dpo-reward-curves`, `04-side-by-side-table`. Bonus: `03b-variants`, `06-gguf-smoke`,
   `07-benchmark-comparison`, `08-grpo-reward`, `bonus-beta-sweep` (`06-gguf-smoke` chụp tay).
3. `submission/REFLECTION.md` điền đủ, có số liệu thật của bạn.
4. `make verify` qua.

Chấm điểm: [`rubric.md`](rubric.md).

---

## Tech stack (kiểm tra 07/10/2026)

| Layer | Tool | Version | Ghi chú |
|---|---|---|---|
| Training | Unsloth | ≥ 2026.10.1 | chặn trên TRL ≤ 1.13.0, transformers ≤ 5.17, datasets < 5 |
| Trainers | TRL | 1.13.x | `loss_type` dạng list (`sigmoid`, `sft`, `sigmoid_norm`…), `ld_alpha`, ORPO ở `trl.experimental.orpo`, GRPO |
| Model | transformers | 5.2–5.17 | v5 bỏ `warmup_ratio` → dùng `warmup_steps` dạng float |
| Adapters | PEFT | ≥ 0.18 | LoRA r=16, α=32 |
| Data | datasets | 4.7–4.x | |
| Eval | lm-eval | ≥ 0.4.13 | `enable_thinking`, `peft=`, chat template |
| Serving | llama-cpp-python / vLLM | ≥ 0.3.16 / ≥ 0.10 | vLLM phục vụ LoRA không cần merge |

---

## Common gotchas

| Triệu chứng | Cách xử lý |
|---|---|
| OOM khi load | Sai tier. T4 dùng Qwen3-4B; vẫn OOM thì giảm `MAX_LEN` (768 → 512) |
| `rewards/chosen` âm, margin vẫn tăng | Likelihood displacement. Ghi vào REFLECTION §3, so với RPO ở NB3b |
| Margin ≈ 0 sau cả epoch | lr quá thấp hoặc reference sai. Kiểm tra `adapter_config.json` trỏ tới `models/sft-merged` |
| `TypeError: ... warmup_ratio` / `max_prompt_length` | Code cũ viết cho transformers 4 / TRL 0.x (kể cả snippet `DPOConfig` trong slide: `ref_model` riêng, lr 5e-7, `max_prompt_length`). Dùng `lab22.modeling.dpo_config` |
| Câu trả lời có `<think>` | Model Qwen3 hybrid: đã tắt bằng `enable_thinking=False`; kiểm tra `C.CHAT_TEMPLATE_KWARGS` |
| GGUF trả lời giống hệt SFT | Adapter DPO không được load. NB5 assert có tensor `lora_`; đừng export từ `adapters/sft-mini` |
| llama-cpp-python không cài được | `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python` (CUDA) hoặc `-DGGML_METAL=on` (Mac) |
| NB6 quá lâu trên T4 | Giảm limit trong `BENCHMARKS`; nhớ limit của Global-MMLU tính **theo môn** |

---

## Submission

Nộp URL GitHub công khai vào LMS (không cần PR):

1. Copy repo lên tài khoản của bạn, đặt **public**.
2. Chạy NB0–NB4 (giữ output), điền REFLECTION, `make verify`.
3. `git add -A && git commit -m "Lab 22 submission" && git push`.

Bài được chấm tự động từ repo, nên chỉ những gì đã commit mới được tính. `.gitignore` đã giữ lại các file
bằng chứng nhỏ (`adapters/*/adapter_config.json`, `dpo_metrics.json`, `split.json`, `data/pref/*.parquet`,
`data/eval/*.json[l]`) và chặn trọng số (`models/`, `*.safetensors`, GGUF). Đừng commit `.env`.

Option B (+5): đẩy adapter lên Hugging Face Hub. Option C: chỉ code + report.

---

## Bonus Challenge (không chấm điểm)

Xem [`BONUS-CHALLENGE.md`](BONUS-CHALLENGE.md) · [`BONUS-CHALLENGE-EN.md`](BONUS-CHALLENGE-EN.md) và
[`VIBE-CODING.md`](VIBE-CODING.md).

---

## Dữ liệu và giấy phép

- Code: MIT ([`LICENSE`](LICENSE)).
- `sailor2/sea-ultrafeedback-onpolicy`: dataset card không ghi license, nhưng bài báo Sailor2
  ([arXiv 2502.12982](https://arxiv.org/abs/2502.12982), Bảng 1) công bố model, dữ liệu và code theo
  **Apache-2.0**. Prompt gốc từ UltraFeedback (MIT); nhãn chosen/rejected do reward model Skywork gán.
- `vuongtsc/vi-gsm8k-agentic` (NB7): MIT, tác giả Trần Đình Minh Vương (CAIR, VinUniversity). Lời giải do các
  LLM sinh rồi lọc; xem dataset card.
- Judge NB4: `Skywork/Skywork-Reward-V2-Qwen3-4B` (Apache-2.0) và `Skywork/Skywork-Reward-V2-Llama-3.2-3B`
  (Llama 3.2 Community License); xem model card.
- `5CD-AI/Vietnamese-alpaca-cleaned`: xem dataset card trước khi dùng ngoài lớp học.
- GSM8K (MIT), IFEval (Apache-2.0), Global-MMLU (Apache-2.0).

## Acknowledgments

Unsloth, TRL, PEFT, lm-evaluation-harness, llama.cpp; Sailor2 (SEA UltraFeedback); 5CD-AI; Trần Đình Minh Vương, CAIR (vi-gsm8k-agentic).

© VinUniversity AICB program · Track 3 Day 22.
