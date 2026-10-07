# Tài liệu tham khảo — Lab 22

Chi tiết kỹ thuật tách khỏi README. Học viên chỉ cần mở khi gặp lỗi hoặc muốn tuỳ biến.

## Hai tier (T4 / BigGPU)

| Tier | Compute | Base model | SFT | Preference (train / held-out) | Khi nào dùng |
|---|---|---|---|---|---|
| **T4 (default)** | Colab T4 16 GB / GPU ≥ 12 GB | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` | 1k VN Alpaca | 800 / 100 cặp tiếng Việt | Hầu hết học viên |
| **BigGPU** | A100 / L4 / H100 | `unsloth/Qwen3-8B-unsloth-bnb-4bit` (thinking tắt) | 2k VN Alpaca | 3500 / 200 | Có GPU lớn, muốn số liệu ổn định hơn |

Đổi tier bằng `COMPUTE_TIER` trong `.env`. Mọi tham số nằm ở [`lab22/config.py`](../lab22/config.py) và
đều override được bằng biến môi trường.

**Model thay thế:** Gemma 4 E4B (instruction-tuned, ~4B hiệu dụng) cũng vừa T4. Đặt `BASE_MODEL` tới bản
Unsloth 4-bit của nó (kiểm tra đúng id trên Hugging Face trước). Gemma dùng chat template khác ChatML, nên
`train_on_responses_only` ở NB1 cần đổi marker; Qwen3 là đường mặc định đã được kiểm tra mã nguồn.

**VRAM:** với LoRA, TRL không nạp model reference thứ hai. Ở lab này reference log-prob được **tính trước**
(`precompute_ref_log_probs=True`) bằng chính model SFT, nên lúc train chỉ có policy + chosen/rejected trong
batch. DPO tốn hơn SFT chủ yếu vì mỗi bước xử lý hai câu trả lời, không phải vì hai bản trọng số.

Bảng VRAM, ổ đĩa và chọn tier: [`HARDWARE-GUIDE.md`](../HARDWARE-GUIDE.md).

## Chạy trên laptop / server (GPU ≥ 12 GB)

Yêu cầu: Python 3.10–3.13, NVIDIA GPU, driver CUDA 12.x.

```bash
bash setup-laptop.sh    # venv + deps + CUDA probe
make smoke              # import + GPU + sources
make pipeline           # NB0 → NB4 (core)
make verify             # gatekeeper trước khi nộp
```

Tất cả lệnh:

```
make nb0 / sft / data / dpo / eval     core: NB0, NB1, NB2, NB3, NB4
make variants / deploy / bench / grpo  bonus: NB3b, NB5, NB6, NB7
make pipeline | pipeline-full          core | core + bonus
make beta-sweep                        β ∈ {0.05, 0.1, 0.5} + biểu đồ held-out
make colab                             sinh lại colab/*.ipynb
make test | verify | clean
```

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

## Lỗi thường gặp

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

## Dữ liệu và giấy phép

- Code: MIT ([`LICENSE`](LICENSE)).
- `sailor2/sea-ultrafeedback-onpolicy`: dataset card không ghi license, nhưng bài báo Sailor2
  ([arXiv 2502.12982](https://arxiv.org/abs/2502.12982), Bảng 1) công bố model, dữ liệu và code theo
  **Apache-2.0**. Prompt gốc từ UltraFeedback (MIT); nhãn chosen/rejected do reward model Skywork gán.
- `vuongtsc/vi-gsm8k-agentic` (NB7): MIT, tác giả Trần Đình Minh Vương (CAIR, VinUniversity). Lời giải do các
  LLM sinh rồi lọc; xem dataset card.
- Judge NB4: `Skywork/Skywork-Reward-V2-Qwen3-4B` (Apache-2.0) và `Skywork/Skywork-Reward-V2-Llama-3.2-3B`
  (Llama 3.2 Community License); xem model card.
- `saillab/alpaca-vietnamese-cleaned` (SFT): Alpaca-52K + Dolly-15K dịch sang tiếng Việt bằng Google Translate
  (dự án TaCo, UNH SAIL Lab), **CC BY-NC**, chỉ dùng cho học tập và nghiên cứu. Bản cũ
  `5CD-AI/Vietnamese-alpaca-cleaned` đã bị gỡ khỏi Hub (10/2026).
- GSM8K (MIT), IFEval (Apache-2.0), Global-MMLU (Apache-2.0).

## Acknowledgments

Unsloth, TRL, PEFT, lm-evaluation-harness, llama.cpp; Sailor2 (SEA UltraFeedback); SAIL Lab UNH (alpaca-vietnamese-cleaned); Trần Đình Minh Vương, CAIR (vi-gsm8k-agentic).
