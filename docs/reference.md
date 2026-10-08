# Tài liệu tham khảo — Lab 22

Các chi tiết kỹ thuật được tách khỏi README. Học viên chỉ cần mở khi gặp lỗi hoặc muốn tuỳ biến.

## Hai tier (T4 / BigGPU)

| Tier | Tài nguyên tính toán | Mô hình gốc | SFT | Dữ liệu sở thích (huấn luyện / held-out) | Khi nào dùng |
|---|---|---|---|---|---|
| **T4 (mặc định)** | Colab T4 16 GB / GPU ≥ 12 GB | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` | 1.000 mẫu Alpaca tiếng Việt | 800 / 100 cặp tiếng Việt | Hầu hết học viên |
| **BigGPU** | A100 / L4 / H100 | `unsloth/Qwen3-8B-unsloth-bnb-4bit` (tắt chế độ suy nghĩ) | 2.000 mẫu Alpaca tiếng Việt | 3.500 / 200 | Có GPU lớn, muốn số liệu ổn định hơn |

Đổi tier bằng `COMPUTE_TIER` trong `.env`. Mọi tham số nằm ở [`lab22/config.py`](../lab22/config.py) và
đều ghi đè được bằng biến môi trường.

**Mô hình thay thế:** Gemma 4 E4B (đã tinh chỉnh theo chỉ dẫn, ~4B tham số hiệu dụng) cũng vừa T4. Đặt `BASE_MODEL` tới bản
Unsloth 4-bit của nó (kiểm tra đúng id trên Hugging Face trước). Gemma dùng mẫu hội thoại (chat template) khác ChatML, nên
`train_on_responses_only` ở NB1 cần đổi chuỗi đánh dấu; Qwen3 là lựa chọn mặc định đã được kiểm tra mã nguồn.

**VRAM:** với LoRA, TRL không nạp bản thứ hai của mô hình tham chiếu. Ở lab này log-xác suất của mô hình tham
chiếu được **tính trước** (`precompute_ref_log_probs=True`) bằng chính mô hình SFT, nên lúc huấn luyện GPU chỉ
giữ mô hình đang học cùng các câu `chosen`/`rejected` trong batch. DPO tốn hơn SFT chủ yếu vì mỗi bước xử lý hai
câu trả lời, không phải vì có hai bản trọng số.

Bảng VRAM, ổ đĩa và chọn tier: [`HARDWARE-GUIDE.md`](../HARDWARE-GUIDE.md).

## Chạy trên laptop / máy chủ (GPU ≥ 12 GB)

Yêu cầu: Python 3.10–3.13, NVIDIA GPU, driver CUDA 12.x.

```bash
bash setup-laptop.sh    # tạo venv, cài thư viện, kiểm tra CUDA
make smoke              # kiểm tra import, GPU, nguồn dữ liệu
make pipeline           # NB0 → NB4 (phần bắt buộc)
make verify             # tự kiểm tra trước khi nộp
```

Tất cả lệnh:

```
make nb0 / sft / data / dpo / eval     bắt buộc: NB0, NB1, NB2, NB3, NB4
make variants / deploy / bench / grpo  bonus: NB3b, NB5, NB6, NB7
make pipeline | pipeline-full          bắt buộc | bắt buộc + bonus
make beta-sweep                        β ∈ {0.05, 0.1, 0.5} + biểu đồ held-out
make colab                             sinh lại colab/*.ipynb
make test | verify | clean
```

## Công cụ sử dụng (kiểm tra 07/10/2026)

| Tầng | Công cụ | Phiên bản | Ghi chú |
|---|---|---|---|
| Huấn luyện | Unsloth | ≥ 2026.10.1 | chặn trên TRL ≤ 1.13.0, transformers ≤ 5.17, datasets < 5 |
| Thuật toán huấn luyện | TRL | 1.13.x | `loss_type` dạng list (`sigmoid`, `sft`, `sigmoid_norm`…), `ld_alpha`, ORPO ở `trl.experimental.orpo`, GRPO |
| Mô hình | transformers | 5.2–5.17 | v5 bỏ `warmup_ratio` → dùng `warmup_steps` dạng float |
| Adapters | PEFT | ≥ 0.18 | LoRA r=16, α=32 |
| Dữ liệu | datasets | 4.7–4.x | |
| Đánh giá | lm-eval | ≥ 0.4.13 | `enable_thinking`, `peft=`, chat template |
| Phục vụ mô hình | llama-cpp-python / vLLM | ≥ 0.3.16 / ≥ 0.10 | vLLM phục vụ LoRA không cần gộp |

## Lỗi thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| Hết bộ nhớ GPU (OOM) khi nạp mô hình | Sai tier. T4 dùng Qwen3-4B; vẫn OOM thì giảm `MAX_LEN` (768 → 512) |
| `rewards/chosen` âm, margin vẫn tăng | Dịch chuyển xác suất (likelihood displacement). Ghi vào REFLECTION §3, so với RPO ở NB3b |
| Margin ≈ 0 sau cả epoch | Tốc độ học (lr) quá thấp hoặc sai mô hình tham chiếu. Kiểm tra `adapter_config.json` trỏ tới `models/sft-merged` |
| `TypeError: ... warmup_ratio` / `max_prompt_length` | Code cũ viết cho transformers 4 / TRL 0.x (kể cả snippet `DPOConfig` trong slide: `ref_model` riêng, lr 5e-7, `max_prompt_length`). Dùng `lab22.modeling.dpo_config` |
| Câu trả lời có `<think>` | Qwen3 là mô hình lai có chế độ suy nghĩ: đã tắt bằng `enable_thinking=False`; kiểm tra `C.CHAT_TEMPLATE_KWARGS` |
| GGUF trả lời giống hệt SFT | Adapter DPO không được nạp. NB5 assert có tensor `lora_`; đừng export từ `adapters/sft-mini` |
| llama-cpp-python không cài được | `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python` (CUDA) hoặc `-DGGML_METAL=on` (Mac) |
| NB6 quá lâu trên T4 | Giảm giới hạn số câu trong `BENCHMARKS`; nhớ giới hạn của Global-MMLU tính **theo môn** |

## Dữ liệu và giấy phép

- Mã nguồn: MIT ([`LICENSE`](../LICENSE)).
- `sailor2/sea-ultrafeedback-onpolicy`: thẻ mô tả bộ dữ liệu không ghi giấy phép, nhưng bài báo Sailor2
  ([arXiv 2502.12982](https://arxiv.org/abs/2502.12982), Bảng 1) công bố mô hình, dữ liệu và mã nguồn theo
  **Apache-2.0**. Câu hỏi gốc lấy từ UltraFeedback (MIT); nhãn `chosen`/`rejected` do reward model của Skywork gán.
- `vuongtsc/vi-gsm8k-agentic` (NB7): MIT, tác giả Trần Đình Minh Vương (CAIR, VinUniversity). Lời giải do các
  LLM sinh rồi lọc; xem thẻ mô tả bộ dữ liệu.
- Giám khảo NB4: `Skywork/Skywork-Reward-V2-Qwen3-4B` (Apache-2.0) và `Skywork/Skywork-Reward-V2-Llama-3.2-3B`
  (Llama 3.2 Community License); xem thẻ mô tả mô hình.
- `saillab/alpaca-vietnamese-cleaned` (SFT): Alpaca-52K + Dolly-15K dịch sang tiếng Việt bằng Google Translate
  (dự án TaCo, UNH SAIL Lab), **CC BY-NC**, chỉ dùng cho học tập và nghiên cứu. Bản cũ
  `5CD-AI/Vietnamese-alpaca-cleaned` đã bị gỡ khỏi Hub (10/2026).
- GSM8K (MIT), IFEval (Apache-2.0), Global-MMLU (Apache-2.0).

## Lời cảm ơn

Unsloth, TRL, PEFT, lm-evaluation-harness, llama.cpp; Sailor2 (SEA UltraFeedback); SAIL Lab UNH (alpaca-vietnamese-cleaned); Trần Đình Minh Vương, CAIR (vi-gsm8k-agentic).
