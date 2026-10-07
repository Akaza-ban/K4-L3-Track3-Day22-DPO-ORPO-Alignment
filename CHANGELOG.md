# Changelog

## 0.2.0 — 2026-10-07

Bản sửa và cập nhật của K4-Track3-Day22. **Chưa chạy end-to-end trên GPU**: mã nguồn đã qua test CPU,
lint và kiểm tra tĩnh; thời gian và VRAM trong tài liệu là ước tính.

### Sửa lỗi

- **GGUF thiếu DPO.** NB5 cũ merge adapter SFT nên file GGUF không chứa DPO. Giờ NB5 load `adapters/dpo`
  ở 16-bit, assert có tensor LoRA, rồi so câu trả lời HF vs GGUF (`deploy_meta.json`).
- **Reference sai.** DPO cũ dùng base model làm reference. NB1 giờ lưu `models/sft-merged` (16-bit); NB3
  gắn LoRA mới lên đó và tính trước reference log-prob, nên reference = SFT. `make verify` kiểm tra
  `adapter_config.json` trỏ tới `sft-merged`.
- **lr 5e-7 → 5e-6.** 5e-7 hầu như không làm LoRA dịch chuyển trong ~100 bước.
- **Eval rò dữ liệu train.** Bản cũ lấy 50 cặp cuối của tập train làm eval. Giờ tách train / held-out
  **theo prompt** (chuẩn hoá, có assert không trùng); NB3 đánh giá trên held-out, NB4 lấy prompt từ held-out.
- **lm-eval.** `--limit` tính theo subtask (MMLU 57 môn) nên limit đặt theo môn; thêm
  `--apply_chat_template`, `--fewshot_as_multiturn`, `enable_thinking=False`, báo stderr. Bỏ AlpacaEval-lite
  (dataset dạng script, không load được với `datasets` ≥ 4).
- **Judge.** Mỗi cặp chấm hai lần đổi chỗ A/B; lệch nhau tính hoà. Báo CI 95% (bootstrap), position
  consistency và tỉ lệ "câu dài hơn thắng". Model judge lấy từ `JUDGE_PROVIDER`/`JUDGE_MODEL`, không
  hard-code. ≥ 50 prompt held-out. Không có key → phiếu chấm tay ẩn danh, không tự ghi "hoà".

### Sửa sau review chéo

- Judge: câu trả lời sai định dạng được hỏi lại một lần, rồi đánh dấu `failed` và loại khỏi win rate
  (đếm ở `n_failed`), không còn tính là hoà. "Câu dài hơn thắng" so người thắng với người thua, bỏ cặp dài bằng nhau.
- Chấm tay: mỗi prompt hai dòng (hai thứ tự A/B), xáo trộn, gộp bằng cùng luật với judge API; position
  consistency là số thật. Có key provider nhưng thiếu API key thì tự chuyển sang chấm tay.
- `judge_summary.json` lưu hash của `side_by_side.jsonl`; sinh lại output thì xoá kết quả cũ; `make verify`
  báo STALE nếu lệch và yêu cầu ≥ 50 prompt held-out khác nhau.
- `.env` được đọc bởi `lab22/config.py` (không cần python-dotenv). Template kwargs (`enable_thinking=False`)
  đi vào dữ liệu train qua cột `chat_template_kwargs`. Lọc độ dài theo token cả hai phía chosen/rejected.
- IPO trong NB0 chuẩn hoá theo số token như TRL. Colab giải phóng GPU giữa các notebook.
- Adapter DPO lưu dấu vân tay (`split.json`) của train/held-out; NB4 và `make verify` từ chối nếu
  split đã bị sinh lại sau khi train (held-out có thể đã lọt vào train).
- Chấm tay luôn đọc lại phiếu; `make verify` báo WRONG REF là lỗi (không còn là cảnh báo) và kiểm tra
  từng mục cốt lõi của REFLECTION còn placeholder hay không.
- NB5 so HF vs GGUF trên cùng prompt đã áp chat template (thinking tắt), cùng số token.

### Cập nhật

- Stack: Unsloth ≥ 2026.10.1, TRL 1.13, transformers 5.2–5.17, datasets 4.x, PEFT ≥ 0.18, lm-eval ≥ 0.4.13.
  API mới: `loss_type` dạng list, `warmup_steps` float, ORPO ở `trl.experimental.orpo`.
- Base model: Qwen3-4B-Instruct-2507 (T4), Qwen3-8B thinking tắt (BigGPU). Gemma 4 E4B ghi là phương án thay thế.
- Dữ liệu preference tiếng Việt: `sailor2/sea-ultrafeedback-onpolicy` lọc `vi`.
- Notebook mới: NB0 (DPO loss từ đầu, CPU, từ K3), NB3b (DPO / RPO / DPO-norm / LD-DPO / ORPO), NB7 (GRPO, RLVR trên GSM8K).
- Package `lab22/` dùng chung (config, data, judge, dpo_math, modeling) + test CPU `scripts/test_lab22.py`.
- Notebook Colab sinh tự động bằng `make colab` (`scripts/build_colab.py`), test kiểm tra không lệch nguồn.
- `make verify` viết lại: kiểm tra reference, held-out, chẩn đoán reward, judge summary.
- Tài liệu: README, rubric, HARDWARE-GUIDE (sửa nhận định "2× VRAM"), REFLECTION, screenshots.
