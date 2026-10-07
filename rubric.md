# Lab Ngày 22 — Thang điểm (100 điểm bắt buộc + tối đa 20 điểm bonus)

Lab chiếm 30% điểm Daily Lab của Track 3. Cả hai tier (T4, BigGPU) tạo ra cùng loại kết quả; bạn được chấm
theo **bằng chứng và cách bạn giải thích**, không theo điểm số tuyệt đối của mô hình.

| # | Notebook | Tiêu chí | Điểm |
|---|---|---|---:|
| 0 | `00_dpo_loss_from_scratch` | `my_dpo_loss` qua các `assert` (loss = log 2 lúc khởi đầu; khớp công thức đóng) | 6 |
| 0 | `00_dpo_loss_from_scratch` | Trả lời câu hỏi về dịch chuyển xác suất: vì sao margin tăng được trong khi log-xác suất của câu `chosen` giảm? | 4 |
| 1 | `01_sft_mini` | Loss SFT giảm; lưu được `models/sft-merged/` (đây là mô hình tham chiếu của DPO) | 8 |
| 2 | `02_preference_data` | Chia tập huấn luyện / held-out theo câu hỏi (không trùng, `assert` qua); đã xem 3 cặp mẫu | 8 |
| 2 | `02_preference_data` | Đo thiên vị độ dài (`02b-pref-length.png`, báo tỉ lệ cặp có `chosen` dài hơn) | 4 |
| 3 | `03_dpo_train` | Adapter DPO huấn luyện trên `models/sft-merged` (thể hiện trong config của adapter) | 6 |
| 3 | `03_dpo_train` | Vẽ đường reward cho cả tập huấn luyện **và** held-out, tách riêng `chosen` và `rejected` | 10 |
| 3 | `03_dpo_train` | Giải thích chẩn đoán (INTENDED / LIKELIHOOD DISPLACEMENT / FAILURE / AMBIGUOUS) trong REFLECTION §3 | 8 |
| 4 | `04_compare_and_eval` | 8 câu hỏi cố định so sánh song song + sinh câu trả lời cho ≥ 50 câu held-out | 6 |
| 4 | `04_compare_and_eval` | Chấm tự động (mặc định là hội đồng hai reward model chạy local): báo win rate kèm khoảng tin cậy 95%, sanity accuracy, tỉ lệ câu dài thắng và win rate trên các cặp dài gần bằng nhau | 10 |
| — | Phản tư | §3, §4 và §6 trả lời bằng số liệu của chính bạn (§3 + §6 tổng cộng ≥ 150 từ) | 20 |
| — | Tái lập | `make pipeline` (hoặc Colab "Chạy tất cả") chạy được từ môi trường sạch | 5 |
| — | Kiểm tra | `make verify` kết thúc với mã 0 | 5 |
| | | **Tổng phần bắt buộc** | **100** |

### Cách đọc đường reward (§3)

Reward ngầm bắt đầu từ 0 vì lúc đầu mô hình đang học trùng với mô hình tham chiếu SFT.

- **Đúng kỳ vọng (Intended):** `chosen` ↑, `rejected` ↓, margin ↑.
- **Dịch chuyển xác suất (Likelihood displacement):** margin ↑ nhưng `chosen` ↓ (`rejected` giảm nhanh hơn).
  Hay gặp với DPO; không tự động là thất bại, nhưng phải giải thích được. RPO (NB3b) là một cách khắc phục.
- **Thất bại (Failure):** margin ≤ 0 trên dữ liệu held-out.

Chỉ thấy margin tăng thì **chưa** được 10 điểm biểu đồ: bằng chứng là đường held-out và việc tách riêng
`chosen`/`rejected`.

### Cách đọc kết quả chấm (§4)

- Khoảng tin cậy chứa 0.5 nghĩa là "không phát hiện khác biệt", không phải "DPO thắng".
- Giám khảo reward model: sanity accuracy dưới 80% trên các cặp tiếng Việt nghĩa là kết luận của nó không đáng tin.
  Cả hai giám khảo trong hội đồng đều từ cùng nhóm phát triển (Skywork) với reward model đã gán nhãn dữ liệu
  huấn luyện, và giám khảo Qwen3 cùng họ với mô hình sinh dữ liệu; bài làm tốt cần nêu điều này và so sánh `per_judge`.
- Giám khảo API: position consistency (độ nhất quán khi đổi chỗ A/B) thấp nghĩa là giám khảo không đáng tin cho cặp mô hình này.
- Nếu câu dài hơn gần như luôn thắng và câu trả lời của DPO dài hơn, hãy bàn về hiện tượng "hack độ dài".

## Phần bonus (tối đa +20)

| Phần | Điểm | Yêu cầu |
|---|---:|---|
| NB3b — biến thể | +8 | Bảng DPO / RPO / DPO-norm / LD-DPO / ORPO + biến thể nào thay đổi độ dài câu trả lời nhiều nhất, và vì sao |
| NB5 — GGUF | +4 | Xuất SFT+DPO sang Q4_K_M; so câu trả lời HF và GGUF trong `deploy_meta.json` |
| NB6 — benchmark | +6 | IFEval / GSM8K / Global-MMLU-vi có dùng chat template; đọc chênh lệch so với sai số chuẩn (stderr) trong REFLECTION §7 |
| NB7 — GRPO | +8 | Đường reward + độ chính xác trước/sau, kèm ước lượng nhiễu |
| β-sweep | +6 | `make beta-sweep`; margin và độ chính xác held-out theo β, giải thích ≥ 100 từ |
| Chấm chéo | +4 | Chạy NB4 bằng reward model và một giám khảo API khác họ; báo `cross_judge.agreement` |
| Đẩy lên HF Hub | +3 | Adapter + thẻ mô tả mô hình (mô hình gốc, dữ liệu, siêu tham số, kết quả đánh giá) |

## Nộp bài

Nộp đường dẫn GitHub public vào LMS (không tạo PR). Bao gồm NB0–NB4 đã chạy (giữ output) hoặc file Colab đã
chạy, `submission/screenshots/` và `submission/REFLECTION.md`. Giữ repo public đến khi có điểm.

Nộp muộn: hạn 23:59 ngày hôm sau; trừ 10% mỗi ngày; quá 3 ngày được 0 điểm. Phúc khảo trong vòng 1 tuần.
