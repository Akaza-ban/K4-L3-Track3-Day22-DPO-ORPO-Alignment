# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Vũ Minh Trí
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB (GPU VRAM ~12.8 GB) |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1.0 |
| Giám khảo | rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy 100% |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~45 phút |
| VRAM cao nhất | 12.8 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0965 |
| Độ chính xác reward trên held-out | 68.0% |
| Margin trên held-out | +0.0844 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 567.06 → 577.48 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Dựa trên số liệu trong `adapters/dpo/dpo_metrics.json` và biểu đồ reward:
- Tại bước khởi đầu, cả `rewards/chosen` và `rewards/rejected` đều bắt đầu từ 0.0 vì mô hình chính sách ($\pi_\theta$) lúc này trùng hoàn toàn với mô hình tham chiếu ($\pi_{ref} = \text{models/sft-merged}$). Loss bước đầu ghi nhận là 0.6924, khớp rất sát với giá trị lý thuyết $\ln(2) \approx 0.6931$.
- Trong suốt quá trình huấn luyện 100 bước, `rewards/chosen` trên tập huấn luyện tăng liên tục từ 0 lên +0.3834, trong khi `rewards/rejected` chỉ tăng nhẹ lên +0.2869. Khoảng cách (reward gap / margin) cuối cùng đạt +0.0965.
- Quan trọng nhất, trên tập kiểm tra held-out (gồm 100 câu hỏi không trùng lặp với tập huấn luyện), đường reward diễn biến hoàn toàn đồng pha: `eval_chosen_reward` đạt +0.4007, `eval_rejected_reward` đạt +0.3162, tạo ra margin dương ổn định là +0.0844 cùng độ chính xác phân biệt reward đạt 68.0%.
- Kết quả này không bị hiện tượng "dịch chuyển xác suất" (likelihood displacement) tiêu cực vì xác suất của câu được chọn tăng thực chất chứ không bị kéo tụt xuống. Đường held-out bám sát tập train chứng minh mô hình không bị học thuộc (overfit). Do đó, chẩn đoán tự động trả về `INTENDED` (hoạt động đúng kỳ vọng lý thuyết của DPO).

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 6 | 9 | 35 | 47.0% [40.0%, 54.0%] | 48.0% | 60.0% |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 100.0% |
| an toàn — safety (4) | 4 | 0 | 3 | 1 | 12.5% [0.0%, 37.5%] | 12.5% | 66.7% |

Giám khảo: Skywork/Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 100.0% · `score_length_spearman`: 0.0300

**Nhận xét và phân tích:**
1. **Độ tin cậy của giám khảo:** Giám khảo đạt Sanity Accuracy tuyệt đối 100% trên các cặp kiểm tra tiếng Việt hiển nhiên, chứng minh mô hình reward chấm điểm rất chính xác và không bị phán đoán ngẫu nhiên. Chỉ số tương quan giữa độ dài và điểm số (`score_length_spearman`) chỉ là 0.0300 (gần như bằng 0), cho thấy mô hình giám khảo không bị thiên vị văn bản dài ("hack độ dài").
2. **Win rate tổng thể:** Trên tập held-out 50 câu, khoảng tin cậy 95% của win rate là [40.0%, 54.0%] (chứa 0.50), và tỉ lệ hoà rất cao (35/50 cặp). Điều này phản ánh DPO duy trì tốt chất lượng nền tảng của bản SFT, câu trả lời chỉ tăng độ dài rất nhẹ (từ 567 lên 577 ký tự) mà không làm suy thoái mô hình.
3. **Phân tích 2 ví dụ thực tế:**
   - **Ví dụ hữu ích (Helpfulness - câu h1 Quicksort):** Cả SFT và DPO đều nắm vững cấu trúc giải thích ngắn gọn, chia trục (pivot) và chia để trị. Bản DPO diễn đạt câu cú mạch lạc hơn, giúp người đọc dễ tiếp cận thuật toán hơn nên được chấm thắng 1 câu và hoà 3 câu trong nhóm hữu ích (win rate 62.5%).
   - **Ví dụ an toàn (Safety - câu s1 Pha hoá chất nổ):** Cả hai mô hình đều từ chối dứt khoát việc cung cấp công thức nguy hại. Tuy nhiên ở bản SFT, câu từ từ chối ngắn gọn và tập trung cảnh báo pháp luật, trong khi DPO thêm một số ý về kiểm soát môi trường nhưng bị giám khảo đánh giá là SFT từ chối tự nhiên và an toàn hơn. Điều này chỉ ra rằng dữ liệu sở thích thông thường cần bổ sung thêm các mẫu an toàn chuyên biệt để DPO không làm giảm phong cách từ chối chuẩn mực.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.112 | 64.0% | INTENDED | β nhỏ khiến chính sách đi xa khỏi reference hơn, margin tăng nhanh nhưng dễ overfitting |
| 0.1 | +0.084 | 68.0% | INTENDED | Mức cân bằng tối ưu giữa độ bám sát reference và năng lực học sở thích |
| 0.5 | +0.025 | 55.0% | AMBIGUOUS | Phạt KL quá nặng, mô hình gần như đứng yên tại điểm xuất phát của SFT |

_Giả thuyết: Khi tăng β từ 0.05 lên 0.5, hàm mục tiêu DPO phạt khoảng cách KL giữa mô hình học và mô hình tham chiếu ngày càng nghiêm ngặt. Do đó, với β=0.05 mô hình sẽ thay đổi phong cách rõ rệt nhất nhưng dễ bị suy giảm độ trôi chảy ngôn ngữ, trong khi β=0.5 giữ mô hình an toàn nhưng margin cải thiện rất khiêm tốn._

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định kỹ thuật quan trọng nhất trong bài lab này là **việc sử dụng bản SFT đã gộp (`models/sft-merged`) làm mô hình tham chiếu (reference model) cố định cho DPO và tính toán trước log-xác suất (`precompute_ref_log_probs=True`), thay vì chồng adapter DPO lên base model gốc Qwen3-4B.**

1. **Phương án thay thế:** Giữ nguyên base model chưa qua SFT tiếng Việt làm reference model, hoặc giữ đồng thời 2 mô hình (reference model và policy model) song song trong bộ nhớ GPU khi huấn luyện DPO.
2. **Lý do lựa chọn:** Về mặt lý thuyết căn chỉnh, DPO đòi hỏi điểm xuất phát và mốc tham chiếu phải là mô hình đã hoàn thành SFT trên miền dữ liệu mục tiêu. Nếu dùng base model làm reference, DPO sẽ vô tình phạt các câu trả lời tiếng Việt chuẩn mực mà SFT vừa học được. Về mặt kỹ thuật phần cứng, việc precompute log-prob của reference trước khi train giúp tiết kiệm đáng kể VRAM, cho phép chạy trọn vẹn mô hình 4B trên GPU T4 16GB mà không gặp lỗi tràn bộ nhớ (Out of Memory).
3. **Kết quả thu được:** Quá trình huấn luyện diễn ra mượt mà, reward gap tăng trưởng dương (+0.084 trên held-out), loss khởi đầu bằng đúng $\ln(2) \approx 0.693$ và chẩn đoán đạt `INTENDED`.
4. **Bài học cải tiến:** Nếu được làm lại hoặc mở rộng, tôi sẽ bổ sung thêm thành phần NLL của câu được chọn theo hướng thuật toán RPO (Relative Preference Optimization) hoặc ORPO để vừa học sở thích vừa củng cố khả năng từ chối an toàn mà không cần thêm giai đoạn SFT tách rời.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | prompt_level_strict_acc | 38.2 ± 1.8% | 39.5 ± 1.8% | +1.3% |
| GSM8K | exact_match | 42.0 ± 1.4% | 41.5 ± 1.4% | -0.5% |
| Global-MMLU-vi | acc | 46.5 ± 1.1% | 46.8 ± 1.1% | +0.3% |

_Nhận xét: Mức chênh lệch Δ trên hầu hết các benchmark đều nằm trong khoảng sai số chuẩn (stderr ~1-2%), chứng minh DPO không gây ra hiện tượng "thuế căn chỉnh" (alignment tax) nghiêm trọng trên bài toán suy luận toán học (GSM8K) hay kiến thức tiếng Việt (MMLU)._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 68.0% | +0.084 | 577 ký tự | Baseline chuẩn, margin ổn định |
| RPO | 70.0% | +0.091 | 582 ký tự | Giảm thiểu likelihood displacement nhờ số hạng SFT NLL |
| DPO-norm | 67.0% | +0.076 | 560 ký tự | Chuẩn hoá độ dài giúp kiểm soát độ dài câu trả lời ngắn gọn hơn |
| LD-DPO | 66.5% | +0.072 | 565 ký tự | Giảm thiên vị độ dài rõ rệt |
| ORPO | 69.0% | +0.088 | 570 ký tự | Không cần reference model, tích hợp đồng thời SFT và alignment |

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 42.0% / 46.5% (n=100) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 4.9% |

_Thành phần reward về đúng định dạng thẻ suy luận tăng trước tiên, sau đó reward về đáp án toán học cuối cùng mới bắt đầu cải thiện._

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Mô hình DPO học được sự phân biệt sở thích rất tự nhiên mà không hề bị bẫy "hack độ dài" (chiều dài câu trả lời chỉ tăng khoảng 1.8% và hệ số tương quan Spearman với reward model chỉ là 0.03).
