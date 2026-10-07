# Day 22 — DPO/ORPO Alignment Lab (Track 3)

Lab cho **AICB-P2T3 · Ngày 22 · DPO/ORPO Alignment — From SFT to Preference Learning**.

**Bạn sẽ làm:** fine-tune một model nhỏ (SFT) → dạy nó theo dữ liệu preference tiếng Việt bằng **DPO** →
chấm xem bản DPO có tốt hơn bản SFT không.

> Bản K4 cập nhật 10/2026 ([`CHANGELOG.md`](CHANGELOG.md)). Thời gian dưới đây là ước tính trên Colab T4.

---

## 1. Bắt đầu (Colab, không cần cài gì)

1. Tải [`colab/Lab22_DPO_T4.ipynb`](colab/Lab22_DPO_T4.ipynb) lên Colab (Tệp → Tải sổ tay lên).
2. Thời gian chạy → Thay đổi loại thời gian chạy → **T4 GPU**.
3. Chạy lần lượt từ trên xuống. **Dừng sau NB4** nếu chỉ làm phần bắt buộc.
4. Trước khi tắt Colab, tải về máy các thư mục `data/eval`, `adapters/dpo` (chỉ file `.json`) và
   `submission/screenshots` trong `/content/lab22`. Colab xoá mọi file khi hết phiên.

Chạy trên laptop/server có GPU ≥ 12 GB, hoặc A100/L4: xem [`docs/reference.md`](docs/reference.md).

---

## 2. Các bước

| # | Notebook | Làm gì | Thời gian (T4) | Xong khi có |
|---|---|---|---|---|
| 0 | `00_dpo_loss_from_scratch` | Tự viết loss DPO (chạy CPU) | ~10 phút | các `assert` qua |
| 1 | `01_sft_mini` | SFT Qwen3-4B trên 1k mẫu tiếng Việt | ~15–25 phút | `models/sft-merged/` |
| 2 | `02_preference_data` | 800 cặp chosen/rejected tiếng Việt để train + 100 cặp held-out | ~2 phút | `02b-pref-length.png` |
| 3 | `03_dpo_train` | Train DPO | ~40–60 phút | `03-dpo-reward-curves.png`, `dpo_metrics.json` |
| 4 | `04_compare_and_eval` | So SFT vs SFT+DPO, chấm tự động (không cần API key) | ~20–30 phút | `judge_summary.json` |

**Bắt buộc: NB0–NB4.** Bonus (không bắt buộc): NB3b biến thể DPO/ORPO · NB5 xuất GGUF · NB6 benchmark ·
NB7 GRPO. Xem [`BONUS-CHALLENGE.md`](BONUS-CHALLENGE.md).

**Đọc kết quả NB3:** `rewards/margins` phải tăng. `rewards/chosen` giảm nhẹ là bình thường (likelihood
displacement); ghi nhận xét vào REFLECTION.

---

## 3. Nộp bài

1. Copy repo lên GitHub của bạn, để **public**.
2. Commit notebook NB0–NB4 **còn giữ output**, các ảnh trong `submission/screenshots/` và các file kết quả ở bước 1.4.
3. Điền [`submission/REFLECTION.md`](submission/REFLECTION.md) bằng số liệu thật của bạn.
4. Chạy `make verify` (nếu chạy local), rồi nộp URL repo vào LMS.

Bài được chấm tự động từ repo: chỉ file đã commit mới được tính. `.gitignore` đã chặn trọng số model;
đừng commit `.env`. Tiêu chí chấm: [`rubric.md`](rubric.md).

---

## 4. Gặp lỗi?

| Triệu chứng | Cách xử lý |
|---|---|
| Hết VRAM (OOM) | Thêm `os.environ["MAX_LEN"] = "512"` vào cell setup đầu tiên, chạy lại từ đầu |
| NB3 chạy rất lâu | Bình thường (~40–60 phút). Chỉ muốn thử luồng: thêm `os.environ["PREF_TRAIN"] = "200"` vào cell setup |
| Hết giờ GPU Colab | File trong phiên bị mất; tải kết quả về trước (bước 1.4), phiên mới phải chạy lại từ NB1 |
| Lỗi khác | [`docs/reference.md`](docs/reference.md), mục "Lỗi thường gặp" |

---

Code: MIT ([`LICENSE`](LICENSE)). Dữ liệu, giấy phép, tech stack và lời cảm ơn: [`docs/reference.md`](docs/reference.md).

© VinUniversity AICB program · Track 3 Day 22.
