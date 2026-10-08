# Thử thách thưởng — Xây dựng một thứ có thật (KHÔNG CHẤM ĐIỂM)

**Loại:** Một sân chơi để bạn **đem kiến thức chuyên môn (domain knowledge) của mình vào một mô hình đã căn chỉnh (aligned model) có thật** — không điểm số, không hạn nộp, không thang chấm.
**Vai của bạn:** Bạn vào vai *kỹ sư AI*, xây một mô hình đã căn chỉnh cho một *nhóm người dùng cụ thể*, như làm một sản phẩm thật.
**Thời lượng dự kiến:** 4–8 giờ. Khuyến khích làm theo nhóm 2–3 người. Nghĩ ý tưởng trước, viết mã sau.
**Khuyến khích "vibe coding":** phần mã lặp đi lặp lại để AI lo, còn *lựa chọn về lĩnh vực và mục tiêu ứng dụng* thì bạn tự nghĩ và tự viết.

> Đây là lúc bạn thôi coi DPO như một bài tập trên giấy, mà coi nó là **một công cụ để làm ra thứ có người thật sự dùng**. Phần thưởng thực sự: một sản phẩm bạn có thể đưa vào hồ sơ năng lực và nói "tôi đã xây cái này, người dùng là X, nó giúp làm Y" — chứ không phải "tôi đã chỉnh β=0.05".

---

## 5 gợi ý thử thách — chọn 1, hoặc tự nghĩ ra

Mỗi gợi ý có 4 phần: **Người dùng** (ai dùng) · **Kiến thức chuyên môn** (bạn đem gì vào) · **Mục tiêu ứng dụng** (mô hình làm gì) · **Sản phẩm đầu ra thực tế** (thứ bạn bàn giao được).

### 1. Gia sư cho một môn bạn đang học

> *"Dùng DPO để dạy cách gợi mở từng bước, không phải dạy cách đưa đáp án."*

- **Người dùng:** Học sinh THPT hoặc sinh viên năm nhất đang ôn một môn cụ thể mà bạn giỏi (giải tích, hoá hữu cơ, lịch sử Việt Nam, lập trình Python, vật lý điện từ).
- **Kiến thức chuyên môn:** Bạn hiểu môn đó đủ sâu để phân biệt "câu trả lời sư phạm tốt" với "câu trả lời đưa thẳng đáp án". Đó là yêu cầu cốt lõi.
- **Mục tiêu ứng dụng:** Một mô hình gia sư *không* đưa đáp án — mà gợi ý, hỏi ngược lại, dẫn các công thức theo sách giáo khoa Việt Nam, và tránh dùng thuật ngữ tiếng Anh mà học sinh chưa biết.
- **Sản phẩm đầu ra thực tế:**
  - 200 cặp ưu tiên (prompt = câu hỏi điển hình của học sinh; chosen = câu trả lời gia sư tiếng Việt gợi mở từng bước; rejected = đáp án trực tiếp, pha lẫn tiếng Anh)
  - Adapter DPO + GGUF Q4_K_M chạy được trên CPU laptop của học sinh (`llama-cpp-python`)
  - Bản demo Gradio nhỏ (~50 dòng) để học sinh thử thật
  - Thẻ mô hình (model card): "Gia sư Toán cho học sinh lớp 12 ôn thi THPT — không thay thế sách giáo khoa"

**Câu hỏi để suy nghĩ:**
- Phương pháp sư phạm tốt cho THPT Việt Nam khác Khan Academy của Mỹ thế nào? (cấu trúc đề thi, từ vựng chuẩn trong sách giáo khoa)
- Khi nào mô hình *nên* đưa thẳng đáp án (đã giải xong, học sinh chỉ muốn kiểm tra lại)? Làm sao DPO học được bối cảnh đó?
- Đo "chất lượng dạy kèm" thế nào khi không có giáo viên thật chấm?

---

### 2. Chatbot chăm sóc khách hàng cho một doanh nghiệp Việt có thật

> *"Chọn một cửa hàng. Tưởng tượng 200 câu khách hỏi. Xây một chatbot có thể đem ra dùng thật."*

- **Người dùng:** Khách hàng và chủ của một cơ sở kinh doanh cụ thể — quán cà phê ở Hà Nội, shop quần áo online, tiệm sửa xe máy, trường tiếng Anh nhỏ, homestay ở Sapa. Bạn tự chọn.
- **Kiến thức chuyên môn:** Bạn biết cơ sở đó vận hành ra sao — giờ mở cửa, dịch vụ, giá tham khảo, giọng điệu đặc trưng (thân thiện, lễ phép hay trẻ trung).
- **Mục tiêu ứng dụng:** Một chatbot trả lời các câu hỏi thường gặp đúng giọng thương hiệu, đưa ra bước tiếp theo rõ ràng (số điện thoại, link đặt chỗ, địa chỉ Google Maps), và không tự bịa thông tin mà nó không có.
- **Sản phẩm đầu ra thực tế:**
  - 200 cặp ưu tiên (chosen = đúng giọng thương hiệu + có lời kêu gọi liên hệ; rejected = câu trả lời AI chung chung khô khan, hoặc trả lời bằng tiếng Anh)
  - Adapter DPO + GGUF
  - Triển khai được: một file `serve.py` 30 dòng dùng `llama-cpp-python` + endpoint FastAPI `/chat`
  - Thẻ mô hình có mục "Tôi xây cái này cho: <cơ sở kinh doanh>" + 5 đoạn hội thoại mẫu

**Câu hỏi để suy nghĩ:**
- Bạn xin được 200 tin nhắn khách thật không? (Có người quen mở shop và cho bạn đọc hộp thư Facebook không?)
- Giọng điệu: `anh/chị` trang trọng, `bạn` thân mật, hay xưng `shop/em` kiểu quán trẻ — DPO có học được cách xưng hô không?
- Khi khách hỏi ngoài phạm vi (giá đối thủ, đánh giá tiêu cực), chatbot nên làm gì? Từ chối? Xoa dịu? Chuyển cho chủ shop?

---

### 3. Trợ lý theo sát một nghề — mô hình cho nghề bạn quan sát hằng ngày

> *"Xây cho người không phải là bạn. Xây cho tài xế Grab, không phải cho sinh viên VinUni."*

- **Người dùng:** Một vai trò lao động cụ thể ở Việt Nam — tài xế Grab/be, shipper giao đồ ăn, lễ tân khách sạn, tiểu thương chợ đầu mối, nhà sáng tạo nội dung TikTok, thợ điện nước trong khu phố.
- **Kiến thức chuyên môn:** Bạn quan sát hoặc phỏng vấn một người làm nghề này. Nghề đó có ngôn ngữ riêng, áp lực thời gian riêng và cách ra quyết định khác với công việc văn phòng.
- **Mục tiêu ứng dụng:** Một mô hình trả lời những câu hỏi *người làm nghề thật sự gặp*, chứ không phải "câu hỏi của sinh viên về nghề đó". Nhanh, hướng tới hành động, không giảng giải dài dòng, hiểu bối cảnh nghề.
- **Sản phẩm đầu ra thực tế:**
  - 200 cặp ưu tiên (prompt = câu hỏi thật bạn quan sát được; chosen = ngắn gọn, đưa ra việc cần làm ngay tiếp theo, hiểu bối cảnh; rejected = bài giảng 5 đoạn mặc định của ChatGPT)
  - Adapter DPO + GGUF (thiết kế để chạy trên điện thoại Android tầm trung — người dùng có thể không có laptop)
  - Bản demo nhỏ: một script dòng lệnh để bạn ngồi cạnh người làm nghề thật và thử 5–10 câu hỏi thật
  - Thẻ mô hình: "Dành cho ai" + "Tôi đã quan sát một người làm nghề này trong X giờ, đây là 5 quyết định thiết kế dựa trên đó"

**Câu hỏi để suy nghĩ:**
- Độ dài câu trả lời: tài xế đang lái xe không thể đọc 200 từ. Làm sao để DPO ép câu trả lời dưới 50 từ?
- Tiếng lóng trong nghề (thuật ngữ của shipper: "boom đơn" = khách không nhận hàng, v.v.) — mô hình có hiểu không? Cần dữ liệu nào để bao phủ?
- Bạn có thể *thử với một người làm nghề thật* không? Nếu được — đó là cách đánh giá tốt nhất, hơn mọi bộ chấm điểm GPT-4.

---

### 4. Trợ lý an toàn theo lĩnh vực — mô hình có ranh giới rõ trong một lĩnh vực nhạy cảm

> *"Cân bằng khó nhất: không từ chối quá nhiều, cũng không giúp quá tay. Bạn xây việc căn chỉnh cho một lĩnh vực cụ thể."*

- **Người dùng:** Người dùng Việt Nam nói chung tìm thông tin trong một lĩnh vực nhiều rủi ro — hỗ trợ sức khoẻ tinh thần, thông tin pháp lý cơ bản (luật lao động, dân sự), thông tin cơ bản về thuốc, lập kế hoạch tài chính cá nhân, bot thông tin về tiêm chủng hoặc phòng chống HIV.
- **Kiến thức chuyên môn:** Bạn biết (hoặc tự tìm hiểu) đâu là *thông tin hữu ích* và đâu là *lời khuyên phải để chuyên gia đưa ra*. Bạn biết các đường dây nóng và nguồn chính thức của Việt Nam (các số 1900-XXXX, Bộ Y tế, Tổng đài Phụ nữ).
- **Mục tiêu ứng dụng:** Một mô hình:
  1. Cung cấp thông tin nền + nguồn chính thức
  2. Không đưa lời khuyên cá nhân hoá (kê đơn thuốc, chẩn đoán, tư vấn pháp lý)
  3. Giữ được sự đồng cảm — không lạnh lùng đẩy người dùng đi
  4. Luôn hướng người dùng đến chuyên gia + cung cấp đường dây nóng của Việt Nam khi cần
- **Sản phẩm đầu ra thực tế:**
  - 200 cặp ưu tiên (chosen = thông tin + nguồn + đường dây nóng + chuyển tiếp nhẹ nhàng; rejected = lời khuyên trực tiếp HOẶC từ chối lạnh lùng)
  - Adapter DPO + GGUF
  - **Một thẻ mô hình đặc biệt:** có danh sách rõ ràng "Mô hình này sẽ KHÔNG làm gì" — đây là phần quan trọng nhất. Ghi rõ ranh giới.
  - Bộ kiểm thử 20 câu hỏi: 10 câu lành mạnh nhưng nhạy cảm (mô hình NÊN trả lời kèm nguồn tham khảo), 10 câu vượt ranh giới (mô hình PHẢI từ chối + chuyển tiếp). Báo cáo precision/recall trên cả hai nhóm.

**Câu hỏi để suy nghĩ:**
- "Từ chối đầy đồng cảm" trong tiếng Việt là như thế nào? Nó khác câu "I can't help with that" kiểu Mỹ ra sao?
- Người dùng có thể đang rất căng thẳng — DPO có vô tình khiến mô hình trở nên "máy móc" không? Làm sao tránh?
- Tài liệu tham khảo: Hội Tâm lý học Việt Nam, tổng đài 1900-1567, hoặc thẻ mô hình của Claude (Anthropic) về constitutional AI.

---

### 5. Bắt chước văn phong — mô hình viết giống một người bạn ngưỡng mộ

> *"Chọn một người. 50 mẫu văn. Chuyển giao văn phong bằng DPO."*

- **Người dùng:** Chính bạn (trợ lý viết), hoặc một thương hiệu bạn yêu thích (người viết thuê theo văn phong), hoặc một tổ chức (báo Tuổi Trẻ, VTV24, một kênh TikTok cụ thể).
- **Kiến thức chuyên môn:** Bạn đã đọc đủ nhiều văn của người/thương hiệu đó để *cảm* được phong cách — câu dài hay ngắn, dùng nhiều hay ít từ Hán Việt, hài hước hay nghiêm túc, có dùng emoji hay không, đoạn văn một câu hay năm câu.
- **Mục tiêu ứng dụng:** Một mô hình viết tiếng Việt theo văn phong đó — không phải sao chép nội dung, mà là *khớp giọng văn*.
- **Sản phẩm đầu ra thực tế:**
  - 50–100 mẫu văn theo phong cách đích (bài luận, blog, chuỗi tweet, bản ghi lời thoại từ tin nhắn thoại)
  - 200 cặp ưu tiên (prompt = "viết một đoạn về <chủ đề> theo phong cách X"; chosen = đúng phong cách; rejected = giọng AI chung chung, phẳng lặng). *Chosen có thể là những bản bạn tự biên tập lại — đó là kiến thức chuyên môn sâu nhất bạn đem vào.*
  - Adapter DPO
  - 5 cặp kết quả đặt cạnh nhau: cùng một câu hỏi, đầu ra trước/sau DPO, đưa cho một người hâm mộ phong cách đó *thử mù* (blind test)
  - (Tuỳ chọn) Đẩy lên HF Hub kèm thẻ mô hình ghi rõ nguồn cảm hứng + giấy phép/ghi công

**Câu hỏi để suy nghĩ:**
- Chuyển giao văn phong khác với bắt chước nội dung thế nào? DPO có tách được hai thứ này không, hay học cả hai?
- Đạo đức: nếu bạn bắt chước một nhà văn còn sống, đó là tri ân hay xâm phạm? Thẻ mô hình của bạn nói gì về điều này?
- 50 mẫu có đủ không? Một số nghiên cứu cho thấy chuyển giao văn phong cần ít dữ liệu hơn học nội dung — hãy tự kiểm chứng bằng thực nghiệm.

---

## Hoặc — tự nghĩ ra

Khung mẫu để tự nghĩ ra một gợi ý của riêng bạn, đủ cả 4 phần:

```
NGƯỜI DÙNG:         ai sẽ dùng mô hình này? (cụ thể, không phải "tất cả mọi người")
KIẾN THỨC CHUYÊN MÔN: bạn đem gì vào? (kinh nghiệm cá nhân, kỹ năng,
                    quan hệ với người trong ngành, dữ liệu bạn có
                    quyền dùng)
MỤC TIÊU ỨNG DỤNG: mô hình LÀM GÌ cho người dùng? (một tình huống sử dụng rõ ràng,
                    không phải "cải thiện độ hữu ích" chung chung)
SẢN PHẨM ĐẦU RA:    hình dạng của thứ bạn bàn giao (GGUF? Gradio? CLI? FastAPI?
                    thẻ mô hình trên HF Hub? tích hợp vào một ứng dụng có sẵn?)
```

Một số hướng khác đáng khám phá:

- **Ưu tiên khả năng tiếp cận:** đầu ra thân thiện với trình đọc màn hình, câu ngắn cho người ít chữ, hỗ trợ người nước ngoài đang học tiếng Việt
- **Sắc thái văn hoá vùng miền:** huấn luyện ưu tiên giọng Bắc hay Nam, trang trọng hay thân mật, đúng ngữ cảnh (nhắn với sếp hay với bạn)
- **Kho lưu trữ văn hoá chuyên biệt:** thơ Việt, lịch sử các triều đại, ẩm thực vùng miền, văn học dân gian — một mô hình không bịa (hallucinate) lịch sử Việt Nam
- **Trợ lý đánh giá mã cho một framework:** nhận xét đúng phong cách của FastAPI/React/Laravel
- **Từ giọng nói thành hành động:** bản ghi lời thoại → các việc cần làm, ví dụ ghi chú cuộc họp → danh sách việc cần làm

Gợi ý do bạn tự nghĩ ra thường cho ra sản phẩm có chiều sâu hơn 5 gợi ý có sẵn — vì *bạn* thật sự quan tâm, chứ không phải *tôi* quan tâm.

---

## Danh sách tự kiểm tra cho một bài nộp tốt

Một bài nộp tốt thường có:

- [ ] **`bonus/README.md` dài ≥ 400 từ** trả lời đủ 4 phần: Người dùng / Kiến thức chuyên môn / Mục tiêu ứng dụng / Sản phẩm đầu ra thực tế. Đọc xong trang đầu phải thấy rõ "ai dùng cái này và dùng để làm gì".
- [ ] **Dữ liệu ưu tiên đến từ lĩnh vực của bạn**, không phải UltraFeedback dịch lại. Ít nhất 100 trong 200 cặp do chính bạn tạo, dựa trên phán đoán chuyên môn.
- [ ] **Sản phẩm chạy được**: một người khác (trợ giảng, bạn cùng lớp, người ngoài) tải repo về + chạy một lệnh + trò chuyện được với mô hình.
- [ ] **Thẻ mô hình** có mục rõ ràng "Mô hình này dùng để làm gì / KHÔNG dùng để làm gì / Hạn chế đã biết" — đặc biệt quan trọng với gợi ý 4.
- [ ] **5 lượt tương tác mẫu** trong README để người đọc thấy trực tiếp chất lượng, không chỉ là con số.
- [ ] **Hạn chế trung thực**: một đoạn "Bản thử nghiệm này chưa xử lý được gì" — quyền riêng tư, thiên lệch, quy mô, giấy phép.
- [ ] **(Tuỳ chọn) Đẩy lên HF Hub** kèm README đầy đủ — quy ước: đặt chữ "v0" hoặc "experimental" trong tên mô hình để tránh gây hiểu nhầm là đã sẵn sàng cho môi trường thật.

---

## Định dạng chấp nhận

Tự do. Cấu trúc gợi ý:

```
bonus/
├── README.md            # 1 trang: Người dùng, Lĩnh vực, Mục tiêu, Đầu ra
├── data/
│   ├── prompts.jsonl    # 200 prompt từ lĩnh vực của bạn
│   └── pairs.parquet    # các cặp ưu tiên
├── train.py (hoặc .ipynb) # lượt huấn luyện (dùng lại scripts/train_dpo.py với dữ liệu của bạn)
├── adapters/dpo-bonus/  # đưa vào gitignore — đầu ra adapter
├── demo/
│   ├── serve.py         # một file triển khai được (FastAPI / Gradio / CLI)
│   └── 5-samples.md     # 5 prompt + đầu ra trước/sau DPO
└── MODEL-CARD.md        # tài liệu sẵn sàng bàn giao
```

**Khuyến khích làm theo cặp / nhóm ba** — ghi tên những người tham gia ở đầu `bonus/README.md`.

**Nhật ký quy trình vibe coding** — nếu bạn dùng AI nhiều, hãy ghi ngắn (~100 từ) trong `MODEL-CARD.md`: "một câu lệnh hiệu quả nhất, một câu lệnh thất bại."

---

## Nộp bài

Thêm thư mục `bonus/` vào repo công khai của bạn (cùng repo với Lab 22 chính). Trong `submission/REFLECTION.md`, ghi rõ là bạn đã làm phần thưởng. Người chấm sẽ xem từ cùng đường dẫn LMS công khai.

> Phần thưởng **không** ảnh hưởng điểm chính. Một bài thưởng tốt sẽ được giảng viên đọc kỹ và nhận xét sâu về *khả năng phán đoán* và *tư duy ứng dụng* của bạn — không phải về β, loss hay khoảng cách reward.

---

## "Ai cũng có một lĩnh vực của riêng mình"

Bạn không cần là nhà nghiên cứu AI toàn thời gian mới xây được thứ có ý nghĩa. Bạn biết một môn học, biết một nghề, biết một cộng đồng, biết văn phong của một người. Chừng đó đã là đủ kiến thức chuyên môn để mô hình đã căn chỉnh của bạn *phục vụ một người cụ thể*, thay vì là bản sao ChatBot chung chung thứ 1001.

Hãy mở một PR vào repo gốc nếu bạn muốn chia sẻ với khoá sau.
