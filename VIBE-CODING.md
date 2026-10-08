# Vibe Coding — Mẹo cho lập trình viên hiện đại

> Đọc 10–15 phút.
>
> **Phần 1** là phần nhập môn chung — quy trình vibe coding nói chung.
> **Phần 2** đi sâu vào **Claude Code**, trợ lý AI của Anthropic chạy trong terminal — được khuyên dùng cho bài lab này và mọi công việc phần mềm khác.

---

# Phần 1: Vibe coding nói chung

## Vibe coding là gì?

**Vibe coding** (Andrej Karpathy, 02/2025) — bạn để LLM viết phần lớn code, còn bạn đảm nhận vai *kiến trúc sư* và *người rà soát*: mô tả ý định → rà soát phần thay đổi (diff) → chấp nhận hoặc từ chối. Bạn không gõ từng dòng vòng lặp `for`; bạn đặt ra đặc tả (spec) rõ ràng và đảm bảo không có lỗi ngầm trong phần diff trả về.

Vibe coding ≠ "sao chép rồi dán từ ChatGPT". Vibe coding là một *quy trình làm việc*:

```
   ý định (spec)
      ↓
   viết lời nhắc cho LLM
      ↓
   rà soát diff (không bỏ qua!)
      ↓
   chạy + kiểm chứng
      ↓
   commit hoặc hoàn tác
```

Bỏ qua bất kỳ bước nào → vibe coding biến thành "đánh bạc với code".

---

## 2 phong cách kỷ luật: SDD và TDD

### Phát triển dựa trên đặc tả (Spec-Driven Development, SDD)

> Viết **đặc tả** trước, code sau. Đặc tả là bản giao kèo giữa bạn và LLM.

**Một đặc tả đầy đủ** thường gồm:
- *Đầu vào:* tên + kiểu + ràng buộc của mỗi tham số
- *Đầu ra:* hình dạng + kiểu + các bất biến
- *Hành vi:* trường hợp biên, lỗi, tác dụng phụ
- *Ràng buộc:* ngân sách độ trễ, giới hạn bộ nhớ, thư viện cấm dùng

LLM viết code khớp đặc tả. Bạn rà soát diff để kiểm chứng từng dòng có triển khai đúng đặc tả. Đặc tả mơ hồ → code mơ hồ → mất 1 giờ gỡ lỗi.

### Phát triển hướng kiểm thử (Test-Driven Development, TDD) trong thời đại LLM

> Viết **bài kiểm thử** trước, code sau. Bài kiểm thử là đặc tả dưới dạng máy chấm.

```
Vòng 1: "Write a pytest test for <function> that asserts <invariants>.
         Don't write the implementation yet — only the test."
```

Bài kiểm thử "đạt do cách xây dựng" (bạn chạy thì nó thất bại vì chưa có code). Sau đó:

```
Vòng 2: "Now implement <function> such that the test passes."
```

Phần triển khai do vibe code sinh ra, nhưng **bài kiểm thử thì không đổi**. Nếu bài kiểm thử sai (ví dụ khẳng định sai logic), bạn phát hiện ngay từ vòng 1, không phải sau khi triển khai lên môi trường thật.

TDD đặc biệt mạnh với vibe coding vì LLM hay bịa ra trường hợp biên — các bài kiểm thử đóng vai trò hàng rào chống ảo giác.

---

## Khi nào vibe code, khi nào tự nghĩ?

| Vibe code thoải mái | Tự nghĩ kỹ trước khi viết lời nhắc |
|---|---|
| Khung sườn cho API route (FastAPI, Express, …) | Lựa chọn thuật toán / cấu trúc dữ liệu cốt lõi |
| Schema Pydantic / Zod / TypeScript | Mô hình đồng thời (khóa vs không khóa vs CAS) |
| Khung sườn kiểm thử (pytest fixtures, mocks) | Ngữ nghĩa khi thất bại (thử lại, tính lũy đẳng) |
| Tệp cấu hình (YAML, JSON, env) | Di chuyển schema / tương thích ngược |
| Khung README, docstring | Ranh giới bảo mật (xác thực, cô lập) |
| Bộ sinh dữ liệu tổng hợp / fixtures | Đánh đổi về ngân sách hiệu năng |
| Xử lý lỗi cho I/O (try/except mẫu) | Chiến lược vô hiệu hóa cache |
| Tái cấu trúc "đổi tên field X → Y" trên cả repo | Kiến trúc (vector vs đồ thị, nguyên khối vs vi dịch vụ) |

**Quy tắc đơn giản:** nếu lỗi sẽ là *suy giảm âm thầm* (hệ thống vẫn chạy nhưng kém hơn, không báo lỗi rõ ràng) thay vì *thất bại ồn ào* (exception, kiểm thử fail), đó là **vùng phải suy nghĩ kỹ**. Đừng để LLM tự quyết.

---

## 5 mẫu viết lời nhắc phổ quát

### 1. Đưa đặc tả vào, nhận code ra

> Càng hẹp → diff càng gọn, càng ít phải lặp lại.

```
[MƠ HỒ — KHÔNG NÊN]
"Write a function to validate emails"

[HẸP — NÊN]
"Function: validateEmail(addr: str) -> bool
Inputs: addr — non-empty string up to 254 chars
Output: bool — True if matches RFC 5322 simplified regex
Examples: 'user@example.com' → True; 'invalid' → False; 'a@.com' → False
Constraints: pure function, no I/O, no external libs"
```

### 2. Kiểm chứng trước khi sinh code

> Với công thức / thuật toán: hỏi AI giải thích, đối chiếu chéo, rồi mới nhờ triển khai.

```
Step 1: "Explain Reciprocal Rank Fusion. Formula? Rank 0-based or 1-based? k=?"
Step 2: Bạn đối chiếu câu trả lời với tài liệu tham khảo (bài báo / tài liệu / giáo trình).
Step 3: "Implement search_hybrid(...) per the formula above. rank is 1-based, k=60."
```

Nhiều AI ảo giác viết công thức sai mà code vẫn chạy — suy giảm âm thầm rất khó gỡ lỗi.

### 3. Kiểm thử trước, code sau (TDD)

> Bài kiểm thử là đặc tả dạng máy chấm. Viết kiểm thử trước → code phải vượt qua kiểm thử.

```
"Write a pytest test that asserts X. Don't write implementation yet."
```

Sau khi bài kiểm thử được viết đúng (chạy "đạt do cách xây dựng" = thất bại), hãy yêu cầu triển khai.

### 4. Bản tái hiện tối thiểu → mở rộng dần

> Đừng yêu cầu LLM viết toàn bộ tính năng trong 1 lời nhắc. Xây dựng từng bước.

```
Step A: "Write minimal X with 1 feature."
Step B: "Run + verify."
Step C: "Now extend X to handle case Y."
Step D: "Now wrap in benchmark/test loop."
```

LLM ít ảo giác hơn khi ngữ cảnh đã có sẵn một nền chạy được.

### 5. Vòng lặp lập kế hoạch → code → rà soát

> 3 vòng: AI đề xuất 3 hướng tiếp cận → bạn chọn → AI triển khai → bạn rà soát.

```
Vòng 1: "Propose 3 approaches to do X. Compare on (cost, complexity, scalability)."
Vòng 2: Bạn chọn 1: "Use approach #2 because Z."
Vòng 3: "Implement approach #2 + write test."
Vòng 4: Bạn rà soát diff từng dòng một.
```

Đừng bỏ qua vòng 1 — bạn sẽ kẹt trong cực trị cục bộ mà LLM nghĩ ra đầu tiên.

---

## 3 phản mẫu phổ biến

### 1. Hỏi AI quyết định kiến trúc

❌ "Which embedding model should I use?"
→ AI chọn mặc định có trong dữ liệu huấn luyện, không biết tập văn bản của bạn.

✅ "I have a 1M-doc Vietnamese corpus, GPU=A10, latency budget=20ms. List 3 candidate embedding models with (MTEB-vi score, dim, RAM/1M vecs, cost). Recommend top 1, explain why."

### 2. Sinh code rồi tin luôn, không kiểm thử

❌ Chấp nhận code do AI viết → commit → push → phát hiện lỗi trên môi trường thật.

✅ AI sinh code → bạn chạy kiểm thử → kiểm thử đạt → rà soát diff → commit. Nếu chưa có kiểm thử, hãy viết kiểm thử trước (mẫu số 3 ở trên).

### 3. "Làm cho nhanh hơn" mà không có con số

❌ "Make this latency faster"
→ AI tối ưu ngẫu nhiên, có thể còn chậm hơn.

✅ "P99 hiện tại = 87ms (measured by `<command>`). Target < 50ms. Profile shows 60% time in `<function>`. Suggest 3 optimizations with expected speedup."

### 4. Phần thưởng thêm — lời nhắc thiếu ngữ cảnh

❌ "Fix this bug"

✅ Dán nguyên văn thông báo lỗi + kết quả mong đợi + bản tái hiện tối thiểu + đường dẫn các tệp liên quan + commit gần nhất còn chạy tốt. Đầu vào mơ hồ → đầu ra mơ hồ.

---

## Quy trình điển hình cho 1 tác vụ

```
1. Đọc / viết spec (5 phút)         → bạn nghĩ
2. Lập kế hoạch: vùng phải nghĩ kỹ? (1 phút)  → bạn nghĩ
3. Viết lời nhắc với spec rõ ràng    → AI sinh
4. Rà soát diff từng dòng            → bạn xác minh
5. Chạy kiểm thử / đo hiệu năng      → máy kiểm chứng
6. Commit hoặc hoàn tác              → bạn quyết định
```

Không bỏ qua bước 4 và 5. Đó là chỗ vibe coding thất bại âm thầm thay vì thất bại ồn ào.

---

# Phần 2: Claude Code

[Claude Code](https://code.claude.com) là trợ lý lập trình có tính tác tử (agentic) của Anthropic chạy trong terminal. Đây là *công cụ vibe coding* tốt nhất năm 2026 cho công việc phần mềm nghiêm túc — kế hoạch nhiều tệp, chỉnh sửa cẩn thận, chế độ lập kế hoạch + danh sách tác vụ + vòng lặp rà soát ngay trong terminal. Bài lab này khuyến khích dùng Claude Code (hoặc Codex CLI / OpenCode tương tự); README sẽ giả định bạn có 1 trong 3.

> Tham khảo gốc: [code.claude.com/docs/en/how-claude-code-works](https://code.claude.com/docs/en/how-claude-code-works) · [memory](https://code.claude.com/docs/en/memory) · [permission-modes](https://code.claude.com/docs/en/permission-modes) · [common-workflows](https://code.claude.com/docs/en/common-workflows) · [best-practices](https://code.claude.com/docs/en/best-practices) · [claude-directory](https://code.claude.com/docs/en/claude-directory)

---

## Khái niệm cốt lõi

### Cách Claude Code hoạt động — vòng lặp tác tử

Claude Code không phải chatbot trả lời rồi ngồi đợi. Nó **tự lặp** qua 3 giai đoạn: **thu thập ngữ cảnh → hành động → kiểm chứng kết quả**, lặp lại đến khi tác vụ xong:

```
Lời nhắc của bạn → Claude thu thập ngữ cảnh (đọc tệp, grep)
            → Claude hành động (chỉnh sửa, chạy lệnh)
            → Claude kiểm chứng (chạy kiểm thử, xem kết quả)
            → lặp lại đến khi hoàn thành tác vụ
            (bạn có thể ngắt bất cứ lúc nào)
```

Vòng lặp thích nghi theo tác vụ. Một câu hỏi về codebase chỉ cần ngữ cảnh. Sửa một lỗi thì lặp nhiều lần. Một đợt tái cấu trúc cần kiểm chứng kỹ lưỡng. Claude tự quyết định mỗi bước cần gì dựa trên kết quả của bước trước.

Bạn cũng là một phần của vòng lặp — bấm `Esc` để ngắt, gõ chỉnh sửa, Claude điều chỉnh ngay mà không phải khởi động lại.

**Các thành phần:**
- **Mô hình**: Sonnet (mặc định cho lập trình), Opus (suy luận phức tạp). Chuyển đổi bằng `/model`.
- **Công cụ**: 5 nhóm — Thao tác tệp · Tìm kiếm · Thực thi · Web · Thông minh hóa code. Mỗi lần gọi công cụ trả về thông tin → đưa ngược vào vòng lặp.

### Mở rộng Claude Code

5 cơ chế để thêm khả năng cho Claude Code, từ nhẹ → nặng:

| Cơ chế | Bổ sung gì | Khi nào dùng |
|---|---|---|
| **CLAUDE.md** | Chỉ dẫn tĩnh Claude đọc mỗi phiên | Quy ước dự án, lệnh thường dùng, quy tắc "luôn làm X" |
| **Skills** (`.claude/skills/`) | Kiến thức chuyên ngành + quy trình lặp lại được, nạp khi cần | Mẫu, script, tác vụ nhiều bước gọi bằng `/<skill-name>` |
| **Hooks** (`.claude/settings.json`) | Lệnh shell chạy tự động ở các sự kiện vòng đời | Định dạng khi lưu, lint sau khi chỉnh sửa, chặn chỉnh sửa vào một số đường dẫn |
| **MCP servers** | Kết nối tới dịch vụ bên ngoài | Truy vấn cơ sở dữ liệu, GitHub API, Notion, Figma, Sentry |
| **Subagents** (`.claude/agents/`) | Trợ lý chuyên biệt có cửa sổ ngữ cảnh riêng | Rà soát code, kiểm toán bảo mật, nghiên cứu song song |
| **Plugins** | Gói gồm skills + hooks + MCP + subagents | Cài gói từ cộng đồng — `/plugin` để duyệt |

**Chọn tính năng theo mục tiêu:**
- "Luôn chạy X" → **hook** (mang tính xác định)
- "Thỉnh thoảng cần Y" → **skill** (nạp khi cần)
- "Kết nối tới dịch vụ Z" → **MCP**
- "Rà soát/tái cấu trúc song song" → **subagent** (ngữ cảnh riêng)

### Khám phá thư mục `.claude/`

Cấu trúc thư mục Claude Code đọc, ở 2 cấp:

**Cấp dự án** (`./.claude/` — được commit vào git, chia sẻ với nhóm):
```
your-project/
├── CLAUDE.md                    # Chỉ dẫn dự án, nạp mỗi phiên
├── .claude/
│   ├── CLAUDE.md                # Vị trí thay thế (giống ./CLAUDE.md)
│   ├── settings.json            # Quyền, hooks, MCP servers
│   ├── settings.local.json      # Ghi đè cá nhân (nằm trong .gitignore)
│   ├── rules/                   # Chỉ dẫn theo đường dẫn (nạp khi mở tệp khớp)
│   │   ├── api-design.md
│   │   └── testing.md
│   ├── agents/                  # Subagent tùy chỉnh
│   ├── skills/                  # Skills + quy trình dùng lại được
│   ├── commands/                # Lệnh gạch chéo kiểu cũ (nên dùng skills thay thế)
│   └── hooks/                   # Script cho hook
└── CLAUDE.local.md              # Ghi chú dự án cá nhân (nằm trong .gitignore)
```

**Cấp người dùng** (`~/.claude/` — áp dụng cho mọi dự án):
```
~/.claude/
├── CLAUDE.md                    # Sở thích cá nhân dùng chung mọi dự án
├── settings.json                # Cài đặt toàn cục cá nhân
├── skills/                      # Skills cá nhân
├── agents/                      # Subagent cá nhân
└── projects/<project>/memory/   # Bộ nhớ tự động (theo từng repo)
    ├── MEMORY.md                # Mục lục ngắn gọn, nạp mỗi phiên
    └── <topic>.md               # Tệp chi tiết, nạp khi cần
```

**Thứ tự nạp:** chính sách được quản lý → toàn cục của người dùng → các thư mục cha → gốc dự án → thư mục đang làm việc → CLAUDE.local.md sau cùng (gần thư mục hiện tại nhất thì thắng).

### Khám phá cửa sổ ngữ cảnh

Cửa sổ ngữ cảnh = bộ nhớ làm việc. Mỗi token Claude đọc đều tiêu tốn ngữ cảnh. Khi đầy, Claude sẽ nén (tóm tắt ngữ cảnh cũ hơn). Điều quan trọng:

**Được nạp khi bắt đầu phiên:**
- Lời nhắc hệ thống
- CLAUDE.md (toàn bộ nội dung, mọi cấp cha)
- 200 dòng / 25 KB đầu của `MEMORY.md` (bộ nhớ tự động)
- Mô tả của skill (nội dung đầy đủ chỉ nạp khi được gọi)
- Định nghĩa công cụ

**Phình ra khi bạn làm việc:**
- Lịch sử hội thoại
- Nội dung tệp (mỗi lần gọi công cụ `Read` sẽ thêm tệp vào)
- Kết quả đầu ra của lệnh
- Bản tóm tắt subagent trả về

**Chiến lược quản lý:**
- Chạy `/context` — xem cái gì đang chiếm chỗ
- `/clear` — xóa sạch giữa các tác vụ không liên quan
- Subagents — việc khám phá chạy trong ngữ cảnh *riêng*, bạn chỉ nhận bản tóm tắt
- Skills với `disable-model-invocation: true` — mô tả không được nạp cho đến khi dùng
- Tham chiếu `@file.py` nạp tệp một lần vào cuộc hội thoại
- Đặt các quy tắc cần duy trì vào CLAUDE.md (chịu được việc nén), không để trong cuộc trò chuyện

**Khi ngữ cảnh đầy:** tự động nén sẽ kích hoạt. Thêm mục "Compact Instructions" vào CLAUDE.md để kiểm soát những gì được giữ lại. Dùng `/compact focus on the API changes` để nén có trọng tâm.

---

## Sử dụng Claude Code

### Lưu chỉ dẫn và ký ức

Hai cơ chế song song:

#### CLAUDE.md (bạn viết)
- Tệp markdown thuần (không có schema)
- Được nạp khi bắt đầu mỗi phiên
- Chạy `/init` để tự động sinh bản khởi đầu từ codebase của bạn
- Giữ dưới 200 dòng (tệp dài hơn sẽ bị bỏ qua một phần)
- **Nên có**: lệnh build, phong cách code khác mặc định, hướng dẫn kiểm thử, quy tắc ứng xử của repo, điểm đặc thù của dự án
- **Không nên có**: bất cứ gì Claude đọc được từ code, quy ước chuẩn của ngôn ngữ, mô tả từng tệp một
- Kiểm tra bằng cách xóa một dòng — nếu Claude mắc lỗi thì giữ lại; nếu không thì lược bỏ

**Vị trí** (cụ thể hơn thì thắng):
- `~/.claude/CLAUDE.md` — sở thích cá nhân toàn cục
- `./CLAUDE.md` hoặc `./.claude/CLAUDE.md` — dự án dùng chung (hãy commit)
- `./CLAUDE.local.md` — dự án cá nhân (hãy đưa vào gitignore)
- `CLAUDE.md` trong thư mục con — nạp khi cần lúc Claude đọc thư mục con đó

#### Bộ nhớ tự động (Claude viết)
- Nằm tại `~/.claude/projects/<repo>/memory/`
- Claude lưu ghi chú khi bạn sửa nó hoặc khi nó phát hiện một mẫu của dự án
- Bật/tắt trong `/memory`
- 200 dòng đầu của `MEMORY.md` được nạp mỗi phiên; các tệp chủ đề nạp khi cần
- Markdown thuần — chỉnh sửa/xóa thoải mái

#### Lệnh `/memory`
- Liệt kê CLAUDE.md + các tệp bộ nhớ tự động đã nạp
- Bật/tắt bộ nhớ tự động
- Mở thư mục bộ nhớ tự động

#### Phím tắt `#`
- Gõ `#` ở đầu bất kỳ tin nhắn nào — Claude lưu nó thành một mục ghi nhớ

#### Nhập tệp bằng `@`
- `@README.md` bên trong CLAUDE.md sẽ chèn nội dung tệp được tham chiếu lúc nạp
- Dùng để kéo vào chỉ dẫn dùng chung: `@~/.claude/my-prefs.md`

### Chế độ cấp quyền

Nhấn `Shift+Tab` để chuyển vòng. Tổng cộng 6 chế độ:

| Chế độ | Việc chạy không cần hỏi | Phù hợp nhất cho |
|---|---|---|
| `default` | Chỉ đọc | Công việc nhạy cảm, mới làm quen |
| `acceptEdits` | Đọc, chỉnh sửa tệp, lệnh hệ thống tệp thông dụng (`mkdir`, `mv`, `cp`) | Xem lại thay đổi qua `git diff` sau đó |
| `plan` | Chỉ đọc — Claude đề xuất kế hoạch, không chỉnh sửa cho đến khi bạn duyệt | Khám phá trước khi thay đổi, tính năng nhiều tệp |
| `auto` | Mọi thứ (có bộ phân loại kiểm tra an toàn) | Tác vụ dài không bị gián đoạn; yêu cầu gói Max/Team/Enterprise |
| `dontAsk` | Chỉ các công cụ đã duyệt trước (danh sách cho phép) | Quy trình CI, môi trường bị khóa chặt |
| `bypassPermissions` | Mọi thứ, không kiểm tra | **NGUY HIỂM** — chỉ dùng trong container/máy ảo cô lập |

**Mẹo quy trình — chế độ `plan`:**
1. `Shift+Tab` hai lần → `plan` (chỉ đọc)
2. Nhờ Claude đọc code + đề xuất kế hoạch
3. Xem lại kế hoạch, tinh chỉnh qua trò chuyện
4. Duyệt và thoát chế độ `plan` → Claude triển khai
5. Nhấn `Esc Esc` để tua lại nếu có gì sai

**Cấu hình mặc định bền vững** trong `.claude/settings.json`:
```json
{ "permissions": { "defaultMode": "acceptEdits" } }
```

**Danh sách cho phép cụ thể** (bỏ qua hỏi cho các lệnh đáng tin):
```json
{ "permissions": { "allow": ["Bash(npm test:*)", "Bash(git status)"] } }
```

### Quy trình làm việc thông dụng

#### Khám phá → Lập kế hoạch → Code → Kiểm thử
Quy trình chủ lực:
```
[plan mode]   read /src/auth and explain how sessions work
[plan mode]   create a plan for adding Google OAuth
[default]     implement the plan, write tests, run them
[default]     commit with descriptive message and open a PR
```

#### Sửa lỗi hiệu quả
```
I'm seeing this error when I run npm test: <paste>
suggest 3 ways to fix the @ts-ignore in user.ts
update user.ts to add the null check you suggested
```

#### Tái cấu trúc code
```
find deprecated API usage in our codebase
suggest how to refactor utils.js to use modern JavaScript features
refactor utils.js to use ES2024 features while maintaining same behavior
run tests for the refactored code
```

#### Làm việc với hình ảnh
- Kéo-thả hình ảnh vào CLI
- Hoặc dán bằng `Ctrl+V` (KHÔNG phải `Cmd+V`)
- Hoặc truyền đường dẫn: `Analyze this image: /path/to/image.png`
- Hữu ích: dán ảnh chụp màn hình lỗi, bản phác thảo giao diện, sơ đồ kiến trúc

#### Tham chiếu tệp bằng `@`
- `@src/utils/auth.js` — chèn nội dung tệp
- `@src/components/` — liệt kê thư mục
- `@github:repos/owner/repo/issues` — lấy tài nguyên MCP

#### Tiếp tục / rẽ nhánh một phiên
- `claude --continue` — tiếp tục phiên gần nhất
- `claude --resume` — chọn từ danh sách
- `/branch` hoặc `--fork-session` — sao chép lịch sử sang phiên mới
- Hữu ích khi một tác vụ kéo dài qua nhiều buổi làm việc

#### Chạy các phiên song song (worktree)
- `claude --worktree feature-auth` ở terminal A
- `claude --worktree bugfix-login` ở terminal B
- Mỗi bản checkout tách biệt, không đụng độ khi chỉnh sửa
- Duyệt bằng các phím tắt của `/resume`

#### Dẫn Claude vào script qua đường ống
```bash
claude -p "summarize these commits" < git_log.txt
git log --oneline -20 | claude -p "what changed?"
```
Dùng trong CI, hook tiền commit, xử lý hàng loạt.

#### Lệnh gạch chéo
- `/init` — sinh CLAUDE.md khởi đầu
- `/clear` — đặt lại ngữ cảnh
- `/compact <focus>` — nén có trọng tâm
- `/context` — hiển thị cái gì đang chiếm chỗ ngữ cảnh
- `/permissions` — quản lý quy tắc cho phép/hỏi/từ chối
- `/agents` — cấu hình subagent
- `/doctor` — chẩn đoán cài đặt
- `/memory` — xem + chỉnh sửa ký ức
- `/rewind` (hoặc `Esc Esc`) — hoàn tác về trạng thái trước

### Thực hành tốt nhất

Mẹo có đòn bẩy cao nhất: **cho Claude một cách để tự kiểm chứng công việc của nó.** Kiểm thử, kết quả mong đợi, ảnh chụp màn hình — bất cứ thứ gì Claude chạy được để tự kiểm tra.

| Trước | Sau |
|---|---|
| "implement validateEmail" | "implement validateEmail. Test cases: 'user@example.com' → True, 'invalid' → False, 'a@.com' → False. Run the tests after." |
| "make the dashboard look better" | "[paste screenshot] implement this design. Take a screenshot of the result and compare. List differences and fix them." |
| "the build is failing" | "build fails with: <paste error>. Fix and verify it succeeds. Address root cause, don't suppress." |

**Các nguyên tắc khác:**

1. **Khám phá trước khi triển khai.** Dùng chế độ `plan` cho mọi việc trải rộng nhiều tệp. Chỉ bỏ qua lập kế hoạch với các lỗi sửa một dòng.
2. **Cụ thể ngay từ đầu.** Tham chiếu tệp (`@src/auth.ts`), nêu ràng buộc, chỉ vào mẫu có sẵn ("follow HotDogWidget.php's approach").
3. **Ủy thác, đừng ra lệnh từng li.** Đưa ngữ cảnh + ý định, để Claude tự tìm ra cần đọc tệp nào.
4. **Chỉnh hướng sớm.** Nhấn `Esc` để ngắt ngay khi Claude đi sai. Đừng đợi nó chạy hết một hướng sai.
5. **`/clear` giữa các tác vụ không liên quan.** Ngữ cảnh cũ = hiệu năng suy giảm.
6. **Dùng subagent để điều tra.** Đọc 50 tệp bằng subagent chỉ thêm bản tóm tắt vào ngữ cảnh chính của bạn.
7. **Giữ CLAUDE.md gọn gàng.** Giữ dưới 200 dòng. Lược bỏ không thương tiếc. Nếu Claude đã làm đúng X rồi thì xóa quy tắc về X.
8. **Quy tắc hai lần sửa thất bại.** Nếu bạn đã sửa cùng một vấn đề hai lần, hãy `/clear` và viết lời nhắc ban đầu sắc hơn thay vì sửa lần thứ ba.
9. **Tin nhưng phải kiểm chứng.** Code trông hợp lý vẫn có thể giấu trường hợp biên. Luôn chạy kiểm thử / lint / kiểm tra kiểu trước khi gộp.
10. **Dùng `gh` (hoặc công cụ CLI khác).** Bảo Claude dùng `gh` cho GitHub, `aws` cho AWS, v.v. Tiết kiệm token và đã lo sẵn xác thực.

---

## Thiết lập khởi đầu được khuyên dùng cho bài lab này

Riêng cho Lab 22 (hoặc bất kỳ bài lab Day 19/20/22 nào):

1. **Cài Claude Code** — `npm install -g @anthropic-ai/claude-code` rồi chạy `claude` ở thư mục gốc repo
2. **Chạy `/init`** — sinh `CLAUDE.md` khởi đầu từ repo này
3. **Thêm 3 dòng** vào CLAUDE.md vừa sinh:
   ```
   - When editing notebook .py files (jupytext py:percent), preserve the `# %%` cell markers
   - Run `make verify` before suggesting submission readiness
   - VRAM math, hyperparameter values, and dataset names should match the deck (`day07-...tex`); flag deviations
   ```
4. **Dùng chế độ `plan`** (`Shift+Tab` hai lần) trước mọi đợt tái cấu trúc nhiều tệp
5. **Dùng `/clear`** khi chuyển từ "chuẩn bị dữ liệu" sang "huấn luyện" rồi sang "triển khai"
6. **Dùng một subagent** cho việc "điều tra vì sao khoảng cách phần thưởng (reward gap) của tôi bị âm" — giữ ngữ cảnh chính của bạn sạch sẽ

---

## Các công cụ CLI thay thế

Bài lab này dùng được với mọi CLI vibe coding:

| Công cụ | Mạnh nhất ở | Tệp dự án |
|---|---|---|
| **Claude Code** (Anthropic) | Kế hoạch nhiều tệp, chỉnh sửa cẩn thận, suy luận dài hơn, chế độ lập kế hoạch ngay trong terminal | `CLAUDE.md` |
| **Codex CLI** (OpenAI) | Lặp nhanh, họ GPT/o1, chế độ tác tử chạy lệnh thật | `AGENTS.md` |
| **OpenCode** (mã nguồn mở) | Đa nhà cung cấp (Anthropic/OpenAI/Ollama cục bộ), không bị khóa vào một hãng | `AGENTS.md` |
| **Cursor** (IDE) | Quy trình giao diện đồ họa, diff ngay trong dòng, chỉnh sửa nhiều con trỏ | `.cursorrules` |

Đa số công cụ CLI đọc dự phòng sang `AGENTS.md` nếu không có tệp riêng, nên 1 tệp `AGENTS.md` thường đủ cho cả 3 công cụ CLI. Riêng Claude Code đọc `CLAUDE.md` hoặc nhập `AGENTS.md` qua `@AGENTS.md`.

---

## Đọc thêm

- Andrej Karpathy — "Vibe coding" tweet (02/2025)
- Simon Willison — "Vibe coding is here, and it's pretty cool" (02/2025)
- [Claude Code docs](https://code.claude.com/docs) — chính thức, luôn cập nhật
- Anthropic — "Effective coding with Claude" engineering blog
- Geoffrey Litt — "Malleable software" essay (2024) — bối cảnh cho lý do vibe coding hiệu quả
