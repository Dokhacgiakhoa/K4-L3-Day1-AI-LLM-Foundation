# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Temperature càng thấp thì câu trả lời càng ổn định, ngắn gọn và gần như lặp lại giữa các lần chạy. Temperature càng cao thì phản hồi càng đa dạng, dùng nhiều từ ngữ và ví dụ khác nhau, đôi khi "bay" hơn. Ở 1.5 câu trả lời sáng tạo nhất nhưng cũng dễ lan man, kém nhất quán.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình chọn temperature thấp, khoảng 0.2–0.3. Chatbot hỗ trợ khách hàng cần trả lời nhất quán, chính xác và bám đúng thông tin/chính sách, không nên "sáng tạo" ra nội dung sai. Temperature thấp giúp cùng một câu hỏi luôn nhận được câu trả lời ổn định, dễ kiểm soát chất lượng.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Theo bảng giá output (gpt-4o 0.010 vs gpt-4o-mini 0.0006 USD/1K token), gpt-4o đắt hơn khoảng 16–17 lần cho cùng lượng token. Với workload này (10.000 user × 3 lần × 350 token ≈ 10,5 triệu token/ngày), chênh lệch chi phí rất lớn. GPT-4o xứng đáng khi tác vụ cần suy luận sâu, độ chính xác cao (ví dụ tư vấn pháp lý, phân tích phức tạp); nên dùng mini cho các tác vụ đơn giản, khối lượng lớn như trả lời FAQ hay phân loại tin nhắn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona "giáo viên tiểu học", model trả lời ngắn, dùng từ đơn giản và ví dụ đời thường (như trao đổi đồ chơi, sổ ghi chép). Với persona "chuyên gia tài chính", câu trả lời dài hơn, dùng thuật ngữ kỹ thuật (sổ cái phân tán, hàm băm, đồng thuận) và đi sâu vào cơ chế. Cùng một câu hỏi nhưng system prompt định hình rõ giọng điệu, độ sâu và cách chọn ví dụ. Điều này cho thấy system prompt là công cụ mạnh để điều khiển hành vi model mà không cần đổi câu hỏi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Số token thật từ tiktoken thường cao hơn ước lượng "số từ / 0.75" khoảng 30–60% với đoạn văn tiếng Việt. Lý do là tokenizer được tối ưu cho tiếng Anh, còn tiếng Việt có dấu và ký tự Unicode nên nhiều từ bị tách thành 2–3 token (thậm chí theo từng byte). Vì vậy đếm từ chỉ là ước lượng thô, tiếng Việt cùng độ dài thường tốn nhiều token hơn tiếng Anh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất với các ứng dụng hội thoại thời gian thực (chatbot, trợ lý) và khi câu trả lời dài: người dùng thấy chữ hiện dần ngay lập tức nên cảm giác phản hồi nhanh, không phải chờ đợi màn hình trắng. Ngược lại, non-streaming phù hợp hơn khi cần lấy trọn kết quả rồi mới xử lý — ví dụ gọi API để phân tích, trích xuất JSON, hoặc chạy trong pipeline/batch mà không có người trực tiếp đọc từng chữ.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng dần thời gian chờ (0.1s → 0.2s → 0.4s...) nên giảm dần áp lực lên server đang nghẽn, cho nó thời gian hồi phục. Nếu hàng nghìn client cùng retry với delay cố định giống nhau, tất cả sẽ dội request vào cùng một thời điểm tạo thành "đợt sóng" đồng loạt, khiến server càng quá tải và có thể sập — hiện tượng thundering herd. (Thực tế người ta còn thêm jitter ngẫu nhiên để các client lệch nhịp nhau.)

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona của mình: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt." Yêu cầu "trả lời ngắn gọn" giúp câu trả lời tập trung, dễ đọc trên terminal và tiết kiệm token/chi phí. Chỉ định "tiếng Việt" để đảm bảo model luôn trả lời đúng ngôn ngữ người học, không lẫn tiếng Anh; còn "trợ giảng thân thiện" đặt giọng điệu gần gũi, phù hợp môi trường học tập.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là history chỉ giữ 3 lượt gần nhất, nên trợ lý "quên" thông tin từ đầu cuộc trò chuyện — hỏi lại điều đã nói ở lượt 1 thì nó không nhớ. Một cải thiện: thay vì cắt cứng 6 message, định kỳ tóm tắt các lượt cũ thành một đoạn ngắn rồi đưa vào system prompt (summary memory). Cách triển khai: khi history vượt ngưỡng, gọi thêm một lần API yêu cầu model tóm tắt hội thoại cũ, lưu bản tóm tắt đó và chỉ giữ vài lượt gần nhất ở dạng chi tiết — vừa nhớ được ngữ cảnh dài vừa không để token phình vô hạn.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
