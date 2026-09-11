"""
Giao diện review localhost cho Lab 01 — chạy: streamlit run app.py

Đây KHÔNG phải file nộp bài (bài chấm chỉ đọc template.py / solution/).
App gọi lại đúng các hàm bạn đã viết trong template.py để bạn thử trực tiếp
trên trình duyệt: so sánh model, đếm token / ước tính chi phí, và chat nhiều
lượt có streaming + history + thống kê.
"""

import os

import streamlit as st
from dotenv import load_dotenv

import template as lab  # dùng lại các hàm đã viết trong template.py

load_dotenv()

st.set_page_config(page_title="Lab 01 — LLM API Review", page_icon="🤖", layout="wide")

st.title("🤖 Lab 01 — Review LLM API")
st.caption("Giao diện thử nhanh các hàm trong template.py. Cần API key trong .env để gọi thật.")

# --- Trạng thái key ---
has_key = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("OPENAI_API_KEY") != "sk-your-key-here"
with st.sidebar:
    st.subheader("Cấu hình")
    st.write(f"Model lớn: `{lab.OPENAI_MODEL}`")
    st.write(f"Model nhỏ: `{lab.OPENAI_MINI_MODEL}`")
    st.write(f"Base URL: `{os.getenv('OPENAI_BASE_URL', 'mặc định OpenAI')}`")
    if has_key:
        st.success("Đã có API key — gọi thật được.")
    else:
        st.warning("Chưa có API key trong .env. Phần đếm token vẫn chạy; "
                   "phần gọi model (so sánh / chat) sẽ báo lỗi khi bấm.")

tab_compare, tab_token, tab_chat = st.tabs(
    ["⚖️ So sánh model", "🔢 Token & chi phí", "💬 Trợ lý CLI"]
)

# ---------------------------------------------------------------------------
# Tab 1 — compare_models
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("So sánh model lớn vs nhỏ (Part 1)")
    prompt = st.text_area("Prompt", "Việt Nam có bao nhiêu tỉnh thành?", key="cmp_prompt")
    if st.button("Chạy so sánh", disabled=not has_key):
        with st.spinner("Đang gọi cả hai model..."):
            try:
                r = lab.compare_models(prompt)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**{lab.OPENAI_MODEL}** · {r['gpt4o_latency']:.2f}s")
                    st.info(r["gpt4o_response"])
                    st.metric("Chi phí output ước tính", f"${r['gpt4o_cost_estimate']:.6f}")
                with c2:
                    st.markdown(f"**{lab.OPENAI_MINI_MODEL}** · {r['mini_latency']:.2f}s")
                    st.info(r["mini_response"])
            except Exception as e:
                st.error(f"Lỗi khi gọi API: {e}")

# ---------------------------------------------------------------------------
# Tab 2 — count_tokens + estimate_cost (chạy được không cần key)
# ---------------------------------------------------------------------------
with tab_token:
    st.subheader("Đếm token & ước tính chi phí (Part 2)")
    st.caption("Chạy offline được — dùng tiktoken, không gọi API.")
    text_in = st.text_area("Văn bản input (prompt)", "Giải thích token là gì.", key="tok_in")
    text_out = st.text_area("Văn bản output (response)",
                            "Token là đơn vị văn bản mà model xử lý.", key="tok_out")
    model = st.selectbox("Model tính giá", list(lab.PRICING_PER_1K_TOKENS.keys()))
    if st.button("Tính token & chi phí"):
        cost = lab.estimate_cost(text_in, text_out, model=model)
        c1, c2, c3 = st.columns(3)
        c1.metric("Input tokens", cost["input_tokens"])
        c2.metric("Output tokens", cost["output_tokens"])
        c3.metric("Tổng chi phí", f"${cost['total_cost']:.8f}")
        st.json(cost)

# ---------------------------------------------------------------------------
# Tab 3 — Trợ lý chat nhiều lượt (streaming + history + thống kê)
# ---------------------------------------------------------------------------
with tab_chat:
    st.subheader("Trợ lý hội thoại (Part 3 + 4)")
    persona = st.text_input(
        "Persona (system prompt)",
        "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []   # để hiển thị trên UI
        st.session_state.history = []     # history gửi cho model (đã cắt 6)
        st.session_state.tokens = 0
        st.session_state.cost = 0.0
        st.session_state.turns = 0

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Lượt", st.session_state.turns)
    colB.metric("Token", st.session_state.tokens)
    colC.metric("Chi phí", f"${st.session_state.cost:.6f}")
    if colD.button("Xoá hội thoại"):
        for k in ("messages", "history", "tokens", "cost", "turns"):
            st.session_state.pop(k, None)
        st.rerun()

    # Hiển thị lịch sử chat
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_msg = st.chat_input("Nhập câu hỏi..." if has_key else "Cần API key trong .env")
    if user_msg and has_key:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)

        # Ghép messages có persona đứng đầu — đúng như run_assistant.
        messages = ([{"role": "system", "content": persona}]
                    + st.session_state.history
                    + [{"role": "user", "content": user_msg}])

        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        with st.chat_message("assistant"):
            placeholder = st.empty()
            reply = ""
            try:
                stream = lab.retry_with_backoff(
                    lambda: client.chat.completions.create(
                        model=lab.OPENAI_MODEL, messages=messages, stream=True,
                    )
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    reply += delta
                    placeholder.markdown(reply + "▌")   # hiệu ứng gõ dần
                placeholder.markdown(reply)
            except Exception as e:
                reply = f"[Lỗi API] {e}"
                placeholder.error(reply)

        # Cập nhật state y hệt run_assistant
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.history.append({"role": "user", "content": user_msg})
        st.session_state.history.append({"role": "assistant", "content": reply})
        st.session_state.history = st.session_state.history[-6:]
        st.session_state.turns += 1
        st.session_state.tokens += lab.count_tokens(user_msg) + lab.count_tokens(reply)
        st.session_state.cost += lab.estimate_cost(user_msg, reply)["total_cost"]
        st.rerun()
