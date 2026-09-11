"""
Giao diện review localhost cho Lab 01 — chạy: streamlit run app.py

Đây KHÔNG phải file nộp bài (bài chấm chỉ đọc template.py / solution/).
App gọi lại đúng các hàm bạn đã viết trong template.py để bạn thử trực tiếp
trên trình duyệt: so sánh model, đếm token / ước tính chi phí, và chat nhiều
lượt có streaming + history + thống kê.
"""

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import template as lab  # dùng lại các hàm đã viết trong template.py

load_dotenv()

HERE = Path(__file__).parent  # thư mục gốc lab, để chạy pytest/grade đúng chỗ


def run_command(args: list[str]):
    """Chạy một lệnh con (pytest/grade.py) và trả về (returncode, output gộp)."""
    proc = subprocess.run(
        args, cwd=str(HERE), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


import time  # noqa: E402  (dùng cho đo latency ở call_provider)
from openai import OpenAI  # noqa: E402


def call_provider(prompt: str, api_key: str, base_url: str, model: str,
                  max_tokens: int = 256):
    """Gọi một endpoint tương thích OpenAI bất kỳ (OpenAI/Groq/Gemini/...).

    Trả về (text, latency, error). Nếu lỗi thì text="" và error là thông báo.
    base_url rỗng -> dùng endpoint mặc định của OpenAI SDK.
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url or None)
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        latency = time.perf_counter() - start
        return (resp.choices[0].message.content or ""), latency, None
    except Exception as e:
        return "", 0.0, str(e)


import re  # noqa: E402


# Bảng giá tham chiếu để ước tính chi phí: model lạ -> rơi về gpt-4o (như lab).
def _price_key_for(model: str) -> str:
    # Tách theo "từ" (không phải chuỗi con) để tránh bẫy: "geMINI" chứa "mini"
    # nhưng KHÔNG phải model mini. Chỉ coi là loại nhỏ khi có token rõ ràng.
    tokens = re.split(r"[^a-z0-9]+", model.lower())
    small_markers = {"mini", "lite", "small", "8b", "7b", "9b"}
    if any(t in small_markers for t in tokens):
        return "gpt-4o-mini"
    return "gpt-4o"


def price_is_real(model: str) -> bool:
    """True nếu lab CÓ bảng giá thật cho model này; False nếu phải giả định."""
    return model in lab.PRICING_PER_1K_TOKENS


# ---------------------------------------------------------------------------
# Giới hạn request theo IP — tránh một người xài hết quota key free dùng chung.
# ---------------------------------------------------------------------------
import threading  # noqa: E402

RATE_LIMIT = 30       # số request gọi model tối đa mỗi IP...
RATE_WINDOW = 3600    # ...trong mỗi cửa sổ (giây) = 1 giờ.


@st.cache_resource
def _rate_store():
    """Kho đếm CHUNG cho mọi phiên/người dùng (singleton toàn server)."""
    return {"data": {}, "lock": threading.Lock()}


def get_client_ip() -> str:
    """Lấy IP người truy cập từ header (Streamlit Cloud đứng sau proxy)."""
    try:
        h = st.context.headers
        xff = h.get("X-Forwarded-For") or h.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()   # IP đầu tiên là client thật
        return h.get("X-Real-Ip") or h.get("x-real-ip") or "local"
    except Exception:
        return "local"


def check_rate(consume: bool = True):
    """Kiểm tra (và trừ) hạn mức của IP hiện tại.

    Trả về (allowed: bool, remaining: int, reset_in: int giây).
    consume=False chỉ xem còn bao nhiêu, không trừ.
    """
    ip = get_client_ip()
    store = _rate_store()
    now = time.time()
    with store["lock"]:
        window_start, count = store["data"].get(ip, (now, 0))
        # Hết cửa sổ -> reset bộ đếm.
        if now - window_start >= RATE_WINDOW:
            window_start, count = now, 0
        remaining = RATE_LIMIT - count
        reset_in = int(RATE_WINDOW - (now - window_start))
        if remaining <= 0:
            return False, 0, reset_in
        if consume:
            store["data"][ip] = (window_start, count + 1)
            remaining -= 1
        return True, remaining, reset_in

st.set_page_config(page_title="Lab 01 — LLM API Review", page_icon="🤖", layout="wide")

st.title("🤖 Lab 01 — Review LLM API")
st.caption("Giao diện thử nhanh các hàm trong template.py. Cần API key trong .env để gọi thật.")

# --- Trạng thái key ---
has_key = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("OPENAI_API_KEY") != "sk-your-key-here"
MENU_ITEMS = ["⚖️ So sánh model", "🔢 Token & chi phí", "💬 Trợ lý CLI", "🧪 Test & Chấm điểm"]

with st.sidebar:
    st.subheader("📋 Menu")
    menu = st.radio("Chọn chức năng", MENU_ITEMS, label_visibility="collapsed")
    st.divider()
    st.subheader("Cấu hình")
    st.write(f"Model lớn: `{lab.OPENAI_MODEL}`")
    st.write(f"Model nhỏ: `{lab.OPENAI_MINI_MODEL}`")
    st.write(f"Base URL: `{os.getenv('OPENAI_BASE_URL', 'mặc định OpenAI')}`")
    if has_key:
        st.success("Đã có API key — gọi thật được.")
    else:
        st.warning("Chưa có API key trong .env. Phần đếm token vẫn chạy; "
                   "phần gọi model (so sánh / chat) sẽ báo lỗi khi bấm.")

    # Hạn mức request theo IP (không trừ, chỉ xem).
    _ok, _remaining, _reset = check_rate(consume=False)
    st.divider()
    st.caption(f"Hạn mức gọi model: **{_remaining}/{RATE_LIMIT}** request "
               f"còn lại (reset sau ~{_reset // 60} phút). Giới hạn theo IP để "
               f"chia sẻ công bằng key free.")


# ---------------------------------------------------------------------------
# Tab 1 — compare_models
# ---------------------------------------------------------------------------
if menu == "⚖️ So sánh model":
    st.subheader("So sánh nhiều model / nhiều nhà cung cấp")
    st.caption("Nhập key cho từng provider (OpenAI, Groq, Gemini...) hoặc tự thêm. "
               "Bấm 'Chạy tất cả' để gọi mọi model đang bật rồi so sánh chi phí input/output/tổng. "
               "Key chỉ nằm trong phiên trình duyệt, không lưu ra file.")

    with st.expander("🔑 Lấy API key MIỄN PHÍ ở đâu?"):
        st.markdown(
            "- **Gemini (Google AI Studio)** — miễn phí, không cần thẻ: "
            "[aistudio.google.com/apikey](https://aistudio.google.com/apikey)\n"
            "- **Groq** — miễn phí, tốc độ rất nhanh: "
            "[console.groq.com/keys](https://console.groq.com/keys)\n"
            "- **NVIDIA NIM** — miễn phí (đúng gợi ý của lab, Phụ lục B): "
            "[build.nvidia.com](https://build.nvidia.com) · "
            "base URL `https://integrate.api.nvidia.com/v1`\n"
            "- **OpenAI** — trả phí (có credit dùng thử): "
            "[platform.openai.com/api-keys](https://platform.openai.com/api-keys)\n\n"
            "Lấy key xong dán vào ô **API key** của provider tương ứng rồi tick **Bật**. "
            "Đừng dán key vào chat hay commit lên GitHub."
        )

    # Danh sách provider mặc định (điền sẵn cấu hình, key để trống trừ Gemini
    # đã có trong .env). base_url là endpoint TƯƠNG THÍCH OpenAI của mỗi bên.
    if "providers" not in st.session_state:
        st.session_state.providers = [
            {"label": "Gemini (lớn)", "key": os.getenv("OPENAI_API_KEY", ""),
             "base": "https://generativelanguage.googleapis.com/v1beta/openai/",
             "model": "gemini-3.5-flash", "on": True},
            {"label": "Gemini (nhỏ)", "key": os.getenv("OPENAI_API_KEY", ""),
             "base": "https://generativelanguage.googleapis.com/v1beta/openai/",
             "model": "gemini-3.5-flash-lite", "on": True},
            {"label": "OpenAI", "key": "", "base": "https://api.openai.com/v1",
             "model": "gpt-4o-mini", "on": False},
            {"label": "Groq", "key": "", "base": "https://api.groq.com/openai/v1",
             "model": "llama-3.3-70b-versatile", "on": False},
        ]

    prompt = st.text_area("Prompt (gửi cho mọi model)",
                          "Việt Nam có bao nhiêu tỉnh thành?", key="cmp_prompt")

    st.markdown("##### Cấu hình provider")
    remove_idx = None
    for i, p in enumerate(st.session_state.providers):
        cols = st.columns([0.5, 2, 2.5, 2.5, 0.6])
        p["on"] = cols[0].checkbox("Bật", value=p["on"], key=f"on_{i}", label_visibility="collapsed")
        p["label"] = cols[1].text_input("Tên", value=p["label"], key=f"lb_{i}", label_visibility="collapsed")
        p["model"] = cols[2].text_input("Model", value=p["model"], key=f"md_{i}",
                                        placeholder="model", label_visibility="collapsed")
        p["key"] = cols[3].text_input("API key", value=p["key"], key=f"ky_{i}",
                                      type="password", placeholder="API key", label_visibility="collapsed")
        if cols[4].button("🗑", key=f"rm_{i}"):
            remove_idx = i
        # Base URL trên dòng phụ (dài).
        p["base"] = st.text_input(f"Base URL — {p['label']}", value=p["base"],
                                  key=f"bs_{i}", label_visibility="collapsed",
                                  placeholder="Base URL (tương thích OpenAI)")
    if remove_idx is not None:
        st.session_state.providers.pop(remove_idx)
        st.rerun()

    ca, cb = st.columns(2)
    if ca.button("➕ Thêm provider"):
        st.session_state.providers.append(
            {"label": "Provider mới", "key": "", "base": "", "model": "", "on": True})
        st.rerun()

    if cb.button("🚀 Chạy tất cả model đang bật", type="primary"):
        active = [p for p in st.session_state.providers if p["on"] and p["key"] and p["model"]]
        if not active:
            st.warning("Chưa có provider nào bật + có đủ key và model.")
        else:
            rows = []
            for p in active:
                # Trừ hạn mức theo IP trước mỗi lời gọi model thật.
                allowed, remaining, reset_in = check_rate()
                if not allowed:
                    st.error(f"⛔ Hết hạn mức {RATE_LIMIT} request/giờ cho IP này. "
                             f"Thử lại sau ~{reset_in // 60} phút.")
                    break
                with st.spinner(f"Đang gọi {p['label']} ({p['model']}) · còn {remaining} request..."):
                    text, latency, err = call_provider(prompt, p["key"], p["base"], p["model"])
                if err:
                    st.error(f"**{p['label']}** lỗi: {err[:200]}")
                    continue
                # Ước tính chi phí. Nếu model có giá thật trong lab thì dùng luôn;
                # nếu không thì tính theo giá tham chiếu và ĐÁNH DẤU là giả định.
                real = price_is_real(p["model"])
                price_model = p["model"] if real else _price_key_for(p["model"])
                cost = lab.estimate_cost(prompt, text, model=price_model)
                rows.append({**p, "text": text, "latency": latency,
                             "cost": cost, "real": real, "price_model": price_model})

            if rows:
                # Bảng so sánh: mỗi cột là một provider. Cột chi phí ghi rõ THẬT/giả định.
                st.markdown("#### 💰 So sánh chi phí (input / output / tổng)")
                any_assumed = any(not r["real"] for r in rows)
                if any_assumed:
                    st.warning("⚠️ Các model không có trong bảng giá của lab được tính "
                               "chi phí **GIẢ ĐỊNH** theo giá OpenAI tham chiếu (cột có dấu *), "
                               "**không phải giá thật** của nhà cung cấp. Token là số thật.")
                table = {"Hạng mục": ["Model", "Giá theo", "Latency (s)", "Input tokens",
                                      "Output tokens", "Chi phí input ($)", "Chi phí output ($)",
                                      "TỔNG ($)"]}
                for r in rows:
                    c = r["cost"]
                    star = "" if r["real"] else " *"
                    price_note = r["price_model"] if r["real"] else f"{r['price_model']} (giả định)"
                    table[r["label"] + star] = [
                        r["model"], price_note, f"{r['latency']:.2f}",
                        c["input_tokens"], c["output_tokens"],
                        f"{c['input_cost']:.6f}", f"{c['output_cost']:.6f}", f"{c['total_cost']:.6f}",
                    ]
                st.table(table)

                # Chỉ ra model rẻ nhất / đắt nhất.
                cheapest = min(rows, key=lambda r: r["cost"]["total_cost"])
                priciest = max(rows, key=lambda r: r["cost"]["total_cost"])
                m1, m2 = st.columns(2)
                m1.metric("Rẻ nhất", cheapest["label"], f"${cheapest['cost']['total_cost']:.6f}")
                m2.metric("Đắt nhất", priciest["label"], f"${priciest['cost']['total_cost']:.6f}")

                # Câu trả lời đầy đủ của từng model.
                st.markdown("#### 📝 Câu trả lời từng model")
                for r in rows:
                    with st.expander(f"{r['label']} · {r['model']} · {r['latency']:.2f}s"):
                        st.write(r["text"])

# ---------------------------------------------------------------------------
# Tab 2 — count_tokens + estimate_cost (chạy được không cần key)
# ---------------------------------------------------------------------------
elif menu == "🔢 Token & chi phí":
    st.subheader("Đếm token & ước tính chi phí (Part 2)")
    st.caption("Chạy offline được — dùng tiktoken, không gọi API.")
    text_in = st.text_area("Văn bản input (prompt)", "Giải thích token là gì.", key="tok_in")
    text_out = st.text_area("Văn bản output (response)",
                            "Token là đơn vị văn bản mà model xử lý.", key="tok_out")
    # Cho chọn cả model thật đang cấu hình (Gemini...) lẫn các mức giá của lab.
    model_options = list(dict.fromkeys(
        [lab.OPENAI_MODEL, lab.OPENAI_MINI_MODEL, *lab.PRICING_PER_1K_TOKENS.keys()]
    ))
    model = st.selectbox("Model (để đếm token & chọn mức giá)", model_options)
    real = price_is_real(model)
    price_ref = model if real else _price_key_for(model)
    if real:
        st.caption(f"✅ Chi phí THẬT: lab có bảng giá cho `{model}` "
                   f"(`PRICING_PER_1K_TOKENS`).")
    else:
        st.caption(f"⚠️ Chi phí **GIẢ ĐỊNH**: lab KHÔNG có giá cho `{model}`. "
                   f"Con số dưới đây tính theo mức tham chiếu `{price_ref}` của OpenAI, "
                   f"**không phải giá thật** của nhà cung cấp. Muốn số đúng, tra bảng giá "
                   f"chính thức của họ.")
    if st.button("Tính token & chi phí"):
        cost = lab.estimate_cost(text_in, text_out, model=model)
        tag = "" if real else " (giả định)"
        c1, c2, c3 = st.columns(3)
        c1.metric("Input tokens", cost["input_tokens"])
        c2.metric("Output tokens", cost["output_tokens"])
        c3.metric(f"Tổng chi phí{tag}", f"${cost['total_cost']:.8f}")
        st.json(cost)
        if not real:
            st.info("Số token là thật (đếm bằng tiktoken/fallback); chỉ phần chi phí "
                    "là giả định theo giá tham chiếu.")

# ---------------------------------------------------------------------------
# Tab 3 — Trợ lý chat nhiều lượt (streaming + history + thống kê)
# ---------------------------------------------------------------------------
elif menu == "💬 Trợ lý CLI":
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
        # Trừ hạn mức theo IP trước khi gọi model thật.
        allowed, remaining, reset_in = check_rate()
        if not allowed:
            st.error(f"⛔ Hết hạn mức {RATE_LIMIT} request/giờ cho IP này. "
                     f"Thử lại sau ~{reset_in // 60} phút.")
            st.stop()
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

# ---------------------------------------------------------------------------
# Tab 4 — Chạy test & chấm điểm ngay trong giao diện (dùng mock, không cần key)
# ---------------------------------------------------------------------------
elif menu == "🧪 Test & Chấm điểm":
    st.subheader("Chạy test & chấm điểm")
    st.caption("Chạy pytest/grade.py bằng mock — không tốn API, không cần key. "
               "Bấm nút để xem chương trình chạy và kết quả pass/fail ngay tại đây.")

    # Các nút chạy test theo từng phần hoặc toàn bộ.
    parts = {
        "Toàn bộ (35 test)": ["tests/"],
        "Part 1 — API": ["tests/test_part1.py"],
        "Part 2 — Token": ["tests/test_part2.py"],
        "Part 3 — Streaming/Retry": ["tests/test_part3.py"],
        "Part 4 — Trợ lý": ["tests/test_part4.py"],
    }
    choice = st.radio("Chọn nhóm test", list(parts.keys()), horizontal=True)

    c1, c2 = st.columns(2)

    if c1.button("▶️ Chạy pytest", use_container_width=True):
        with st.spinner("Đang chạy pytest..."):
            code, out = run_command([sys.executable, "-m", "pytest", *parts[choice], "-v"])
        # Dòng tổng kết cuối của pytest (ví dụ "35 passed in 2.1s")
        summary = next((l for l in reversed(out.splitlines()) if "passed" in l or "failed" in l), "")
        if code == 0:
            st.success(f"✅ PASS — {summary.strip()}")
        else:
            st.error(f"❌ Có test fail — {summary.strip()}")
        st.code(out, language="text")

    if c2.button("🏆 Chấm điểm (grade.py)", use_container_width=True):
        with st.spinner("Đang chấm..."):
            code, out = run_command([sys.executable, "grade.py"])
        # Lấy dòng TỔNG .../100 để hiện nổi bật
        total = next((l for l in out.splitlines() if "TỔNG" in l), "")
        if total:
            st.success("✅ " + total.strip())
        st.code(out, language="text")
