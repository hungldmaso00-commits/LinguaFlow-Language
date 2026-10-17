"""LinguaFlow - Ứng dụng học Anh, Trung, Việt bằng Groq AI."""
from __future__ import annotations

import asyncio
import difflib
import html
import json
import re
import tempfile
from pathlib import Path

import edge_tts
import pandas as pd
import streamlit as st
from groq import Groq

import database as db

st.set_page_config(page_title="LinguaFlow", page_icon="🌊", layout="wide")


def secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


db.init_db(secret("DATABASE_URL"))
st.markdown("""<style>
.stApp{background:#0b1020;color:#e5e7eb}.hero{padding:1.2rem 1.5rem;border-radius:18px;background:linear-gradient(120deg,#312e81,#0e7490);margin-bottom:1rem}.hero h1{margin:0;color:white}.hero p{margin:.35rem 0 0;color:#cbd5e1}.card{background:#151d38;padding:1rem;border-radius:14px;border:1px solid #26345d}div[data-testid='stMetric']{background:#18213f;border-radius:12px;padding:.6rem}.stButton>button{border-radius:9px;border:0;background:#14b8a6;color:#03151a;font-weight:700}.mistake{background:#7f1d1d;color:#fecaca;padding:2px 4px;border-radius:4px;font-weight:700}.missing{background:#78350f;color:#fde68a;padding:2px 4px;border-radius:4px;font-weight:700}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_groq_client(api_key: str):
    return Groq(api_key=api_key, timeout=30, max_retries=2) if api_key else None


@st.cache_data(ttl=3600, show_spinner=False)
def call_groq_cached(prompt: str, json_mode: bool = False) -> str:
    """Lớp cache duy nhất cho mọi request Groq, hiệu lực trong 1 giờ."""
    client = get_groq_client(secret("GROQ_API_KEY"))
    if client is None:
        raise RuntimeError("Chưa cấu hình GROQ_API_KEY")
    options = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "Bạn là trợ lý học ngoại ngữ chính xác, thân thiện."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.45,
    }
    if json_mode:
        options["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**options)
    return response.choices[0].message.content or ""


def ask_ai(prompt: str, json_mode: bool = False):
    try:
        result = call_groq_cached(prompt, json_mode)
        if not json_mode:
            return result
        match = re.search(r"\{.*\}", result, re.S)
        return json.loads(match.group(0) if match else result)
    except json.JSONDecodeError:
        st.error("AI trả về dữ liệu chưa đúng định dạng. Hãy thử lại.")
    except Exception as exc:
        st.error(f"Không thể gọi Groq. Kiểm tra API key, hạn mức hoặc kết nối. ({type(exc).__name__})")
    return None


async def create_tts_mp3(text: str, language: str) -> bytes:
    """Tạo file MP3 tạm từ Edge TTS, sau đó trả dữ liệu cho st.audio."""
    voice = "en-US-AriaNeural" if language == "English" else "zh-CN-XiaoxiaoNeural"
    with tempfile.TemporaryDirectory() as folder:
        filename = Path(folder) / "linguaflow.mp3"
        await edge_tts.Communicate(text=text, voice=voice).save(str(filename))
        return filename.read_bytes()


def play_tts(text: str, language: str, key: str, label: str):
    if st.button(label, key=key, disabled=not bool(text)):
        try:
            with st.spinner("Đang tạo giọng đọc..."):
                st.audio(asyncio.run(create_tts_mp3(text, language)), format="audio/mp3")
        except Exception as exc:
            st.warning(f"Không tạo được âm thanh. ({type(exc).__name__})")


def clean_tts_text(text: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", text or "").strip()


def bilingual_terms(item: dict) -> tuple[str, str]:
    term, other = item.get("term", ""), item.get("other_language", "")
    if item.get("source_language", "").lower() == "chinese":
        return clean_tts_text(other), clean_tts_text(term)
    return clean_tts_text(term), clean_tts_text(other)


def difficulty_options(language: str) -> list[str]:
    return ["A1-A2", "B1-B2", "C1"] if language == "English" else [f"HSK{i}" for i in range(1, 10)]


def tokenise(text: str, language: str) -> list[str]:
    return list(re.sub(r"\s+", "", text)) if language == "Chinese" else re.findall(r"\S+", text)


def highlight_dictation(expected: str, typed: str, language: str) -> str:
    """Tô đỏ phần người dùng gõ sai, tô vàng các token bị thiếu."""
    expected_tokens, typed_tokens = tokenise(expected, language), tokenise(typed, language)
    output = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, expected_tokens, typed_tokens).get_opcodes():
        if tag == "equal":
            output += [html.escape(token) for token in typed_tokens[j1:j2]]
        elif tag in {"replace", "delete"}:
            output += [f"<span class='mistake'>{html.escape(token)}</span>" for token in typed_tokens[j1:j2]]
        if tag in {"replace", "insert"}:
            output += [f"<span class='missing'>thiếu: {html.escape(token)}</span>" for token in expected_tokens[i1:i2]]
    return ("" if language == "Chinese" else " ").join(output) or "<span class='missing'>Chưa có nội dung hợp lệ</span>"


def generate_dictation(language: str, difficulty: str, source: str):
    if source == "database":
        vocab = db.get_random_vocabulary(4)
        if not vocab:
            st.warning("Kho từ vựng đang trống; hãy lưu từ trước.")
            return
        word_list = ", ".join(f"{item['term']} ({item['source_language']})" for item in vocab)
        prompt = f"Tạo đúng một câu {language}, độ khó {difficulty}, dùng tự nhiên một hoặc nhiều từ: {word_list}. Chỉ trả về câu, không giải thích."
    else:
        prompt = f"Tạo đúng một câu {language} tự nhiên cho bài nghe chép, độ khó {difficulty}, dài 8-18 từ. Chỉ trả về câu, không giải thích."
    sentence = ask_ai(prompt)
    if sentence:
        st.session_state["dictation_sentence"] = sentence.strip().strip('"')
        st.session_state["dictation_language"] = language
        st.session_state.pop("dictation_result", None)


def dictionary_prompt(term: str, language: str) -> str:
    return f'''Bạn là giáo viên Anh-Trung-Việt. Phân tích "{term}" (ngôn ngữ: {language}).
Trả về DUY NHẤT JSON hợp lệ gồm term, source_language (English/Chinese), vietnamese_meaning,
other_language, part_of_speech, pronunciation, examples (mảng 3 đối tượng sentence/vietnamese),
synonyms (mảng), antonyms (mảng), tags. Nghĩa và ví dụ phải tự nhiên, chính xác.'''


def render_dictionary(result: dict):
    title, english_column, chinese_column = st.columns([3, 1, 1])
    title.subheader(f"{result.get('term', '')} · {result.get('part_of_speech', '')}")
    english, chinese = bilingual_terms(result)
    with english_column:
        play_tts(english, "English", f"dict_en_{result.get('term', '')}", "🔊 Nghe tiếng Anh")
    with chinese_column:
        play_tts(chinese, "Chinese", f"dict_zh_{result.get('term', '')}", "🔊 Nghe tiếng Trung")
    st.caption(f"🔊 {result.get('pronunciation', '—')} | Ngôn ngữ: {result.get('source_language', '—')}")
    first, second = st.columns(2)
    first.info(f"**Tiếng Việt:** {result.get('vietnamese_meaning', '—')}")
    second.success(f"**Tương đương:** {result.get('other_language', '—')}")
    st.markdown("**Ví dụ thực tế**")
    for index, example in enumerate(result.get("examples", []), 1):
        st.markdown(f"{index}. **{example.get('sentence', '')}**  \n{example.get('vietnamese', '')}")
    st.caption(f"Đồng nghĩa: {', '.join(result.get('synonyms', [])) or '—'} · Trái nghĩa: {', '.join(result.get('antonyms', [])) or '—'}")


with st.sidebar:
    st.title("🌊 LinguaFlow")
    st.caption("AI Language Ecosystem")
    stats = db.get_statistics()
    col_one, col_two = st.columns(2)
    col_one.metric("Từ đã học", stats["words"])
    col_two.metric("Streak", f"{stats['streak']} ngày")
    st.metric("Đến hạn ôn", stats["due"])
    st.divider()
    st.text_input("Tên hiển thị", value="Language Learner", key="profile_name")
    st.caption("Lưu từ mới và ôn flashcard mỗi ngày để duy trì streak.")

st.markdown("<div class='hero'><h1>LinguaFlow</h1><p>Học Anh · Trung · Việt theo nhịp của bạn.</p></div>", unsafe_allow_html=True)
tab_dictionary, tab_grammar, tab_bank, tab_quiz, tab_practice = st.tabs([
    "📖 Từ điển 3 ngữ", "✍️ Ngữ pháp & Ngữ nghĩa", "🗂️ Kho từ & Flashcard", "🎯 Bài tập", "🎭 Nhập vai & Nghe chép"
])

with tab_dictionary:
    term = st.text_input("Từ hoặc cụm từ", placeholder="computer / 学习 / artificial intelligence")
    input_language = st.selectbox("Ngôn ngữ đầu vào", ["Tự nhận diện", "English", "Chinese"])
    if st.button("Phân tích bằng AI", key="lookup") and term.strip():
        with st.spinner("LinguaFlow đang phân tích..."):
            result = ask_ai(dictionary_prompt(term, input_language), True)
        if result:
            st.session_state["dictionary_result"] = result
    result = st.session_state.get("dictionary_result")
    if result:
        render_dictionary(result)
        if st.button("➕ Lưu vào Kho từ vựng"):
            if db.save_vocabulary(result):
                db.log_history("lookup", result.get("term", ""))
                st.success("Đã lưu từ mới vào kho!")
            else:
                st.info("Từ này đã có trong kho hoặc thiếu dữ liệu bắt buộc.")

with tab_grammar:
    mode = st.radio("Chế độ", ["Kiểm tra ngữ pháp", "So sánh hai câu"], horizontal=True)
    if mode == "Kiểm tra ngữ pháp":
        text = st.text_area("Đoạn tiếng Anh hoặc Trung", height=160, placeholder="I has went to school yesterday.")
        if st.button("Kiểm tra & giải thích", key="grammar") and text.strip():
            answer = ask_ai(f"Phân tích ngữ pháp: {text}\nTrả lời tiếng Việt Markdown: đánh giá, bản sửa, giải thích từng lỗi và in đậm từ sai.")
            if answer:
                st.markdown(answer)
                db.log_history("grammar", text[:80])
    else:
        english_sentence = st.text_area("Câu tiếng Anh", key="en_compare")
        chinese_sentence = st.text_area("Câu tiếng Trung", key="zh_compare")
        if st.button("So sánh ngữ nghĩa", key="compare") and english_sentence.strip() and chinese_sentence.strip():
            answer = ask_ai(f"So sánh ngữ nghĩa English: {english_sentence}\nChinese: {chinese_sentence}\nTrả lời tiếng Việt Markdown: % tương đồng, giống và khác về sắc thái/ngữ pháp.")
            if answer:
                st.markdown(answer)

with tab_bank:
    left, right = st.columns([1, 2])
    with left:
        filter_language = st.selectbox("Lọc ngôn ngữ", ["Tất cả", "English", "Chinese"])
        tag = st.text_input("Lọc tag")
    with right:
        st.download_button("⬇️ Xuất toàn bộ CSV", db.export_csv(), "linguaflow_vocabulary.csv", "text/csv")
    vocab = db.get_vocabulary(filter_language, tag)
    if vocab:
        columns = ["id", "term", "pronunciation", "source_language", "vietnamese_meaning", "other_language", "part_of_speech", "tags"]
        original = pd.DataFrame(vocab)[columns]
        edited = st.data_editor(original, key="vocabulary_editor", num_rows="dynamic", hide_index=True, use_container_width=True, disabled=["id"], column_config={"id": st.column_config.NumberColumn("ID", width="small")})
        if st.button("💾 Lưu thay đổi Kho từ vựng"):
            old_by_id = {int(row["id"]): row for _, row in original.iterrows()}
            kept_ids = {int(row["id"]) for _, row in edited.dropna(subset=["id"]).iterrows()}
            for vocab_id in set(old_by_id) - kept_ids:
                db.delete_vocabulary(vocab_id)
            for _, row in edited.iterrows():
                item = row.fillna("").to_dict()
                vocab_id = item.pop("id", None)
                if pd.isna(vocab_id) or not vocab_id:
                    db.save_vocabulary(item)
                elif item != {field: old_by_id[int(vocab_id)][field] for field in item}:
                    db.update_vocabulary(int(vocab_id), item)
            st.success("Đã lưu các thay đổi.")
            st.rerun()
    else:
        st.info("Kho từ vựng đang trống. Hãy lưu một kết quả từ tab Từ điển.")

    st.divider()
    st.subheader("🧠 Flashcard SM-2")
    due_cards = db.get_due_cards()
    if due_cards:
        card_index = st.session_state.get("card_index", 0) % len(due_cards)
        card = due_cards[card_index]
        text_col, english_col, chinese_col = st.columns([3, 1, 1])
        with text_col:
            st.markdown(f"<div class='card'><h2>{card['term']}</h2><p>{card['source_language']} · {card['pronunciation'] or 'Không có phiên âm'}</p></div>", unsafe_allow_html=True)
        english, chinese = bilingual_terms(card)
        with english_col:
            play_tts(english, "English", f"flash_en_{card['id']}", "🔊 Nghe tiếng Anh")
        with chinese_col:
            play_tts(chinese, "Chinese", f"flash_zh_{card['id']}", "🔊 Nghe tiếng Trung")
        if st.toggle("Lật thẻ — xem đáp án", key=f"flip_{card['id']}"):
            st.success(f"**{card['vietnamese_meaning']}** · {card['other_language'] or ''}")
            st.caption(card["examples"][:300])
            for col, (quality, label) in zip(st.columns(4), [(0, "Quên"), (2, "Khó"), (4, "Tốt"), (5, "Dễ")]):
                if col.button(label, key=f"review_{quality}_{card['id']}"):
                    db.review_card(card["id"], quality)
                    st.session_state["card_index"] = card_index + 1
                    st.rerun()
    else:
        st.success("Không có từ đến hạn ôn hôm nay. Tuyệt vời!")

with tab_quiz:
    st.write("AI sinh bài tập từ Kho từ vựng của bạn.")
    quiz_type = st.selectbox("Dạng bài", ["Trắc nghiệm", "Điền từ", "Sắp xếp câu"])
    if st.button("Tạo bài tập", key="make_quiz"):
        words = db.get_vocabulary()[:30]
        if not words:
            st.warning("Cần ít nhất một từ trong Kho từ vựng.")
        else:
            compact = [{"term": word["term"], "meaning": word["vietnamese_meaning"], "lang": word["source_language"]} for word in words]
            quiz = ask_ai(f"Tạo 5 bài {quiz_type} từ: {json.dumps(compact, ensure_ascii=False)}. Chỉ JSON: {{'questions':[{{'question':'','options':['','','',''],'answer':'','explanation':''}}]}}. Điền từ/sắp xếp có options=[].", True)
            if quiz:
                st.session_state["quiz"] = quiz
                st.session_state["quiz_submitted"] = False
    quiz = st.session_state.get("quiz")
    if quiz:
        answers = []
        for index, question in enumerate(quiz.get("questions", [])):
            st.markdown(f"**Câu {index + 1}.** {question.get('question', '')}")
            options = question.get("options", [])
            answers.append(st.radio("Đáp án", options, key=f"quiz_answer_{index}") if options else st.text_input("Câu trả lời", key=f"quiz_answer_{index}"))
        if st.button("Nộp bài", key="submit_quiz"):
            correct = sum(str(answer).strip().lower() == str(question.get("answer", "")).strip().lower() for answer, question in zip(answers, quiz.get("questions", [])))
            total = len(quiz.get("questions", []))
            db.save_quiz_result(quiz_type, correct, total)
            st.session_state["quiz_submitted"] = True
            st.session_state["quiz_score"] = correct, total
        if st.session_state.get("quiz_submitted"):
            correct, total = st.session_state["quiz_score"]
            st.success(f"Kết quả: {correct}/{total} câu đúng")
            for index, question in enumerate(quiz.get("questions", [])):
                st.caption(f"Câu {index + 1}: **{question.get('answer', '')}** — {question.get('explanation', '')}")

with tab_practice:
    activity = st.radio("Hoạt động", ["Roleplay", "Nghe chép"], horizontal=True)
    if activity == "Roleplay":
        situation = st.selectbox("Tình huống", ["Mua sắm", "Đặt phòng khách sạn", "Phỏng vấn xin việc"])
        language = st.selectbox("Ngôn ngữ luyện", ["English", "Chinese"])
        difficulty = st.selectbox("Mức độ khó", difficulty_options(language), key="roleplay_difficulty")
        if "chat" not in st.session_state:
            st.session_state.chat = []
        for message in st.session_state.chat:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        user_message = st.chat_input(f"Nhắn bằng {language}...")
        if user_message:
            st.session_state.chat.append({"role": "user", "content": user_message})
            reply = ask_ai(f"Roleplay {situation} bằng {language}, độ khó {difficulty}. Người học: {user_message}. Đáp tự nhiên bằng {language}, sau đó dòng 'Gợi ý:' bằng tiếng Việt.")
            if reply:
                st.session_state.chat.append({"role": "assistant", "content": reply})
                db.log_history("roleplay", situation)
                st.rerun()
        first, second = st.columns(2)
        if first.button("💡 Gợi ý câu trả lời"):
            hint = ask_ai(f"Cho 3 câu trả lời mẫu {language}, độ khó {difficulty}, cho tình huống {situation}; kèm dịch Việt.")
            if hint:
                st.info(hint)
        if second.button("✅ Sửa câu vừa chat") and st.session_state.chat:
            last = next((message["content"] for message in reversed(st.session_state.chat) if message["role"] == "user"), "")
            correction = ask_ai(f"Sửa ngữ pháp {language}: {last}. Giải thích rất ngắn bằng tiếng Việt.")
            if correction:
                st.success(correction)
    else:
        language = st.selectbox("Ngôn ngữ nghe", ["English", "Chinese"], key="dictation_language_select")
        difficulty = st.selectbox("Mức độ khó", difficulty_options(language), key="dictation_difficulty")
        from_database, random_sentence = st.columns(2)
        if from_database.button("📚 Tạo câu từ Database"):
            generate_dictation(language, difficulty, "database")
        if random_sentence.button("🎲 Tạo câu ngẫu nhiên"):
            generate_dictation(language, difficulty, "random")
        sentence = st.session_state.get("dictation_sentence")
        if sentence:
            sentence_language = st.session_state.get("dictation_language", language)
            st.caption(f"Câu {sentence_language} · {difficulty}")
            try:
                st.audio(asyncio.run(create_tts_mp3(sentence, sentence_language)), format="audio/mp3")
            except Exception as exc:
                st.warning(f"Không tạo được âm thanh. ({type(exc).__name__})")
            typed = st.text_input("Gõ lại câu bạn nghe được", key="dictation_typed")
            if st.button("Chấm điểm nghe chép") and typed.strip():
                ratio = difflib.SequenceMatcher(None, tokenise(sentence.lower(), sentence_language), tokenise(typed.lower(), sentence_language)).ratio() * 100
                st.success(f"Độ chính xác: {ratio:.0f}%")
                st.markdown("**Bài bạn gõ** (đỏ = sai, vàng = phần bị thiếu):")
                st.markdown(highlight_dictation(sentence, typed, sentence_language), unsafe_allow_html=True)
                st.caption(f"Câu chuẩn: {sentence}")
                db.log_history("dictation", sentence_language, ratio)
