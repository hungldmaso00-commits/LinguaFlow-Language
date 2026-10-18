"""LinguaFlow - Ứng dụng học Anh, Trung, Việt bằng Groq AI."""
from __future__ import annotations

import asyncio
import difflib
import hashlib
import html
import json
import secrets
import re
import string
import tempfile
import uuid
from pathlib import Path

import edge_tts
import pandas as pd
import requests
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


def _password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return f"{salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, expected = stored.split("$", 1)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1).hex()
        return secrets.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _supabase_auth(action: str, email: str, password: str):
    """Call Supabase Auth REST API using the project's public anon key."""
    base_url = secret("SUPABASE_URL").rstrip("/")
    anon_key = secret("SUPABASE_ANON_KEY")
    if not base_url or not anon_key:
        raise RuntimeError("Supabase Auth cần SUPABASE_URL và SUPABASE_ANON_KEY trong secrets.")
    endpoint = f"{base_url}/auth/v1/signup" if action == "signup" else f"{base_url}/auth/v1/token?grant_type=password"
    response = requests.post(endpoint, headers={"apikey": anon_key, "Content-Type": "application/json"},
                             json={"email": email, "password": password}, timeout=20)
    if not response.ok:
        try:
            detail = response.json().get("msg") or response.json().get("message") or ""
        except ValueError:
            detail = ""
        raise ValueError(detail or f"Supabase Auth trả về HTTP {response.status_code}.")
    return response.json()


def render_auth_gate() -> str | None:
    """Show login/signup and return authenticated user ID, or stop rendering."""
    current = st.session_state.get("lingua_user")
    if current:
        with st.sidebar:
            st.caption(f"Đăng nhập: {current['email']}")
            if st.button("Đăng xuất", key="logout"):
                st.session_state.pop("lingua_user", None)
                st.session_state.pop("supabase_access_token", None)
                for key in ("chat", "dictionary_result", "quiz", "quiz_submitted", "quiz_result", "wrong_questions", "wrong_questions_scope", "roadmap_lesson", "roadmap_quiz_result", "roadmap_review_test", "roadmap_review_result", "quiz_review_test", "quiz_review_result", "dictation_sentence"):
                    st.session_state.pop(key, None)
                st.rerun()
        return current["user_id"]

    st.markdown("<div class='hero'><h1>🌊 LinguaFlow</h1><p>Đăng nhập để đồng bộ và lưu riêng tiến độ học của bạn.</p></div>", unsafe_allow_html=True)
    cloud_auth_ready = bool(secret("SUPABASE_URL") and secret("SUPABASE_ANON_KEY"))
    local_auth_label = "Tài khoản nội bộ (PostgreSQL)" if secret("DATABASE_URL") else "Tài khoản nội bộ (SQLite)"
    auth_options = ["Supabase Cloud", local_auth_label] if cloud_auth_ready else [local_auth_label]
    auth_choice = st.radio("Loại tài khoản", auth_options, horizontal=True,
                           help=None if cloud_auth_ready else "Để đăng nhập Supabase, cấu hình SUPABASE_URL và SUPABASE_ANON_KEY trong Streamlit secrets.")
    login_tab, signup_tab = st.tabs(["Đăng nhập", "Đăng ký"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Mật khẩu", type="password", key="login_password")
            submitted = st.form_submit_button("Đăng nhập", use_container_width=True)
        if submitted:
            try:
                if auth_choice == "Supabase Cloud":
                    result = _supabase_auth("login", email.strip(), password)
                    user = result.get("user") or {}
                    if not result.get("access_token") or not user.get("id"):
                        raise ValueError("Tài khoản cần xác nhận email trước khi đăng nhập.")
                    db.ensure_external_user(str(user["id"]), user.get("email", email))
                    db.claim_legacy_records(str(user["id"]))
                    st.session_state["supabase_access_token"] = result["access_token"]
                    st.session_state["lingua_user"] = {"user_id": str(user["id"]), "email": user.get("email", email)}
                else:
                    user = db.get_local_user(email.strip())
                    if not user or not user.get("password_hash") or not _verify_password(password, user["password_hash"]):
                        raise ValueError("Email hoặc mật khẩu không đúng.")
                    st.session_state["lingua_user"] = {"user_id": user["user_id"], "email": user["email"]}
                    db.claim_legacy_records(user["user_id"])
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with signup_tab:
        with st.form("signup_form"):
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Mật khẩu (ít nhất 8 ký tự)", type="password", key="signup_password")
            confirm_password = st.text_input("Nhập lại mật khẩu", type="password", key="signup_confirm")
            create_account = st.form_submit_button("Tạo tài khoản", use_container_width=True)
        if create_account:
            try:
                if not new_email.strip() or len(new_password) < 8:
                    raise ValueError("Nhập email hợp lệ và mật khẩu tối thiểu 8 ký tự.")
                if new_password != confirm_password:
                    raise ValueError("Hai mật khẩu chưa khớp.")
                if auth_choice == "Supabase Cloud":
                    result = _supabase_auth("signup", new_email.strip(), new_password)
                    user = result.get("user") or {}
                    if not user.get("id"):
                        raise ValueError("Supabase chưa tạo được tài khoản.")
                    if not result.get("access_token"):
                        st.success("Đã đăng ký. Hãy xác nhận email từ Supabase rồi đăng nhập.")
                    else:
                        db.ensure_external_user(str(user["id"]), user.get("email", new_email))
                        db.claim_legacy_records(str(user["id"]))
                        st.session_state["supabase_access_token"] = result["access_token"]
                        st.session_state["lingua_user"] = {"user_id": str(user["id"]), "email": user.get("email", new_email)}
                        st.rerun()
                else:
                    normalized_email = new_email.strip().lower()
                    if not db.create_local_user(normalized_email, _password_hash(new_password), str(uuid.uuid4())):
                        raise ValueError("Email này đã được đăng ký.")
                    st.success("Tài khoản Local đã tạo. Hãy đăng nhập bằng tab bên cạnh.")
            except Exception as exc:
                st.error(str(exc))
    st.info("Đăng nhập Supabase dùng SUPABASE_URL và SUPABASE_ANON_KEY. Tài khoản nội bộ được lưu trong database hiện cấu hình; nếu không có DATABASE_URL, dữ liệu nằm trong SQLite trên máy này.")
    st.stop()


user_id = render_auth_gate()
st.session_state.setdefault("wrong_questions", [])
st.markdown("""<style>
.stApp{background:#0b1020;color:#e5e7eb}.hero{padding:1.2rem 1.5rem;border-radius:18px;background:linear-gradient(120deg,#312e81,#0e7490);margin-bottom:1rem}.hero h1{margin:0;color:white}.hero p{margin:.35rem 0 0;color:#cbd5e1}.card{background:#151d38;padding:1rem;border-radius:14px;border:1px solid #26345d}div[data-testid='stMetric']{background:#18213f;border-radius:12px;padding:.6rem}.stButton>button{border-radius:9px;border:0;background:#14b8a6;color:#03151a;font-weight:700}.mistake{background:#7f1d1d;color:#fecaca;padding:2px 4px;border-radius:4px;font-weight:700}.missing{background:#78350f;color:#fde68a;padding:2px 4px;border-radius:4px;font-weight:700}
</style>""", unsafe_allow_html=True)


def get_effective_groq_api_key() -> str:
    """Prefer the current user's key; use the deployment-wide secret as fallback."""
    current_user = st.session_state.get("lingua_user") or {}
    personal_key = db.get_user_groq_key(current_user.get("user_id", user_id))
    return personal_key or secret("GROQ_API_KEY")


@st.cache_resource
def _cached_groq_client(api_key: str):
    return Groq(api_key=api_key, timeout=30, max_retries=2) if api_key else None


def get_groq_client(api_key: str | None = None):
    """Return a Groq client using a supplied key or the signed-in user's key."""
    return _cached_groq_client(api_key if api_key is not None else get_effective_groq_api_key())


@st.cache_data(ttl=3600, show_spinner=False)
def call_groq_cached(prompt: str, json_mode: bool, api_key: str) -> str:
    """Lớp cache duy nhất cho mọi request Groq, hiệu lực trong 1 giờ."""
    client = get_groq_client(api_key)
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


@st.cache_data(ttl=3600, show_spinner=False)
def transcribe_audio_cached(audio_bytes: bytes, mime_type: str, api_key: str) -> str:
    """Transcribe audio with Groq Whisper; identical recordings are cached for one hour."""
    client = get_groq_client(api_key)
    if client is None:
        raise RuntimeError("Chưa cấu hình GROQ_API_KEY")
    extension = "webm" if "webm" in mime_type else "wav" if "wav" in mime_type else "mp4"
    response = client.audio.transcriptions.create(
        file=(f"roleplay.{extension}", audio_bytes, mime_type or "audio/webm"),
        model="whisper-large-v3",
        response_format="json",
    )
    return response.text or ""


def ask_ai(prompt: str, json_mode: bool = False):
    try:
        api_key = get_effective_groq_api_key()
        if not api_key:
            raise RuntimeError("Chưa có Groq API Key cá nhân hoặc key mặc định trong secrets.")
        result = call_groq_cached(prompt, json_mode, api_key)
        if not json_mode:
            return result
        match = re.search(r"\{.*\}", result, re.S)
        return json.loads(match.group(0) if match else result)
    except json.JSONDecodeError:
        st.error("AI trả về dữ liệu chưa đúng định dạng. Hãy thử lại.")
    except Exception as exc:
        st.error(f"Không thể gọi Groq. Kiểm tra API key, hạn mức hoặc kết nối. ({type(exc).__name__})")
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def translate_roadmap_content_cached(term: str, meaning: str, example: str, api_key: str) -> dict:
    """Cache Vietnamese translations of roadmap vocabulary for one hour."""
    prompt = (
        "Dịch nghĩa và ví dụ từ vựng sang tiếng Việt tự nhiên, giữ nguyên term. "
        "Trả về DUY NHẤT JSON hợp lệ gồm meaning_vi và example_vi. "
        f"term={term!r}; meaning={meaning!r}; example={example!r}"
    )
    response = call_groq_cached(prompt, True, api_key)
    match = re.search(r"\{.*\}", response, re.S)
    return json.loads(match.group(0) if match else response)


@st.cache_data(ttl=3600, show_spinner=False)
def translate_roadmap_prompt_cached(prompt_text: str, api_key: str) -> str:
    """Cache Vietnamese translations of roadmap prompts for one hour."""
    return call_groq_cached(
        f"Dịch đề bài sau sang tiếng Việt rõ ràng, giữ nguyên ý và độ khó. Chỉ trả về bản dịch:\n{prompt_text}",
        False,
        api_key,
    ).strip()


async def _generate_tts_mp3(text: str, language: str) -> bytes:
    """Tạo file MP3 tạm từ Edge TTS."""
    voice = "en-US-AriaNeural" if language == "English" else "zh-CN-XiaoxiaoNeural"
    with tempfile.TemporaryDirectory() as folder:
        filename = Path(folder) / "linguaflow.mp3"
        await edge_tts.Communicate(text=text, voice=voice).save(str(filename))
        return filename.read_bytes()


@st.cache_data(ttl=3600, show_spinner=False)
def create_tts_mp3(text: str, language: str) -> bytes:
    """Cache generated speech for an hour so Streamlit reruns reuse the same audio."""
    return asyncio.run(_generate_tts_mp3(text, language))


def play_tts(text: str, language: str, key: str, label: str):
    if st.button(label, key=key, disabled=not bool(text)):
        try:
            with st.spinner("Đang tạo giọng đọc..."):
                st.audio(create_tts_mp3(text, language), format="audio/mp3")
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
    """Normalize punctuation/case/spacing before scoring and highlighting."""
    punctuation = string.punctuation + "，。！？；：、（）【】《》〈〉“”‘’…—·～「」『』"
    normalized = (text or "").lower().translate(str.maketrans({char: " " for char in punctuation}))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if language == "Chinese":
        return list(normalized.replace(" ", ""))
    return normalized.split() if normalized else []


def highlight_dictation(expected: str, typed: str, language: str) -> str:
    """Align word tokens and render wrong, missing, and extra tokens distinctly."""
    expected_tokens, typed_tokens = tokenise(expected, language), tokenise(typed, language)
    output = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, expected_tokens, typed_tokens).get_opcodes():
        if tag == "equal":
            output += [html.escape(token) for token in typed_tokens[j1:j2]]
        elif tag == "replace":
            expected_run = expected_tokens[i1:i2]
            typed_run = typed_tokens[j1:j2]
            paired_count = min(len(expected_run), len(typed_run))
            output.extend(
                f"<span class='mistake'>{html.escape(typed_run[index])}</span>"
                for index in range(paired_count)
            )
            output.extend(
                f"<span class='missing'>thiếu: {html.escape(token)}</span>"
                for token in expected_run[paired_count:]
            )
            output.extend(
                f"<del>{html.escape(token)}</del>"
                for token in typed_run[paired_count:]
            )
        elif tag == "delete":
            output.extend(
                f"<span class='missing'>thiếu: {html.escape(token)}</span>"
                for token in expected_tokens[i1:i2]
            )
        elif tag == "insert":
            output.extend(f"<del>{html.escape(token)}</del>" for token in typed_tokens[j1:j2])
    separator = "" if language == "Chinese" else " "
    return separator.join(output) or "<span class='missing'>Chưa có nội dung hợp lệ</span>"


def generate_dictation(language: str, difficulty: str, source: str):
    if source == "database":
        vocab = db.get_random_vocabulary(user_id, 4)
        if not vocab:
            st.warning("Kho từ vựng đang trống; hãy lưu từ trước.")
            return
        word_list = ", ".join(f"{item['term']} ({item['source_language']})" for item in vocab)
        prompt = f"Tạo đúng một câu {language}, độ khó {difficulty}, dùng tự nhiên một hoặc nhiều từ: {word_list}. Chỉ trả về câu, không giải thích."
    else:
        previous = st.session_state.get("dictation_sentence", "")
        nonce = uuid.uuid4().hex
        prompt = (
            f"Tạo đúng một câu {language} tự nhiên cho bài nghe chép, độ khó {difficulty}, dài 8-18 từ. "
            f"Câu trước cần tránh: {previous!r}. Tuyệt đối không lặp lại câu trước. "
            f"Mã biến thể ngẫu nhiên cho lượt này: {nonce}. Chỉ trả về câu, không giải thích."
        )
    sentence = ask_ai(prompt)
    if sentence:
        new_sentence = sentence.strip().strip('"')
        if source == "random" and new_sentence.casefold() == st.session_state.get("dictation_sentence", "").casefold():
            st.warning("AI vừa lặp lại câu trước; hãy bấm tạo câu ngẫu nhiên lần nữa để lấy câu khác.")
            return
        st.session_state["dictation_sentence"] = new_sentence
        st.session_state["dictation_language"] = language
        st.session_state.pop("dictation_result", None)


def dictionary_prompt(term: str, language: str) -> str:
    return f'''Bạn là giáo viên Anh-Trung-Việt. Phân tích "{term}" (ngôn ngữ: {language}).
Trả về DUY NHẤT JSON hợp lệ gồm term, source_language (English/Chinese), vietnamese_meaning,
other_language, part_of_speech, pronunciation, examples (mảng 3 đối tượng sentence/vietnamese),
synonyms (mảng), antonyms (mảng), tags. Nghĩa và ví dụ phải tự nhiên, chính xác.'''


def specialty_context(major: str, task: str) -> str:
    """Return a domain instruction only when a non-general major is selected."""
    if major == "Tổng quát":
        return ""
    return f"Bạn là chuyên gia về {major}. Hãy tạo {task} liên quan đến lĩnh vực {major}, dùng thuật ngữ thực tế và phù hợp ngữ cảnh nghề nghiệp.\n"


def grade_quiz_locally(questions: list[dict], answers: list[str]) -> tuple[int, list[dict]]:
    """Chấm ngay trên phiên Streamlit; không gọi AI khi người học chọn đáp án."""
    wrong = []
    score = 0
    for question, selected in zip(questions, answers):
        answer = str(question.get("answer", "")).strip()
        options = question.get("options", []) or []
        if len(answer) == 1 and answer.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and ord(answer.upper()) - 65 < len(options):
            correct = str(options[ord(answer.upper()) - 65]).strip().casefold()
        else:
            correct = answer.casefold()
        chosen = str(selected or "").strip()
        if chosen.casefold() == correct and chosen:
            score += 1
        else:
            wrong.append({
                "question": question.get("question", ""),
                "options": question.get("options", []) or [],
                "answer": question.get("answer", ""),
                "explanation": question.get("explanation", ""),
                "selected": chosen,
            })
    return score, wrong


def display_quiz_review(questions: list[dict], answers: list[str], title: str = "Kết quả") -> tuple[int, list[dict]]:
    score, wrong = grade_quiz_locally(questions, answers)
    st.markdown(f"### {title}: {score}/{len(questions)} câu đúng")
    for index, (question, selected) in enumerate(zip(questions, answers), 1):
        options = question.get("options", []) or []
        def labelled(value: str) -> str:
            if len(value) == 1 and value.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and ord(value.upper()) - 65 < len(options):
                option_index = ord(value.upper()) - 65
                return f"[{chr(65 + option_index)}] — {options[option_index]}"
            for option_index, option in enumerate(options):
                if str(option).strip().casefold() == str(value).strip().casefold():
                    return f"[{chr(65 + option_index)}] — {option}"
            return str(value).strip() or "(Chưa trả lời)"

        correct_answer = str(question.get("answer", "")).strip()
        correct_value = labelled(correct_answer) if options else correct_answer
        correct_text = options[ord(correct_answer.upper()) - 65] if options and len(correct_answer) == 1 and correct_answer.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and ord(correct_answer.upper()) - 65 < len(options) else correct_answer
        st.markdown(f"**Câu {index}. {question.get('question', '')}**")
        if str(selected or "").strip().casefold() == str(correct_text).strip().casefold() and str(selected or "").strip():
            st.markdown(f"✅ Đáp án bạn chọn: **{labelled(selected)} (Đúng)**")
        else:
            st.markdown(f"❌ Đáp án bạn chọn: **{labelled(selected)} (Sai)**")
            st.markdown(f"✅ Đáp án đúng: **{correct_value}**")
            st.markdown(f"💡 Giải thích: {question.get('explanation', 'Chưa có giải thích.')}")
    return score, wrong


def show_wrong_question_replay(key: str, quiz_state_key: str, submitted_state_key: str, scope: str) -> None:
    """Cho phép biến những câu sai gần nhất thành một bài luyện mới."""
    wrong_questions = st.session_state.get("wrong_questions", [])
    if wrong_questions and st.session_state.get("wrong_questions_scope") == scope and st.button("🔁 Luyện lại các câu sai", key=key):
        st.session_state[quiz_state_key] = {"questions": [
            {key: value for key, value in item.items() if key in {"question", "options", "answer", "explanation"}}
            for item in wrong_questions
        ]}
        generation_key = f"{quiz_state_key}_generation"
        st.session_state[generation_key] = st.session_state.get(generation_key, 0) + 1
        st.session_state.pop(f"{scope}_review_result", None)
        st.session_state[submitted_state_key] = False
        st.rerun()


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
    stats = db.get_statistics(user_id)
    col_one, col_two = st.columns(2)
    col_one.metric("Từ đã học", stats["words"])
    col_two.metric("Streak", f"{stats['streak']} ngày")
    st.metric("Đến hạn ôn", stats["due"])
    st.divider()
    if st.button("⚙️ Cài đặt API Key", use_container_width=True):
        st.session_state["show_groq_key_settings"] = not st.session_state.get("show_groq_key_settings", False)
    if st.session_state.get("show_groq_key_settings", False):
        with st.expander("Groq API Key cá nhân", expanded=True):
            has_personal_key = bool(db.get_user_groq_key(user_id))
            st.caption("Đã lưu key cá nhân." if has_personal_key else "Chưa có key cá nhân; hiện dùng key mặc định của ứng dụng nếu có.")
            personal_key_input = st.text_input(
                "Nhập key mới", type="password", key="personal_groq_key_input",
                placeholder="gsk_… (để trống để xóa key cá nhân)",
            )
            save_key_col, clear_key_col = st.columns(2)
            if save_key_col.button("Lưu key", key="save_personal_groq_key", use_container_width=True):
                if not personal_key_input.strip():
                    st.warning("Nhập key mới hoặc chọn Xóa key để quay về key mặc định.")
                else:
                    db.update_user_groq_key(user_id, personal_key_input.strip())
                    _cached_groq_client.clear()
                    call_groq_cached.clear()
                    transcribe_audio_cached.clear()
                    st.success("Đã lưu API key cá nhân.")
            if clear_key_col.button("Xóa key", key="clear_personal_groq_key", use_container_width=True):
                db.update_user_groq_key(user_id, None)
                _cached_groq_client.clear()
                call_groq_cached.clear()
                transcribe_audio_cached.clear()
                st.success("Đã xóa key cá nhân. App sẽ dùng key mặc định nếu được cấu hình.")
    st.text_input("Tên hiển thị", value="Language Learner", key="profile_name")
    st.caption("Lưu từ mới và ôn flashcard mỗi ngày để duy trì streak.")

st.markdown("<div class='hero'><h1>LinguaFlow</h1><p>Học Anh · Trung · Việt theo nhịp của bạn.</p></div>", unsafe_allow_html=True)
tab_dictionary, tab_roadmap, tab_grammar, tab_bank, tab_quiz, tab_practice = st.tabs([
    "📖 Từ điển 3 ngữ", "🧭 Lộ trình", "✍️ Ngữ pháp & Ngữ nghĩa", "🗂️ Kho từ & Flashcard", "🎯 Bài tập", "🎭 Nhập vai & Nghe chép"
])

with tab_dictionary:
    dictionary_major = st.selectbox("Chuyên ngành", ["Tổng quát", "Logistics (Xuất nhập khẩu)"], key="dictionary_major")
    term = st.text_input("Từ hoặc cụm từ", placeholder="computer / 学习 / artificial intelligence")
    input_language = st.selectbox("Ngôn ngữ đầu vào", ["Tự nhận diện", "English", "Chinese"])
    if st.button("Phân tích bằng AI", key="lookup") and term.strip():
        with st.spinner("LinguaFlow đang phân tích..."):
            result = ask_ai(specialty_context(dictionary_major, "phân tích từ vựng") + dictionary_prompt(term, input_language), True)
        if result:
            result["major"] = dictionary_major
            st.session_state["dictionary_result"] = result
    result = st.session_state.get("dictionary_result")
    if result:
        render_dictionary(result)
        if st.button("➕ Lưu vào Kho từ vựng"):
            if db.save_vocabulary(result, user_id, dictionary_major):
                db.log_history(user_id, "lookup", result.get("term", ""))
                st.success("Đã lưu từ mới vào kho!")
            else:
                st.info("Từ này đã có trong kho hoặc thiếu dữ liệu bắt buộc.")

with tab_roadmap:
    st.subheader("Lộ trình học cá nhân")
    track = st.selectbox("Chọn kỳ thi", ["TOEIC", "HSK"], key="roadmap_track")
    skill = st.selectbox("Kỹ năng", ["Listening & Reading", "Writing & Speaking"], key="roadmap_skill")
    if track == "TOEIC" and skill == "Listening & Reading":
        levels = ["0-300", "300-450", "450-600", "600-750", "750-850", "850-990"]
    elif track == "TOEIC":
        levels = ["0-100", "100-150", "150-200", "200-250", "250-300"]
    else:
        levels = [f"HSK {level}" for level in range(1, 7)]
    level = st.selectbox("Cấp độ mục tiêu", levels, key="roadmap_level")
    if st.button("✨ Tạo bài học theo cấp độ", key="generate_roadmap"):
        target_language = "tiếng Anh" if track == "TOEIC" else "tiếng Trung"
        if track == "TOEIC" and skill == "Listening & Reading":
            skill_instructions = "Tập trung nghe hiểu và đọc hiểu TOEIC: hội thoại/thông báo ngắn, từ vựng, ngữ pháp, đọc email. Quiz được phép là trắc nghiệm nghe/đọc dạng văn bản; không tạo đề Writing/Speaking."
        elif track == "TOEIC":
            skill_instructions = "Tập trung TOEIC Writing & Speaking. Không tạo quiz hay bất kỳ câu hỏi trắc nghiệm nào. Tạo writing_prompt và speaking_prompt ngắn, rõ ràng, phù hợp trình độ mục tiêu."
        elif skill == "Listening & Reading":
            skill_instructions = "Tập trung nghe hiểu và đọc hiểu HSK: từ vựng, ngữ pháp, hội thoại và đoạn đọc theo cấp độ. Quiz được phép trắc nghiệm dạng văn bản; không tạo bài Writing/Speaking làm trọng tâm."
        else:
            skill_instructions = "Tập trung viết chữ/câu và nói tiếng Trung theo HSK: thứ tự từ, tạo câu, đoạn viết ngắn, chủ đề hội thoại và rubric góp ý. Quiz chỉ kiểm tra từ vựng, cấu trúc câu hoặc viết; tuyệt đối không hỏi trắc nghiệm nghe hiểu."
        if track == "TOEIC" and skill == "Writing & Speaking" and level == "0-100":
            skill_instructions += (
                " BẮT BUỘC trình độ A1 cực dễ: writing_prompt chỉ yêu cầu viết 1-2 câu đơn giản "
                "về bản thân, gia đình, đồ vật hoặc hoạt động quen thuộc; không viết email, không dùng "
                "từ vựng phức tạp, không yêu cầu đoạn văn dài. speaking_prompt cũng chỉ yêu cầu nói "
                "1-2 câu đơn giản theo chủ đề quen thuộc."
            )
        is_writing_speaking = skill == "Writing & Speaking"
        quiz_schema = "quiz (mảng rỗng [])" if is_writing_speaking else "quiz (mảng 5 câu có question/options 4 lựa chọn/answer/explanation)"
        prompt = f'''Tạo bài học {target_language}, mục tiêu {track} {level}, kỹ năng {skill}.
{skill_instructions}
Trả về JSON hợp lệ gồm title, focus, vocabulary (mảng 5 mục có term/meaning/example),
sentence_patterns (mảng 3 mục có pattern/meaning/example), {quiz_schema},
writing_prompt (đề luyện viết phù hợp cấp độ), speaking_prompt (đề luyện nói phù hợp cấp độ).
Với bài có quiz, trong explanation giải thích vì sao đáp án đúng và các phương án nhiễu sai.
Đảm bảo độ khó đúng band/cấp độ; với Writing & Speaking không sinh câu hỏi trắc nghiệm.'''
        with st.spinner("Đang xây dựng bài học..."):
            lesson = ask_ai(prompt, True)
        if lesson:
            st.session_state["roadmap_lesson"] = lesson
            st.session_state["roadmap_generation"] = st.session_state.get("roadmap_generation", 0) + 1
            st.session_state["roadmap_lesson_level"] = (track, skill, level)
            st.session_state.pop("roadmap_prompt_translation", None)
            st.session_state.pop("roadmap_writing_feedback", None)
            st.session_state.pop("roadmap_speaking_feedback", None)
            st.session_state.pop("roadmap_quiz_result", None)
            st.session_state.pop("roadmap_review_test", None)
            st.session_state.pop("roadmap_review_result", None)
            st.session_state["wrong_questions"] = []
            st.session_state.pop("wrong_questions_scope", None)
    lesson = st.session_state.get("roadmap_lesson")
    if lesson:
        saved_track, saved_skill, saved_level = st.session_state.get("roadmap_lesson_level", (track, skill, level))
        roadmap_generation = st.session_state.get("roadmap_generation", 0)
        st.markdown(f"### {lesson.get('title', f'{saved_track} {saved_level}')}\n**{saved_skill}** · {lesson.get('focus', '')}")
        st.markdown("#### Từ vựng")
        if lesson.get("vocabulary"):
            translation_key = get_effective_groq_api_key()
            for index, word in enumerate(lesson["vocabulary"]):
                term_text = str(word.get("term", "")).strip()
                meaning_text = str(word.get("meaning", "")).strip()
                example_text = str(word.get("example", "")).strip()
                with st.container(border=True):
                    st.markdown(f"**{index + 1}. {term_text}** — {meaning_text}")
                    st.caption(example_text)
                    save_col, translate_col = st.columns(2)
                    if save_col.button("💾 Lưu từ này", key=f"roadmap_save_word_{roadmap_generation}_{index}"):
                        item = {
                            "term": term_text,
                            "source_language": "English" if saved_track == "TOEIC" else "Chinese",
                            "vietnamese_meaning": meaning_text,
                            "other_language": "",
                            "part_of_speech": "",
                            "pronunciation": "",
                            "examples": example_text,
                            "synonyms": "",
                            "antonyms": "",
                            "tags": f"{saved_track}, lộ trình",
                            "major": "Tổng quát",
                        }
                        if term_text and meaning_text and db.save_vocabulary(item, user_id):
                            db.log_history(user_id, "roadmap_vocabulary", term_text)
                            st.success(f"Đã lưu “{term_text}” vào Kho từ.")
                        else:
                            st.info("Từ đã có trong kho hoặc thiếu term/meaning để lưu.")
                    if translate_col.button("🇻🇳 Dịch", key=f"roadmap_translate_word_{roadmap_generation}_{index}"):
                        try:
                            if not translation_key:
                                raise RuntimeError("Chưa cấu hình Groq API Key.")
                            translated = translate_roadmap_content_cached(term_text, meaning_text, example_text, translation_key)
                            st.session_state[f"roadmap_word_translation_{roadmap_generation}_{index}"] = translated
                        except Exception as exc:
                            st.error(f"Không dịch được từ này. ({type(exc).__name__})")
                    translated = st.session_state.get(f"roadmap_word_translation_{roadmap_generation}_{index}")
                    if translated:
                        st.info(f"**Nghĩa:** {translated.get('meaning_vi', '—')}\n\n**Ví dụ:** {translated.get('example_vi', '—')}")

        if saved_skill != "Writing & Speaking":
            st.markdown("#### Mẫu câu")
            for pattern in lesson.get("sentence_patterns", []):
                st.markdown(f"- **{pattern.get('pattern', '')}** — {pattern.get('meaning', '')}  \n  {pattern.get('example', '')}")
            st.markdown("#### Trắc nghiệm")
            roadmap_questions = lesson.get("quiz", [])
            with st.form(f"roadmap_quiz_form_{roadmap_generation}"):
                roadmap_answers = []
                for index, question in enumerate(roadmap_questions):
                    st.markdown(f"**{index + 1}. {question.get('question', '')}**")
                    options = question.get("options", []) or []
                    roadmap_answers.append(st.radio("Chọn đáp án", options, index=None, key=f"roadmap_q_{roadmap_generation}_{index}") if options else "")
                roadmap_submitted = st.form_submit_button("Chấm bài lộ trình", use_container_width=True)
            if roadmap_submitted:
                score, wrong = display_quiz_review(roadmap_questions, roadmap_answers, "Kết quả lộ trình")
                st.session_state["wrong_questions"] = wrong
                st.session_state["wrong_questions_scope"] = "roadmap"
                db.save_quiz_result(user_id, f"Roadmap {saved_track} {saved_skill} {saved_level}", score, len(roadmap_questions))
                st.session_state["roadmap_quiz_result"] = (list(roadmap_questions), list(roadmap_answers))
            elif st.session_state.get("roadmap_quiz_result"):
                previous_questions, previous_answers = st.session_state["roadmap_quiz_result"]
                display_quiz_review(previous_questions, previous_answers, "Kết quả lộ trình")
            show_wrong_question_replay("roadmap_wrong_replay", "roadmap_review_test", "roadmap_review_submitted", "roadmap")
            review_test = st.session_state.get("roadmap_review_test")
            if review_test:
                st.markdown("#### Luyện lại câu sai")
                with st.form("roadmap_wrong_questions_form"):
                    review_answers = []
                    for index, question in enumerate(review_test.get("questions", [])):
                        st.markdown(f"**{index + 1}. {question.get('question', '')}**")
                        review_answers.append(st.radio("Chọn đáp án", question.get("options", []) or [], index=None, key=f"roadmap_review_q_{st.session_state.get('roadmap_review_test_generation', 0)}_{index}") if question.get("options") else "")
                    review_submitted = st.form_submit_button("Chấm bài luyện lại", use_container_width=True)
                if review_submitted:
                    score, wrong = display_quiz_review(review_test["questions"], review_answers, "Kết quả luyện lại")
                    st.session_state["wrong_questions"] = wrong
                    st.session_state["roadmap_review_result"] = (list(review_test["questions"]), list(review_answers))
                elif st.session_state.get("roadmap_review_result"):
                    review_questions, review_answers = st.session_state["roadmap_review_result"]
                    display_quiz_review(review_questions, review_answers, "Kết quả luyện lại")

        writing_prompt_text = str(lesson.get("writing_prompt", "—"))
        speaking_prompt_text = str(lesson.get("speaking_prompt", "—"))
        writing, speaking = st.columns(2)
        writing.info(f"✍️ **Writing**\n\n{writing_prompt_text}")
        speaking.info(f"🗣️ **Speaking**\n\n{speaking_prompt_text}")
        if st.button("🇻🇳 Dịch đề bài", key=f"translate_roadmap_prompts_{roadmap_generation}"):
            try:
                translation_key = get_effective_groq_api_key()
                if not translation_key:
                    raise RuntimeError("Chưa cấu hình Groq API Key.")
                combined_prompts = f"Writing: {writing_prompt_text}\nSpeaking: {speaking_prompt_text}"
                st.session_state["roadmap_prompt_translation"] = translate_roadmap_prompt_cached(combined_prompts, translation_key)
            except Exception as exc:
                st.error(f"Không dịch được đề bài. ({type(exc).__name__})")
        if st.session_state.get("roadmap_prompt_translation"):
            st.info(st.session_state["roadmap_prompt_translation"])

        writing_answer = st.text_area("Thực hành Writing", key=f"roadmap_writing_{saved_track}_{saved_skill}_{saved_level}", placeholder="Viết câu trả lời cho đề bài phía trên...")
        if st.button("📝 Chấm bài Writing", key=f"review_roadmap_writing_{roadmap_generation}") and writing_answer.strip():
            feedback = ask_ai(
                f"Chấm bài Writing của người học mục tiêu {saved_track} {saved_level}. Đề bài: {writing_prompt_text}. "
                f"Bài làm: {writing_answer!r}. Trả lời bằng tiếng Việt, rõ ràng, theo đúng bốn mục: "
                "1) Điểm số (0-100, giải thích ngắn tiêu chí); 2) Danh sách lỗi sai cụ thể (trích lỗi, sửa đúng và giải thích); "
                "3) Đoạn văn đã sửa, giữ ý người học; 4) Sample Answer phù hợp đề bài và trình độ. "
                "Nếu không có lỗi, ghi rõ không có lỗi đáng kể. Với TOEIC band 0-100, dùng chuẩn A1 và mẫu chỉ 1-2 câu đơn giản."
            )
            if feedback:
                st.session_state["roadmap_writing_feedback"] = feedback
        if st.session_state.get("roadmap_writing_feedback"):
            st.markdown(st.session_state["roadmap_writing_feedback"])

        speaking_recording = st.audio_input("Ghi âm luyện Speaking", key=f"roadmap_speaking_{saved_track}_{saved_skill}_{saved_level}")
        if st.button("🎙️ Chấm bài Speaking", key=f"review_roadmap_speaking_{roadmap_generation}", disabled=speaking_recording is None):
            try:
                transcript = transcribe_audio_cached(speaking_recording.getvalue(), getattr(speaking_recording, "type", "audio/webm"), get_effective_groq_api_key())
                feedback = ask_ai(
                    f"Chấm phần Speaking mục tiêu {saved_track} {saved_level}. Đề bài: {speaking_prompt_text}. "
                    f"Whisper nhận diện được transcript: {transcript!r}. Trả lời bằng tiếng Việt, theo đúng bốn mục: "
                    "1) Điểm số (0-100, đánh giá nội dung và độ rõ diễn đạt, không tự suy đoán chất lượng âm thanh); "
                    "2) Danh sách lỗi sai cụ thể trong transcript; 3) Đoạn/câu đã sửa; 4) Sample Answer ngắn phù hợp đề bài. "
                    "Chỉ nhận xét phát âm trong phạm vi transcript cho phép. Với TOEIC band 0-100, trình độ A1, mẫu 1-2 câu đơn giản."
                )
                if feedback:
                    st.session_state["roadmap_speaking_feedback"] = f"**Transcript:** {transcript}\n\n{feedback}"
            except Exception as exc:
                st.error(f"Không thể phân tích bài nói. ({type(exc).__name__})")
        if st.session_state.get("roadmap_speaking_feedback"):
            st.markdown(st.session_state["roadmap_speaking_feedback"])
        if st.button("💾 Lưu bài học vào tài khoản", key="save_roadmap"):
            db.save_roadmap_lesson(user_id, saved_track, f"{saved_skill} · {saved_level}", json.dumps(lesson, ensure_ascii=False))
            db.log_history(user_id, "roadmap", f"{saved_track} {saved_skill} {saved_level}")
            st.success("Đã lưu bài học vào database.")
    with st.expander("Bài học đã lưu"):
        lessons = db.get_roadmap_lessons(user_id)
        if lessons:
            for saved in lessons:
                saved_content = json.loads(saved["content"])
                st.markdown(f"**{saved['track']} {saved['level']} · {saved_content.get('title', 'Bài học')}** — {saved['created_at']}")
        else:
            st.caption("Chưa lưu bài học nào.")

with tab_grammar:
    mode = st.radio("Chế độ", ["Kiểm tra ngữ pháp", "So sánh hai câu"], horizontal=True)
    if mode == "Kiểm tra ngữ pháp":
        text = st.text_area("Đoạn tiếng Anh hoặc Trung", height=160, placeholder="I has went to school yesterday.")
        if st.button("Kiểm tra & giải thích", key="grammar") and text.strip():
            answer = ask_ai(f"Phân tích ngữ pháp: {text}\nTrả lời tiếng Việt Markdown: đánh giá, bản sửa, giải thích từng lỗi và in đậm từ sai.")
            if answer:
                st.markdown(answer)
                db.log_history(user_id, "grammar", text[:80])
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
        available_majors = sorted({row.get("major", "Tổng quát") for row in db.get_vocabulary(user_id) if row.get("major")})
        filter_major = st.selectbox("Lọc chuyên ngành", ["Tất cả"] + available_majors)
    with right:
        st.download_button("⬇️ Xuất toàn bộ CSV", db.export_csv(user_id), "linguaflow_vocabulary.csv", "text/csv")
    vocab = db.get_vocabulary(user_id, filter_language, tag, filter_major)
    if vocab:
        columns = ["id", "term", "pronunciation", "source_language", "vietnamese_meaning", "other_language", "part_of_speech", "major", "tags"]
        original = pd.DataFrame(vocab)[columns]
        edited = st.data_editor(original, key="vocabulary_editor", num_rows="dynamic", hide_index=True, use_container_width=True, disabled=["id"], column_config={"id": st.column_config.NumberColumn("ID", width="small")})
        if st.button("💾 Lưu thay đổi Kho từ vựng"):
            old_by_id = {int(row["id"]): row for _, row in original.iterrows()}
            kept_ids = {int(row["id"]) for _, row in edited.dropna(subset=["id"]).iterrows()}
            for vocab_id in set(old_by_id) - kept_ids:
                db.delete_vocabulary(vocab_id, user_id)
            for _, row in edited.iterrows():
                item = row.fillna("").to_dict()
                vocab_id = item.pop("id", None)
                if pd.isna(vocab_id) or not vocab_id:
                    db.save_vocabulary(item, user_id)
                elif item != {field: old_by_id[int(vocab_id)][field] for field in item}:
                    db.update_vocabulary(int(vocab_id), item, user_id)
            st.success("Đã lưu các thay đổi.")
            st.rerun()
    else:
        st.info("Kho từ vựng đang trống. Hãy lưu một kết quả từ tab Từ điển.")

    st.divider()
    st.subheader("🧠 Flashcard SM-2")
    due_cards = db.get_due_cards(user_id)
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
                    db.review_card(card["id"], quality, user_id)
                    st.session_state["card_index"] = card_index + 1
                    st.rerun()
    else:
        st.success("Không có từ đến hạn ôn hôm nay. Tuyệt vời!")

with tab_quiz:
    st.write("AI sinh bài tập từ Kho từ vựng của bạn.")
    quiz_major = st.selectbox("Chuyên ngành", ["Tổng quát", "Logistics (Xuất nhập khẩu)"], key="quiz_major")
    quiz_type = st.selectbox("Dạng bài", ["Trắc nghiệm", "Điền từ", "Sắp xếp câu"])
    if st.button("Tạo bài tập", key="make_quiz"):
        words = db.get_vocabulary(user_id, major=quiz_major) if quiz_major != "Tổng quát" else db.get_vocabulary(user_id)
        if not words and quiz_major != "Tổng quát":
            # Cho phép tạo quiz chuyên ngành từ kho từ hiện có nếu chưa lưu từ đúng ngành.
            words = db.get_vocabulary(user_id)
            if words:
                st.info("Chưa có từ đã lưu đúng chuyên ngành; AI sẽ tạo quiz theo chuyên ngành đã chọn dựa trên kho từ hiện có.")
        words = words[:30]
        if not words:
            st.warning("Chưa có từ vựng cho chuyên ngành này. Hãy lưu từ từ tab Từ điển hoặc chọn Tổng quát.")
        else:
            compact = [{"term": word["term"], "meaning": word["vietnamese_meaning"], "lang": word["source_language"]} for word in words]
            prompt = specialty_context(quiz_major, f"bài tập {quiz_type} và câu hỏi quiz")
            prompt += f"Tạo 5 bài {quiz_type} từ: {json.dumps(compact, ensure_ascii=False)}. Chỉ JSON: {{'questions':[{{'question':'','options':['','','',''],'answer':'','explanation':''}}]}}. Answer phải là chính xác nội dung lựa chọn đúng hoặc chữ cái A/B/C/D. Trong explanation, nêu vì sao đáp án đúng đúng và vì sao lựa chọn nhiễu sai. Điền từ/sắp xếp có options=[]."
            quiz = ask_ai(prompt, True)
            if quiz:
                st.session_state["quiz"] = quiz
                st.session_state["quiz_type"] = quiz_type
                st.session_state["quiz_generation"] = st.session_state.get("quiz_generation", 0) + 1
                st.session_state["quiz_submitted"] = False
                st.session_state.pop("quiz_result", None)
                st.session_state.pop("quiz_review_test", None)
                st.session_state.pop("quiz_review_result", None)
                st.session_state["wrong_questions"] = []
                st.session_state.pop("wrong_questions_scope", None)
    quiz = st.session_state.get("quiz")
    if quiz:
        quiz_questions = quiz.get("questions", [])
        generation = st.session_state.get("quiz_generation", 0)
        with st.form(f"quiz_form_{generation}"):
            quiz_answers = []
            for index, question in enumerate(quiz_questions):
                st.markdown(f"**Câu {index + 1}.** {question.get('question', '')}")
                options = question.get("options", []) or []
                if options:
                    quiz_answers.append(st.radio("Đáp án", options, index=None, key=f"quiz_answer_{generation}_{index}"))
                else:
                    quiz_answers.append(st.text_input("Câu trả lời", key=f"quiz_answer_{generation}_{index}"))
            quiz_submitted = st.form_submit_button("Nộp bài", use_container_width=True)
        if quiz_submitted:
            score, wrong = display_quiz_review(quiz_questions, quiz_answers)
            st.session_state["wrong_questions"] = wrong
            st.session_state["wrong_questions_scope"] = "quiz"
            db.save_quiz_result(user_id, st.session_state.get("quiz_type", quiz_type), score, len(quiz_questions))
            st.session_state["quiz_result"] = (list(quiz_questions), list(quiz_answers))
            st.session_state["quiz_submitted"] = True
        elif st.session_state.get("quiz_result"):
            previous_questions, previous_answers = st.session_state["quiz_result"]
            display_quiz_review(previous_questions, previous_answers)
        show_wrong_question_replay("quiz_wrong_replay", "quiz_review_test", "quiz_review_submitted", "quiz")
        review_test = st.session_state.get("quiz_review_test")
        if review_test:
            st.markdown("#### Luyện lại câu sai")
            with st.form("quiz_wrong_questions_form"):
                review_answers = []
                for index, question in enumerate(review_test.get("questions", [])):
                    st.markdown(f"**Câu {index + 1}.** {question.get('question', '')}")
                    options = question.get("options", []) or []
                    review_generation = st.session_state.get("quiz_review_test_generation", 0)
                    review_answers.append(st.radio("Đáp án", options, index=None, key=f"quiz_review_answer_{review_generation}_{index}") if options else st.text_input("Câu trả lời", key=f"quiz_review_answer_{review_generation}_{index}"))
                review_submitted = st.form_submit_button("Nộp bài luyện lại", use_container_width=True)
            if review_submitted:
                score, wrong = display_quiz_review(review_test["questions"], review_answers, "Kết quả luyện lại")
                st.session_state["wrong_questions"] = wrong
                st.session_state["quiz_review_result"] = (list(review_test["questions"]), list(review_answers))
            elif st.session_state.get("quiz_review_result"):
                review_questions, review_answers = st.session_state["quiz_review_result"]
                display_quiz_review(review_questions, review_answers, "Kết quả luyện lại")

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
                db.log_history(user_id, "roleplay", situation)
                st.rerun()
        st.markdown("**Hoặc luyện nói bằng giọng nói**")
        voice_recording = st.audio_input("Ghi âm câu trả lời", key="roleplay_audio")
        if st.button("🎙️ Chuyển giọng nói thành chat", key="transcribe_roleplay", disabled=voice_recording is None):
            try:
                with st.spinner("Groq Whisper đang nhận diện giọng nói..."):
                    transcript = transcribe_audio_cached(
                        voice_recording.getvalue(),
                        getattr(voice_recording, "type", "audio/webm"),
                        get_effective_groq_api_key(),
                    ).strip()
                if not transcript:
                    st.warning("Không nhận diện được lời nói. Hãy thử thu âm lại.")
                else:
                    st.session_state.chat.append({"role": "user", "content": f"🎙️ {transcript}"})
                    voice_prompt = (
                        f"Đang roleplay tình huống {situation} bằng {language}, độ khó {difficulty}. "
                        f"Người học nói qua micro: {transcript!r}. Trả lời tự nhiên bằng {language}. "
                        "Sau đó thêm mục 'Nhận xét ngôn ngữ' bằng tiếng Việt: sửa lỗi ngữ pháp trong câu nói, "
                        "nêu 1-2 gợi ý phát âm cho từ/cụm từ đáng chú ý dựa trên transcript. "
                        "Không khẳng định đã nghe được âm vị chi tiết nếu chỉ có transcript."
                    )
                    response = ask_ai(voice_prompt)
                    if response:
                        st.session_state.chat.append({"role": "assistant", "content": response})
                        db.log_history(user_id, "roleplay_voice", transcript[:200])
                        st.rerun()
            except Exception as exc:
                st.error(f"Không thể chuyển giọng nói thành văn bản. Kiểm tra Groq API/hạn mức. ({type(exc).__name__})")
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
                st.audio(create_tts_mp3(sentence, sentence_language), format="audio/mp3")
            except Exception as exc:
                st.warning(f"Không tạo được âm thanh. ({type(exc).__name__})")
            typed = st.text_input("Gõ lại câu bạn nghe được", key="dictation_typed")
            if st.button("Chấm điểm nghe chép") and typed.strip():
                ratio = difflib.SequenceMatcher(None, tokenise(sentence.lower(), sentence_language), tokenise(typed.lower(), sentence_language)).ratio() * 100
                st.success(f"Độ chính xác: {ratio:.0f}%")
                st.markdown("**Bài bạn gõ** (đỏ = sai, vàng = phần bị thiếu):")
                st.markdown(highlight_dictation(sentence, typed, sentence_language), unsafe_allow_html=True)
                st.caption(f"Câu chuẩn: {sentence}")
                db.log_history(user_id, "dictation", sentence_language, ratio)
