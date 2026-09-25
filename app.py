from google import genai
from google.genai import types
import streamlit as st
import os
import re
import io
import tempfile
import time
import json
from pathlib import Path
from pydub import AudioSegment
from opencc_purepy import OpenCC

# 嘗試自動從環境變數讀取 API Key
default_key = os.environ.get("GEMINI_API_KEY", "")

st.set_page_config(page_title="Gemini 聽打小幫手", page_icon="🎙️", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 16px; }
    .stTextArea textarea { font-size: 16px !important; }
    .stTextInput input { font-size: 16px !important; }
    .stRadio label, .stCheckbox label, .stMarkdown p { font-size: 16px !important; }
    .stHeader a, h1 a, h2 a, h3 a { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎙️ Gemini 聽打小幫手")
st.markdown(
    "支援錄音與音訊上傳，提供多種 AI 處理模式、排版風格與基本文字編輯功能。"
)
st.markdown("---")

# 側邊欄：系統設定與各項功能開關
with st.sidebar:
    st.header("⚙️ 系統設定")
    api_key = st.text_input(
        "Gemini API Key", 
        value=default_key, 
        type="password",
        placeholder="請輸入你的 API Key"
    )

    # 選擇模型（預設為 gemini-3.5-flash-lite，改為整體性提示）
    selected_model = st.selectbox(
        "選擇 Gemini 模型",
        ["gemini-3.5-flash-lite", "gemini-3.5-flash"],
        index=0,
        help="flash-lite 模型速度較快且時間點較準，flash 模型理解與綜合表現較好。"
    )

    st.markdown("---")
    st.header("🧠 處理模式")
    task_mode = st.selectbox(
        "選擇任務類型：",
        [
            "逐字稿",
            "會議記錄與摘要",
            "重點行動清單",
        ],
    )

    # 動態聯動：根據不同的處理模式，提供對應的排版風格選項
    st.markdown("---")
    st.header("📐 排版風格")
    
    srt_start_offset = 0.0
    srt_end_offset = 0.2
    srt_gap_threshold = 0.5
    subtitle_max_chars = 14

    if task_mode == "逐字稿":
        output_mode = st.radio(
            "選擇排版方式：",
            [
                "文章段落",
                "對話分行",
                "影視字幕",
            ],
        )
        
        # 🎬 如果選到影視字幕，保留時間微調設定
        if output_mode == "影視字幕":
            st.markdown("🎬 **影視字幕時間微調**")
            
            srt_start_offset = st.slider(
                "起始點提早 (秒)",
                min_value=0.0,
                max_value=0.5,
                value=0.0,
                step=0.1,
                help="將每句字幕的開始時間向前延伸，0代表不提早。"
            )
            srt_end_offset = st.slider(
                "結束點延長 (秒)",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.1,
                help="將每句字幕的結束時間向後延伸，0代表不延長。"
            )
            srt_gap_threshold = st.slider(
                "自動黏貼間隔 (秒)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="若前後兩句間隔小於此秒數，將自動延長前句以無縫接軌，解決畫面閃爍。"
            )
            subtitle_max_chars = st.slider(
                "每行最多字數",
                min_value=6,
                max_value=24,
                value=14,
                step=1,
                help="字幕智慧斷句的硬上限。AI 會優先依語意、詞組與停頓斷句；只有無法自然切分時才接近此上限。"
            )

    elif task_mode == "會議記錄與摘要":
        output_mode = st.radio(
            "選擇排版方式：",
            [
                "文章段落",
                "結構化大綱",
            ],
        )
    else:  # 行動清單
        output_mode = st.radio(
            "選擇排版方式：",
            [
                "項目符號 (Bullet points)",
                "數字序號 (1, 2, 3...)",
                "步驟流程圖 (Step-by-step)",
            ],
        )

    st.markdown("---")
    st.header("✍️ 智慧校稿")
    proofreading_level = st.selectbox(
        "校稿程度",
        options=["基本", "通順", "正式"],
        index=0,
    )

    st.markdown("---")
    st.header("😀 表情符號")
    use_emoji = st.checkbox("自動加入 Emoji", value=False)
    emoji_level = "中等"
    if use_emoji:
        emoji_level = st.select_slider(
            "使用頻率", options=["少量", "中等", "大量"], value="中等"
        )


# -----------------------------
# 音訊 / Gemini / SRT 輔助函式
# -----------------------------
TRANSCRIBE_MODEL = "gemini-3.5-transcribe"
CHUNK_MS = 10 * 60 * 1000       # 10 分鐘一段；1 小時約 6 段
CHUNK_OVERLAP_MS = 1000         # 前後重疊 1 秒，降低切在字中間的風險
MAX_SUBTITLE_CHARS = 14         # 預設單行最多字元；實際值由側欄控制
MIN_SUBTITLE_DURATION = 0.80
MAX_SUBTITLE_DURATION = 4.00
BREAK_GAP_SEC = 0.35

# OpenCC：將 Transcribe 的簡體中文轉成臺灣常用繁體，且不碰時間戳
try:
    _opencc_tw = OpenCC("s2twp")
except Exception:
    _opencc_tw = None


def guess_mime_and_format(uploaded_name: str, is_live: bool = False):
    """回傳 (mime_type, pydub_format)。現場錄音固定為 WAV。"""
    if is_live:
        return "audio/wav", "wav"
    ext = Path(uploaded_name or "").suffix.lower().lstrip(".")
    mapping = {
        "mp3": ("audio/mp3", "mp3"),
        "wav": ("audio/wav", "wav"),
        "m4a": ("audio/m4a", "m4a"),
        "aac": ("audio/aac", "aac"),
        "ogg": ("audio/ogg", "ogg"),
        "flac": ("audio/flac", "flac"),
        "aiff": ("audio/aiff", "aiff"),
        "opus": ("audio/opus", "opus"),
        "webm": ("audio/webm", "webm"),
    }
    return mapping.get(ext, ("application/octet-stream", ext or "wav"))


def split_audio_bytes(audio_bytes: bytes, source_format: str):
    """將音訊切成約 10 分鐘 WAV，回傳 [(bytes, offset_sec), ...]。"""
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=source_format)
    total_ms = len(audio)
    if total_ms <= CHUNK_MS:
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        return [(buf.getvalue(), 0.0)]

    chunks = []
    start = 0
    while start < total_ms:
        end = min(start + CHUNK_MS, total_ms)
        # 除最後一段外，讓下一段向前重疊 1 秒。
        chunk_start = max(0, start - (CHUNK_OVERLAP_MS if start > 0 else 0))
        chunk = audio[chunk_start:end]
        buf = io.BytesIO()
        chunk.export(buf, format="wav")
        chunks.append((buf.getvalue(), chunk_start / 1000.0))
        start = end
    return chunks


def parse_offset_seconds(value):
    """Google annotation 的 offset 通常是 '0.100s'，也兼容 timedelta。"""
    if value is None:
        return None
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    s = str(value).strip()
    if s.endswith("s"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
        return float(m.group(1)) if m else None


def extract_word_annotations(interaction):
    """依照 Google 官方文件，從 interaction.steps/content/annotations 擷取 word_info。"""
    words = []
    for step in getattr(interaction, "steps", []) or []:
        for content in getattr(step, "content", []) or []:
            for annotation in getattr(content, "annotations", []) or []:
                if getattr(annotation, "type", None) == "word_info":
                    text = getattr(annotation, "text", None)
                    start = parse_offset_seconds(getattr(annotation, "start_offset", None))
                    end = parse_offset_seconds(getattr(annotation, "end_offset", None))
                    if text and start is not None and end is not None:
                        words.append({
                            "text": str(text),
                            "start_sec": start,
                            "end_sec": end,
                            "speaker": getattr(annotation, "speaker", None),
                        })
    return words


def normalize_word_for_dedupe(text):
    return re.sub(r"\s+", "", str(text)).strip().lower()


def transcribe_one_chunk(client, chunk_bytes, offset_sec, progress_text=None):
    """使用 Gemini 3.5 Transcribe + word-level timestamps。"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(chunk_bytes)
        temp_path = f.name
    audio_file = None
    try:
        # 明確指定上傳檔的 MIME type，避免 macOS / Python mimetypes
        # 把 .wav 判成 audio/x-wav，進而與 Interactions API 的 audio/wav 不一致。
        audio_file = client.files.upload(
            file=temp_path,
            config=types.UploadFileConfig(mime_type="audio/wav"),
        )
        uploaded_mime = getattr(audio_file, "mime_type", None) or "audio/wav"
        # Gemini 3.5 Transcribe 官方支援的 WAV MIME 是 audio/wav。
        # 若 SDK/環境仍回報 audio/x-wav，統一成 API 要求的 audio/wav。
        if uploaded_mime == "audio/x-wav":
            uploaded_mime = "audio/wav"

        interaction = client.interactions.create(
            model=TRANSCRIBE_MODEL,
            input=[
                {
                    "type": "audio",
                    "uri": audio_file.uri,
                    "mime_type": uploaded_mime,
                }
            ],
            generation_config={
                "transcription_config": {
                    "mode": {
                        "type": "verbatim",
                        "timestamp_granularities": ["word"],
                    },
                    # 不鎖死語言，讓模型自動偵測並處理中英日韓夾雜。
                    "language_codes": [],
                }
            },
        )
        words = extract_word_annotations(interaction)
        if not words:
            # API/SDK 變動時，至少保留完整 transcript，避免整段資料遺失。
            fallback = getattr(interaction, "output_text", "") or ""
            if fallback.strip():
                return [{
                    "text": fallback.strip(),
                    "start_sec": offset_sec,
                    "end_sec": offset_sec + 2.0,
                    "speaker": None,
                }], True
            return [], False

        for w in words:
            w["start_sec"] += offset_sec
            w["end_sec"] += offset_sec
        return words, False
    finally:
        if audio_file is not None:
            try:
                client.files.delete(name=audio_file.name)
            except Exception:
                pass
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def transcribe_with_timestamps(client, audio_bytes, source_format, progress_bar=None, status_box=None):
    """自動切片；word timestamps 啟用時每段控制在 10 分鐘，避免 30 分鐘上限。"""
    chunks = split_audio_bytes(audio_bytes, source_format)
    all_words = []
    fallback_used = False

    for i, (chunk_bytes, offset_sec) in enumerate(chunks, 1):
        if status_box:
            status_box.info(f"🎙️ 正在辨識第 {i}/{len(chunks)} 段")
        # API 偶發 429/5xx 時做少量退避重試，不無限重試。
        last_error = None
        for attempt in range(3):
            try:
                words, used_fallback = transcribe_one_chunk(
                    client, chunk_bytes, offset_sec
                )
                fallback_used = fallback_used or used_fallback
                all_words.extend(words)
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if last_error is not None:
            raise last_error
        if progress_bar:
            progress_bar.progress(i / len(chunks))

    # 依時間排序，並去除切片重疊造成的重複詞。
    all_words.sort(key=lambda x: (x["start_sec"], x["end_sec"]))
    deduped = []
    for word in all_words:
        if deduped:
            prev = deduped[-1]
            overlap = word["start_sec"] < prev["end_sec"] + 0.15
            same = normalize_word_for_dedupe(word["text"]) == normalize_word_for_dedupe(prev["text"])
            if overlap and same:
                # 保留較早、較完整的時間範圍。
                prev["end_sec"] = max(prev["end_sec"], word["end_sec"])
                continue
        deduped.append(word)
    return deduped, len(chunks), fallback_used


def clean_subtitle_token(text):
    """字幕模式移除標點與多餘空白；保留英數與中文/日文/韓文。"""
    text = str(text).strip()
    # 移除常見中英標點、全形標點及括號；不要用 \W，否則會破壞 Unicode 文字。
    text = re.sub(r"[，。！？；：、,.!?;:：；「」『』（）()【】［］\[\]{}<>〈〉《》…—–-]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def join_subtitle_tokens(tokens):
    """中文不插空格；英文/數字 token 之間保留一格。"""
    out = ""
    for token in tokens:
        t = clean_subtitle_token(token)
        if not t:
            continue
        if not out:
            out = t
            continue
        # 兩邊都是 ASCII 英數時插空格；中文與英文相鄰則不強插空格。
        if re.search(r"[A-Za-z0-9]$", out) and re.match(r"^[A-Za-z0-9]", t):
            out += " " + t
        else:
            out += t
    return out


def _subtitle_text_len(text):
    """字幕字數：中文、英數都計 1；空白不計。"""
    return len(re.sub(r"\s+", "", str(text)))


def traditionalize_text(text):
    """轉成臺灣正體／常用詞；失敗時保留原文。"""
    if not text:
        return text
    if _opencc_tw is None:
        return text
    try:
        return _opencc_tw.convert(text)
    except Exception:
        return text


def build_candidate_groups(words, max_chars):
    """保留給舊流程使用；目前字幕斷句改由每個音訊 chunk 一次交給 Gemini 判斷。"""
    return [{"words": words, "start_sec": words[0]["start_sec"], "end_sec": words[-1]["end_sec"]}] if words else []


def ask_gemini_for_breaks(client, words, max_chars, model):
    """只讓 Gemini 決定「在哪個 token 後斷句」，不允許它改文字或時間。

    影視字幕以「一般影片」而非短影音為目標：在不超過使用者上限的前提下，
    優先保留完整語意與較長句群，避免因為短暫停頓或單一子句就頻繁換行。
    """
    if len(words) <= 1:
        return []

    numbered = []
    prev_end = None
    for i, w in enumerate(words):
        token = clean_subtitle_token(w["text"])
        if not token:
            continue
        # 提供模型「這個 token 前的停頓」作為輔助，但不讓短停頓直接等於斷句。
        gap = 0.0 if prev_end is None else max(0.0, w["start_sec"] - prev_end)
        numbered.append(f"[{i}] {token} <gap_before={gap:.2f}s>")
        prev_end = w["end_sec"]
    if not numbered:
        return []

    # 一般影片字幕希望「少而完整」，而不是短影音式一小句一行。
    # max_chars 是硬上限；preferred_min 用來告訴模型通常應盡量把句子延伸到合理長度。
    preferred_min = max(6, int(round(max_chars * 0.68)))
    preferred_soft = max(preferred_min + 1, int(round(max_chars * 0.90)))

    prompt = (
        "你是專業的臺灣影視字幕斷句器，目標是『一般 YouTube／電視／新聞影片字幕』，不是短影音字幕。\n"
        "以下文字已由語音辨識模型取得精確的逐字時間戳；你的任務只有決定『在哪個 token 後換成下一行字幕』。\n\n"
        f"硬性上限：每行最多 {max_chars} 個字元（不計空白）。\n"
        f"建議長度：一般情況盡量落在約 {preferred_min}～{max_chars} 字；若語意完整且接近 {preferred_soft}～{max_chars} 字，優先維持同一行。\n"
        "核心原則：寧可保留較完整、較長的一句，也不要把字幕切成短影音式的碎片。\n\n"
        "請嚴格遵守：\n"
        "1. 絕對不要修改、翻譯、增加或刪除任何文字。\n"
        "2. 只能在兩個 token 之間斷句，回傳 break_after 的 token index。\n"
        "3. 以『完整語意單位』為最高優先：主語、動詞、受詞、時間／地點、修飾語能自然連在一起時，盡量不要拆開。\n"
        "4. 不要因為短暫停頓就換行。小於約 0.8 秒的停頓通常不足以單獨構成字幕斷點；除非同時有明顯的語意完成。\n"
        "5. 優先在完整句子、完整子句或明顯語意轉折處斷句；不要把一個完整子句拆成兩個很短的片段。\n"
        "6. 連接詞（例如『但、而且、所以、因此、如果、因為、原來、還指出、也、並、以及』）附近不要機械式換行；若前後語意仍屬同一單位，盡量連在一起。\n"
        "7. 常見詞組、人名、地名、品牌名、機構名、職稱、英文單字、數字、固定搭配絕對不要拆開。\n"
        "8. 只有在接近硬性上限、或繼續往後會造成明顯超長／語意過度擁擠時，才應該換行。\n"
        "9. 如果目前只有 5～8 個字，但後面仍能自然接成同一個完整語意單位，請不要急著斷句。\n"
        "10. 不要為了讓每行長度看起來平均而換行；自然語意比平均字數重要。\n"
        "11. 最後一行可以比建議長度短，因為它可能只是段落的收尾；但中間字幕行不要無故過短。\n"
        "12. 停頓時間只作輔助：較長停頓（例如約 0.8 秒以上）可以提高斷句優先度，但仍必須確認語意完整。\n"
        "13. 除非超過硬性上限，避免產生只有一個短子句或三、四個詞的字幕行。\n"
        "14. 只輸出 JSON：{\"break_after\":[整數索引...]}，索引必須遞增。\n\n"
        "理想方向示例：\n"
        "不要：『在韓國出道後』→『人氣暴漲』\n"
        "較自然：『在韓國出道後人氣暴漲』\n\n"
        "不要：『日前還風光』→『奪下金曲獎』→『最佳演唱組合』→『的寶座』\n"
        "較自然：『日前還風光奪下金曲獎』→『最佳演唱組合的寶座』\n\n"
        "不要：『坦承自己』→『沒有查證』→『就用看到的貼文』\n"
        "較自然：『坦承自己沒有查證』→『就用看到的貼文與留言發文』\n\n"
        "Token（<gap_before> 是該 token 前面的停頓秒數，只供參考）：\n"
        + "\n".join(numbered)
    )

    schema = {
        "type": "object",
        "properties": {
            "break_after": {"type": "array", "items": {"type": "integer"}}
        },
        "required": ["break_after"]
    }
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        data = json.loads(response.text or "{}")
        breaks = data.get("break_after", []) if isinstance(data, dict) else []
        valid = sorted({int(x) for x in breaks if isinstance(x, int) or str(x).lstrip("-").isdigit()})
        return [i for i in valid if 0 <= i < len(words) - 1]
    except Exception:
        return []

def build_segments_from_breaks(words, breaks):
    """依 Gemini 回傳的 token 斷點建立字幕，時間完全取 word timestamp。"""
    break_set = set(breaks)
    segments = []
    current = []
    for idx, word in enumerate(words):
        if not clean_subtitle_token(word["text"]):
            continue
        current.append(word)
        if idx in break_set:
            text = join_subtitle_tokens([w["text"] for w in current])
            if text:
                segments.append({"start_sec": current[0]["start_sec"], "end_sec": current[-1]["end_sec"], "text": text})
            current = []
    if current:
        text = join_subtitle_tokens([w["text"] for w in current])
        if text:
            segments.append({"start_sec": current[0]["start_sec"], "end_sec": current[-1]["end_sec"], "text": text})
    return segments


def _fallback_breaks(words, max_chars):
    """Gemini 失敗時的保底斷句：優先停頓，再考慮字數，避免硬切詞。"""
    breaks = []
    current_start = 0
    current_text = ""
    last_good = None
    for i, word in enumerate(words):
        token = clean_subtitle_token(word["text"])
        if not token:
            continue
        candidate = join_subtitle_tokens([w["text"] for w in words[current_start:i + 1]])
        gap = 0.0
        if i > current_start:
            gap = max(0.0, word["start_sec"] - words[i - 1]["end_sec"])
        if gap >= BREAK_GAP_SEC and _subtitle_text_len(current_text) >= max(4, max_chars // 2):
            breaks.append(i - 1)
            current_start = i
            current_text = token
            last_good = None
            continue
        current_text = candidate
        if _subtitle_text_len(current_text) <= max_chars:
            last_good = i
        elif last_good is not None:
            breaks.append(last_good)
            current_start = last_good + 1
            current_text = join_subtitle_tokens([w["text"] for w in words[current_start:i + 1]])
            last_good = i if _subtitle_text_len(current_text) <= max_chars else None
    return breaks


def merge_short_subtitle_segments(segments, max_chars):
    """後處理：把明顯過短、且能安全與相鄰字幕合併的碎片收回。

    這是針對一般影片的保底，不會為了湊字數強行合併；只有短行、合併後不超過上限、
    且兩句之間沒有明顯停頓時才合併。時間軸仍直接使用原字幕的首尾 word timestamp。
    """
    if len(segments) < 2:
        return segments

    result = [dict(segments[0])]
    # 4 字以下視為特別容易造成「短影音碎片感」的短行。
    very_short = 4

    i = 1
    while i < len(segments):
        cur = dict(segments[i])
        prev = result[-1]
        prev_len = _subtitle_text_len(prev["text"])
        cur_len = _subtitle_text_len(cur["text"])
        gap = max(0.0, cur["start_sec"] - prev["end_sec"])
        combined_len = _subtitle_text_len(prev["text"] + cur["text"])

        # 優先把極短行併回前句；短暫停頓不足以阻止合併。
        if (
            cur_len <= very_short
            and combined_len <= max_chars
            and gap < 0.65
            and prev_len > 0
        ):
            prev["text"] = join_subtitle_tokens([prev["text"], cur["text"]])
            prev["end_sec"] = cur["end_sec"]
            i += 1
            continue

        # 如果上一行極短，也嘗試和目前這行合併；這可處理「公司」→「將採取法律行動」這類碎片。
        if (
            prev_len <= very_short
            and combined_len <= max_chars
            and gap < 0.65
        ):
            prev["text"] = join_subtitle_tokens([prev["text"], cur["text"]])
            prev["end_sec"] = cur["end_sec"]
            i += 1
            continue

        result.append(cur)
        i += 1

    return result

def build_subtitle_segments(words, client=None, max_chars=14, model="gemini-3.5-flash-lite"):
    """以 word timestamps 為唯一時間來源；Gemini 一次判斷整個 chunk 的自然斷點。"""
    if not words:
        return []

    breaks = ask_gemini_for_breaks(client, words, max_chars, model) if client is not None else []
    if not breaks:
        breaks = _fallback_breaks(words, max_chars)

    segments = build_segments_from_breaks(words, breaks)
    # Gemini 斷得過碎時，用非常保守的規則收回極短字幕行；不改變 word timestamp。
    segments = merge_short_subtitle_segments(segments, max_chars)
    if not segments:
        text = join_subtitle_tokens([w["text"] for w in words])
        segments = [{"start_sec": words[0]["start_sec"], "end_sec": words[-1]["end_sec"], "text": text}]

    # 檢查 Gemini 是否留下過長字幕。不要在詞中硬切；若超長，保留完整詞組，避免生硬斷行。
    for seg in segments:
        if _subtitle_text_len(seg["text"]) > max_chars:
            seg["over_limit"] = True

    # 套用 UI 的時間微調；再做不重疊處理。
    processed = []
    for seg in segments:
        s = max(0.0, seg["start_sec"] - srt_start_offset)
        e = max(s + 0.05, seg["end_sec"] + srt_end_offset)
        processed.append({
            "start_sec": s,
            "end_sec": e,
            "text": traditionalize_text(seg["text"]),
            "over_limit": seg.get("over_limit", False),
        })

    for i in range(len(processed) - 1):
        gap = processed[i + 1]["start_sec"] - processed[i]["end_sec"]
        if 0 <= gap < srt_gap_threshold:
            processed[i]["end_sec"] = processed[i + 1]["start_sec"]
        if processed[i]["end_sec"] > processed[i + 1]["start_sec"]:
            processed[i]["end_sec"] = processed[i + 1]["start_sec"]

    return processed

def polish_subtitle_lines(client, segments, proofreading_level, custom_rules, model):
    """用 Flash 做「文字」校稿，但禁止改變字幕行數與斷行；時間軸完全沿用 Python 產生的結果。"""
    if not segments:
        return segments
    if proofreading_level == "基本":
        rule = "只修正明顯錯字、同音誤字、專有名詞拼寫與口吃造成的文字錯誤；不要改寫語氣。"
    elif proofreading_level == "通順":
        rule = "修正錯字與明顯口語贅詞，讓每行自然易讀，但不要大幅改寫或增刪資訊。"
    else:
        rule = "修正錯字並將用詞調整為較正式、自然的繁體中文，但不要大幅改寫或增刪資訊。"

    numbered = "\n".join(f"{i+1}. {seg['text']}" for i, seg in enumerate(segments))
    prompt = (
        "你是影視字幕校稿器。請校對以下字幕。\n"
        f"校稿程度：{rule}\n"
        "硬性規則：\n"
        "1. 必須輸出 JSON array。\n"
        f"2. 必須剛好輸出 {len(segments)} 個字串，順序完全不變。\n"
        "3. 絕對不能合併、拆分、增加或刪除任何字幕行。\n"
        "4. 不要輸出編號、Markdown、說明文字。\n"
        "5. 保留英文產品名、型號、專有名詞的正確大小寫。\n"
        "6. 中文一律使用臺灣繁體中文；不要輸出簡體中文。\n"
        "7. 中文字幕不要加入標點符號。\n"
        "8. 如果不確定，保留原文，不要自行發明內容。\n"
        f"使用者自訂規則：\n{custom_rules.strip() if custom_rules else '無'}\n\n"
        "字幕如下：\n" + numbered
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        data = json.loads(response.text or "[]")
        if isinstance(data, list) and len(data) == len(segments) and all(isinstance(x, str) for x in data):
            for seg, text in zip(segments, data):
                cleaned = clean_subtitle_token(text)
                if cleaned:
                    seg["text"] = traditionalize_text(cleaned)
            return segments
    except Exception:
        # 校稿失敗時保留原始 ASR 結果，不影響精確時間軸。
        pass
    return segments


def sec_to_srt_time(sec):
    total_ms = max(0, int(round(sec * 1000)))
    h = total_ms // 3600000
    m = (total_ms % 3600000) // 60000
    s = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments):
    lines = []
    for idx, seg in enumerate(segments, 1):
        lines.append(
            f"{idx}\n{sec_to_srt_time(seg['start_sec'])} --> "
            f"{sec_to_srt_time(seg['end_sec'])}\n{seg['text']}\n"
        )
    return "\n".join(lines)


def apply_simple_subtitle_rules(text, custom_rules):
    """只做安全的字串替換，不改變字幕行數/時間軸。"""
    result = text
    if not custom_rules:
        return result
    # 目前保留使用者常見的明確替換規則；複雜語意校稿不在 timestamp 模式直接改字，
    # 避免修改文字後與 word timestamp 失去對齊。
    replacements = {
        "妳": "你",
        "其它": "其他",
    }
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def sec_to_human(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"

# -----------------------------
# API Key / 主介面
# -----------------------------
if not api_key:
    st.warning(
        "👈 請先在左側欄位輸入您的 **Gemini API Key** 才能解鎖完整功能。"
    )
else:
    client = genai.Client(api_key=api_key)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("🎧 音訊輸入來源")
        input_tab1, input_tab2 = st.tabs(["🎙️ 現場錄音", "📂 上傳音訊檔"])

        audio_bytes = None
        audio_name = "recording.wav"
        audio_mime_type = "audio/wav"
        audio_format = "wav"
        is_live_recording = False

        with input_tab1:
            audio_file = st.audio_input("點擊下方麥克風圖示開始錄音")
            if audio_file is not None:
                audio_bytes = audio_file.read()
                audio_name = "recording.wav"
                audio_mime_type, audio_format = guess_mime_and_format(audio_name, True)
                is_live_recording = True
                r_col1, r_col2 = st.columns([2, 1], vertical_alignment="center")
                with r_col1:
                    st.success("✅ 錄音完成！")
                with r_col2:
                    st.download_button(
                        label="💾 下載錄音檔",
                        data=audio_bytes,
                        file_name="recording.wav",
                        mime="audio/wav",
                        use_container_width=True,
                    )

        with input_tab2:
            uploaded_file = st.file_uploader(
                "上傳音訊檔", type=["mp3", "wav", "m4a", "aac", "ogg", "flac", "aiff", "opus", "webm"]
            )
            if uploaded_file is not None:
                audio_bytes = uploaded_file.read()
                audio_name = uploaded_file.name
                audio_mime_type, audio_format = guess_mime_and_format(audio_name)
                is_live_recording = False
                st.success(f"✅ 已成功載入音訊檔：{uploaded_file.name}")

        sample_rules = (
            "- 影片/音檔背景與語言：本音檔為繁體中文，夾雜少量英文科技名詞。講者為台灣人，請使用台灣慣用語與在地化譯名。\n"
            "- 語氣與贅字過濾：請徹底過濾口語贅字（如：然後、呃、就是說）。\n"
            "- 專有名詞與人名/地名對照：例如：「小明」不要寫成「小名」、「Kimi」不要寫成「奇米」、「汐止」不要寫成「西紙」。\n"
            "- 用詞替換與錯別字糾正：例如：將所有「妳」統一改為「你」、「其它」替換為「其他」、副詞統一使用「地」取代「的」。\n"
            "- 其他自訂規則：這是一部關於科技產品開箱的影片，請確保所有英文縮寫與型號（如 OLED、RTX 5090）維持大寫。"
        )

        with st.expander("🛠️ 自訂規則與詞彙表", expanded=False):
            st.caption("在此輸入專有名詞、特定錯別字糾正規則或背景說明，AI 進行任何處理時皆會納入考量。")
            load_sample = st.checkbox("📝 載入參考範例（勾選後自動帶入下方）", value=False)
            initial_rules_text = sample_rules if load_sample else ""
            custom_rules_input = st.text_area(
                "自訂規則內容",
                value=initial_rules_text,
                height=160,
                placeholder="若有特殊專有名詞、錯別字校正或背景說明，可在此輸入..."
            )

        transcribe_btn = False
        if audio_bytes is not None:
            st.markdown("---")
            transcribe_btn = st.button("✨ 開始 AI 處理", type="primary", use_container_width=True)

    with col2:
        st.subheader("📝 輸出結果")
        if "editable_text" not in st.session_state:
            st.session_state["editable_text"] = ""
        if "srt_data" not in st.session_state:
            st.session_state["srt_data"] = ""

        if audio_bytes is not None and transcribe_btn:
            # ---------------------------------------------------------
            # 影視字幕：專用 Transcribe + word timestamps + Python SRT
            # ---------------------------------------------------------
            if task_mode == "逐字稿" and output_mode == "影視字幕":
                with st.spinner("🎬 正在建立精確時間軸…"):
                    try:
                        progress = st.progress(0)
                        status = st.empty()
                        words, chunk_count, fallback_used = transcribe_with_timestamps(
                            client,
                            audio_bytes,
                            audio_format,
                            progress_bar=progress,
                            status_box=status,
                        )
                        status.empty()
                        progress.empty()

                        if not words:
                            raise RuntimeError("Transcribe 沒有回傳可用的 word-level timestamps。")

                        segments = build_subtitle_segments(
                            words,
                            client=client,
                            max_chars=subtitle_max_chars,
                            model=selected_model,
                        )
                        # 先做確定性的字串修正，再讓 Flash「只校稿、不改行數」。
                        for seg in segments:
                            seg["text"] = apply_simple_subtitle_rules(seg["text"], custom_rules_input)
                        segments = polish_subtitle_lines(
                            client, segments, proofreading_level, custom_rules_input, selected_model
                        )

                        srt_data = segments_to_srt(segments)
                        clean_text = "\n".join(seg["text"] for seg in segments)
                        st.session_state["editable_text"] = clean_text
                        st.session_state["srt_data"] = srt_data
                        st.session_state["match_ptr"] = 0

                        if fallback_used:
                            st.warning("⚠️ 某個音訊片段沒有取得 word timestamp，已使用保底文字結果；該片段時間軸可能需要人工確認。")
                        else:
                            st.success(
                                f"✅ 完成：使用 {TRANSCRIBE_MODEL}，共 {chunk_count} 段，"
                                f"以 word-level timestamps 建立 SRT。"
                            )
                    except Exception as e:
                        st.error(f"影視字幕處理失敗：{e}")

            # ---------------------------------------------------------
            # 其他模式：維持原本 Gemini Flash 工作流程
            # ---------------------------------------------------------
            else:
                if task_mode == "逐字稿":
                    if output_mode == "文章段落":
                        format_instruction = "3. 排版風格：請以完整的文章段落呈現，不要過度頻繁斷行或條列，就像在寫一篇流暢的文章一樣。"
                    else:  # 對話分行
                        format_instruction = (
                            "3. 排版風格（對話分行模式）：\n"
                            " - 請以清晰的對話或語句段落分行為主。\n"
                            " - 強制斷行要求：每當語氣轉換、意思告一段落、話題切換或不同句子交替時，必須經常換行。\n"
                            " - 請加上適當且正確的標點符號。"
                        )
                    task_instruction = (
                        "請將這段語音轉錄成文字。要求：\n"
                        "1. 【絕對禁令】：絕對不要輸出任何前言、後記、寒暄、說明，只能輸出轉錄結果本身！\n"
                        "2. 自動略過無意義的口語贅字、助詞與重複語句。\n"
                        "3. 完美支援並對應中文、英文、日文、韓文或其他語言夾雜。\n"
                        f"{format_instruction}"
                    )
                elif task_mode == "會議記錄與摘要":
                    if output_mode == "文章段落":
                        format_instruction = "排版要求：請以流暢的敘述性文章段落來撰寫會議摘要與記錄。"
                    else:
                        format_instruction = "排版要求：請使用結構化大綱（善用主標題與子標題）將會議的主旨、討論焦點與最終決議清晰區隔。"
                    task_instruction = f"請分析這段語音內容，產出專業的會議記錄與核心摘要。\n{format_instruction}"
                else:
                    if output_mode == "項目符號 (Bullet points)":
                        format_instruction = "排版要求：請以清晰的項目符號 (Bullet points) 條列出具體待辦事項。"
                    elif output_mode == "數字序號 (1, 2, 3...)":
                        format_instruction = "排版要求：請使用數字序號 (1, 2, 3...) 依序排列需要執行的任務與優先順序。"
                    else:
                        format_instruction = "排版要求：請以步驟流程（可使用箭頭符號 ➔ 串接）來呈現各個步驟的前後順序與執行細節。"
                    task_instruction = f"請分析這段語音內容，萃取出所有需要執行的行動清單 (Action Items)。\n{format_instruction}"

                if proofreading_level == "基本":
                    proofreading_instruction = "\n智慧校稿規則（基本）：僅將語音轉錄中的停頓、口吃、講錯之處進行修正，其餘大致保持原本原汁原味的口語內容。"
                elif proofreading_level == "通順":
                    proofreading_instruction = "\n智慧校稿規則（通順）：請修飾話中過多的重複用詞與口語習慣，並注意句子間的邏輯與連接詞使語意更明確。"
                else:
                    proofreading_instruction = "\n智慧校稿規則（正式）：此為正式演講或書面發表水準。除了具備「通順」的邏輯與語意優化外，請將過度口語、輕鬆的表達方式替換為適合正式場合的用詞。"

                emoji_instruction = ""
                if use_emoji:
                    if emoji_level == "少量":
                        emoji_instruction = "\nEmoji 規則：請在關鍵段落或句尾適度點綴少量（1-2個）相符的 Emoji。"
                    elif emoji_level == "中等":
                        emoji_instruction = "\nEmoji 規則：請在適當的句子、情緒轉折或重點項目旁加入 Emoji。"
                    else:
                        emoji_instruction = "\nEmoji 規則：請充分且豐富地在各段落與情緒處加入 Emoji。"

                custom_rules_section = ""
                if custom_rules_input.strip():
                    custom_rules_section = f"\n\n【使用者自訂規則與詞彙表（請務必嚴格遵守以下所有條件）】:\n{custom_rules_input.strip()}"
                final_prompt = task_instruction + proofreading_instruction + emoji_instruction + custom_rules_section

                with st.spinner("🎧 AI 處理中..."):
                    try:
                        response = client.models.generate_content(
                            model=selected_model,
                            contents=[
                                final_prompt,
                                types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime_type),
                            ],
                        )
                        st.session_state["editable_text"] = response.text or ""
                        st.session_state["srt_data"] = ""
                        st.session_state["match_ptr"] = 0
                    except Exception as e:
                        st.error(f"發生錯誤：{e}")
        # 尋找與取代工具
        with st.expander("🔍 尋找與取代", expanded=False):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                find_text = st.text_input("尋找字串", key="find_input")
            with f_col2:
                replace_text = st.text_input("取代為", key="replace_input")

            current_text = st.session_state["editable_text"]
            if find_text and find_text in current_text:
                matches = [m.start() for m in re.finditer(re.escape(find_text), current_text)]
                total_matches = len(matches)

                if "match_ptr" not in st.session_state:
                    st.session_state["match_ptr"] = 0
                
                if st.session_state["match_ptr"] >= total_matches:
                    st.session_state["match_ptr"] = 0

                ptr = st.session_state["match_ptr"]
                
                start_idx = matches[ptr]
                snippet_start = max(0, start_idx - 15)
                snippet_end = min(len(current_text), start_idx + len(find_text) + 15)
                snippet = current_text[snippet_start:snippet_end].replace("\n", " ")
                
                st.caption(f"找到 {total_matches} 個符合（目前第 {ptr + 1} 個）｜ 📍 前後文預覽：`...{snippet}...`")

                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    if st.button("⬅️ 上一個", use_container_width=True):
                        st.session_state["match_ptr"] = (ptr - 1) % total_matches
                        st.rerun()
                with b2:
                    if st.button("➡️ 下一個", use_container_width=True):
                        st.session_state["match_ptr"] = (ptr + 1) % total_matches
                        st.rerun()
                with b3:
                    if st.button("✨ 取代", use_container_width=True):
                        end_idx = start_idx + len(find_text)
                        st.session_state["editable_text"] = current_text[:start_idx] + replace_text + current_text[end_idx:]
                        st.success("已取代目前項目！")
                        st.rerun()
                with b4:
                    if st.button("💥 全部取代", use_container_width=True):
                        st.session_state["editable_text"] = current_text.replace(find_text, replace_text)
                        st.success("已全部取代！")
                        st.rerun()
            elif find_text:
                st.caption("找不到符合的字串")

        edited_text = st.text_area(
            "處理結果",
            value=st.session_state["editable_text"],
            height=320,
        )
        st.session_state["editable_text"] = edited_text

        # 下載按鈕區
        if st.session_state["editable_text"]:
            if task_mode == "逐字稿" and output_mode == "影視字幕" and st.session_state["srt_data"]:
                b_col1, b_col2, b_col3 = st.columns([1, 1, 1])
                with b_col1:
                    st.caption(f"目前字數：{len(st.session_state['editable_text'])} 字")
                with b_col2:
                    st.download_button(
                        label="💾 下載純文字 (.txt)",
                        data=st.session_state["editable_text"],
                        file_name="subtitle_text.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with b_col3:
                    st.download_button(
                        label="🎞️ 下載字幕檔 (.srt)",
                        data=st.session_state["srt_data"],
                        file_name="subtitles.srt",
                        mime="application/x-subrip",
                        use_container_width=True,
                    )
            else:
                b_col1, b_col2 = st.columns([1, 1])
                with b_col1:
                    st.caption(f"目前字數：{len(st.session_state['editable_text'])} 字")
                with b_col2:
                    st.download_button(
                        label="💾 下載文字檔 (.txt)",
                        data=st.session_state["editable_text"],
                        file_name="ai_transcript_result.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )