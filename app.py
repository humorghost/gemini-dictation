from google import genai
from google.genai import types
import streamlit as st
import subprocess
import sys
import os
import re

# 1.【最重要】st.set_page_config 必須擺在所有 Streamlit 指令的最前面！
st.set_page_config(
    page_title="Gemini 聽打小幫手", page_icon="🎙️", layout="wide"
)

# 嘗試自動從環境變數讀取 API Key
default_key = os.environ.get("GEMINI_API_KEY", "")

# 注入自訂 CSS 樣式
st.markdown(
    """
    <style>
    /* 放大整體介面文字字級（標題除外） */
    html, body, [class*="css"] {
        font-size: 16px;
    }
    .stTextArea textarea {
        font-size: 16px !important;
    }
    .stTextInput input {
        font-size: 16px !important;
    }
    .stRadio label, .stCheckbox label, .stMarkdown p {
        font-size: 16px !important;
    }
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
    # 這裡會自動帶入環境變數中的 API Key，若沒有則保持空白讓使用者手動輸入
    api_key = st.text_input(
        "Gemini API Key", 
        value=default_key, 
        type="password",
        placeholder="請輸入你的 API Key"
    )

    st.markdown("---")
    st.header("🧠 處理模式")
    task_mode = st.selectbox(
        "選擇任務類型：",
        [
            "逐字稿",
            "會議記錄與摘要",
            "重點清單",
        ],
    )

    if task_mode == "逐字稿":
        st.markdown("---")
        st.header("📐 排版風格")
        output_mode = st.radio(
            "選擇排版方式：",
            [
                "文章段落",
                "對話分行",
                "影視字幕",
            ],
        )
    else:
        output_mode = ""

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
            "使用頻率", options=["少數", "中等", "大量"], value="中等"
        )

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
        with input_tab1:
            audio_file = st.audio_input(
                "點擊下方麥克風圖示開始錄音"
            )
            if audio_file is not None:
                audio_bytes = audio_file.read()
                st.success("✅ 錄音完成！")

        with input_tab2:
            uploaded_file = st.file_uploader(
                "上傳音訊/影片檔", type=["mp3", "wav", "m4a", "aac", "mp4"]
            )
            if uploaded_file is not None:
                audio_bytes = uploaded_file.read()
                st.success(f"✅ 已成功載入檔案：{uploaded_file.name}")

        transcribe_btn = False
        if audio_bytes is not None:
            st.markdown("---")
            transcribe_btn = st.button(
                "✨ 開始 AI 處理", type="primary", use_container_width=True
            )

    with col2:
        st.subheader("📝 輸出結果")

        if "editable_text" not in st.session_state:
            st.session_state["editable_text"] = ""

        if audio_bytes is not None and transcribe_btn:
            if task_mode == "逐字稿":
                if "文章段落" in output_mode:
                    format_instruction = (
                        "3. 排版風格：請以完整的文章段落呈現，"
                        "不要過度頻繁斷行或條列，就像在寫一篇流暢的文章一樣。"
                    )
                elif "影視字幕" in output_mode:
                    format_instruction = (
                        "3. 排版風格：請模擬電影、戲劇或電視節目字幕的排版。要求：\n"
                        "    - 每一句或每一個短語獨立成一行（短行斷句）。\n"
                        "    - 幾乎省略所有標準標點符號（如逗號、句號）。\n"
                        "    - 僅在極少數表示強烈情緒或疑問時，才保留問號（？）或驚嘆號（！）。\n"
                        "    - 不要出現括號或引號，保持乾淨流暢的字幕風格。"
                    )
                else:
                    format_instruction = (
                        "3. 排版風格：自動加上適當的標點符號與段落分行，"
                        "呈現結構清晰的逐字稿/對話排版。"
                    )
                
                task_instruction = (
                    "請將這段語音轉錄成文字。要求：\n"
                    "1. 自動略過無意義的口語贅字、助詞與重複語句（如呃、然後、就是等）。\n"
                    "2. 完美支援並對應中文、英文、日文、韓文夾雜。\n"
                    f"{format_instruction}"
                )
            elif task_mode == "會議記錄與摘要":
                task_instruction = (
                    "請分析這段語音內容，產出結構清晰且專業的「會議記錄與核心摘要」，"
                    "包含討論主旨、重要決議與關鍵細節。"
                )
            else:
                task_instruction = (
                    "請分析這段語音內容，萃取出所有需要執行的「重點行動清單 (Action Items)」，"
                    "以條列式清楚列出具體待辦事項與相關細節。"
                )

            # 智慧校稿指令組裝
            if proofreading_level == "基本":
                proofreading_instruction = (
                    "\n智慧校稿規則（基本）：僅將語音轉錄中的停頓、口吃、講錯之處進行修正，"
                    "其餘大致保持原本原汁原味的口語內容。"
                )
            elif proofreading_level == "通順":
                proofreading_instruction = (
                    "\n智慧校稿規則（通順）：請修飾話中過多的重複用詞與口語習慣（如「然後...然後」、「可是...可是」等），"
                    "並注意句子間的邏輯與連接詞（適度加入或刪除因為、所以、結果等）使語意更明確。"
                    "若主詞或受詞不明確請自動補上，在不大量修改原意的前提下，保持原意讓句子更通順流暢。"
                )
            else:  # 正式
                proofreading_instruction = (
                    "\n智慧校稿規則（正式）：此為正式演講或書面發表水準。除了具備「通順」的邏輯與語意優化外，"
                    "請將過度口語、輕鬆的表達方式替換為適合正式場合的用詞（不需到僵硬的公文或法律條文程度）。"
                )

            emoji_instruction = ""
            if use_emoji:
                if emoji_level == "少數":
                    emoji_instruction = (
                        "\nEmoji 規則：請在關鍵段落或句尾適度點綴少量（1-2個）相符的"
                        " Emoji，保持專業簡潔。"
                    )
                elif emoji_level == "中等":
                    emoji_instruction = (
                        "\nEmoji 規則：請在適當的句子、情緒轉折或重點項目旁加入"
                        " Emoji，讓內容生動易讀。"
                    )
                else:
                    emoji_instruction = (
                        "\nEmoji 規則：請充分且豐富地在各段落與情緒處加入"
                        " Emoji，讓整篇文字充滿活力與趣味。"
                    )

            final_prompt = task_instruction + proofreading_instruction + emoji_instruction

            with st.spinner("🎧 AI 正在處理中..."):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            final_prompt,
                            types.Part.from_bytes(
                                data=audio_bytes, mime_type="audio/wav"
                            ),
                        ],
                    )
                    st.session_state["editable_text"] = response.text
                    st.session_state["match_ptr"] = 0
                except Exception as e:
                    st.error(f"發生錯誤：{e}")

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

        if st.session_state["editable_text"]:
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