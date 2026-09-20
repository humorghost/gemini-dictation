from google import genai
from google.genai import types
import streamlit as st
import subprocess
import sys
import os
import re
from datetime import timedelta

# 1.【最重要】st.set_page_config 必須擺在所有 streamlit 指令的最前面！
st.set_page_config(
    page_title="Gemini 聽打小幫手", page_icon="🎙️", layout="wide"
)

# 嘗試自動從環境變數讀取 API Key
default_key = os.environ.get("GEMINI_API_KEY", "")

# 注入自訂 CSS 樣式（包含移除標題 hover 的迴紋針錨點圖示）
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
    /* 移除所有標題 hover 時出現的迴紋針連結圖示 */
    .stHeader a, h1 a, h2 a, h3 a {
        display: none !important;
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
        is_live_recording = False

        with input_tab1:
            audio_file = st.audio_input(
                "點擊下方麥克風圖示開始錄音"
            )
            if audio_file is not None:
                audio_bytes = audio_file.read()
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
                "上傳音訊檔", type=["mp3", "wav", "m4a", "aac"]
            )
            if uploaded_file is not None:
                audio_bytes = uploaded_file.read()
                is_live_recording = False
                st.success(f"✅ 已成功載入音訊檔：{uploaded_file.name}")

        # 🧠 永遠存在的自訂規則與詞彙表
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
            transcribe_btn = st.button(
                "✨ 開始 AI 處理", type="primary", use_container_width=True
            )

    with col2:
        st.subheader("📝 輸出結果")

        if "editable_text" not in st.session_state:
            st.session_state["editable_text"] = ""
        if "srt_data" not in st.session_state:
            st.session_state["srt_data"] = ""

        if audio_bytes is not None and transcribe_btn:
            if task_mode == "逐字稿":
                if output_mode == "文章段落":
                    format_instruction = "3. 排版風格：請以完整的文章段落呈現，不要過度頻繁斷行或條列，就像在寫一篇流暢的文章一樣。"
                elif output_mode == "影視字幕":
                    format_instruction = (
                        "3. 排版風格（【極致短句】影視字幕模式）：\n"
                        "⚠️ 這是非常嚴格的影視字幕任務，請徹底拋棄寫文章的習慣！\n"
                        " - 【絕對禁令】：絕對不要輸出任何前言、後記、寒暄、說明或「以下為您整理...」等廢話，只能輸出轉錄結果本身！\n"
                        " - 【標點符號禁令】：中文、日文、韓文等非表音文字，**絕對不要加上任何標點符號**（如逗號、句號、問號等），讓文字乾淨俐落呈現！\n"
                        " - 【極短斷行鐵律】：請根據語意、講者呼吸換氣與語氣停頓，**自行決定並合理切分短句斷行**！\n"
                        " - 格式請嚴格對齊：[分:秒,分:秒] 短句內容 或 [秒數,秒數] 短句內容\n"
                        " - 正確範例示範（全中文無標點斷行）：\n"
                        "   [00:06.500,00:07.180] 我們今天\n"
                        "   [00:07.180,00:08.180] 來這裡\n"
                        "   [00:08.180,00:09.500] 想要跟大家\n"
                        "   [00:09.500,00:10.700] 介紹這個"
                    )
                else:  # 對話分行
                    format_instruction = "3. 排版風格：自動加上適當的標點符號與對話分行，呈現結構清晰的對話排版。"
                
                task_instruction = (
                    "請將這段語音轉錄成文字。要求：\n"
                    "1. 【絕對禁令】：絕對不要輸出任何前言、後記、寒暄、說明或「以下為您整理...」等廢話，只能輸出轉錄結果本身！\n"
                    "2. 自動略過無意義的口語贅字、助詞與重複語句。\n"
                    "3. 完美支援並對應中文、英文、日文、韓文或其他語言夾雜。\n"
                    f"{format_instruction}"
                )
            elif task_mode == "會議記錄與摘要":
                if output_mode == "文章段落":
                    format_instruction = "排版要求：請以流暢的敘述性文章段落來撰寫會議摘要與記錄。"
                else:
                    format_instruction = "排版要求：請使用結構化大綱（善用主標題與子標題）將會議的主旨、討論焦點與最終決議清晰區隔。"

                task_instruction = (
                    "請分析這段語音內容，產出專業的會議記錄與核心摘要。\n"
                    f"{format_instruction}"
                )
            else:  # 行動清單
                if output_mode == "項目符號 (Bullet points)":
                    format_instruction = "排版要求：請以清晰的項目符號 (Bullet points) 條列出具體待辦事項。"
                elif output_mode == "數字序號 (1, 2, 3...)":
                    format_instruction = "排版要求：請使用數字序號 (1, 2, 3...) 依序排列需要執行的任務與優先順序。"
                else:
                    format_instruction = "排版要求：請以步驟流程（可使用箭頭符號 ➔ 串接）來呈現各個步驟的前後順序與執行細節。"

                task_instruction = (
                    "請分析這段語音內容，萃取出所有需要執行的行動清單 (Action Items)。\n"
                    f"{format_instruction}"
                )

            if proofreading_level == "基本":
                proofreading_instruction = (
                    "\n智慧校稿規則（基本）：僅將語音轉錄中的停頓、口吃、講錯之處進行修正，"
                    "其餘大致保持原本原汁原味的口語內容。"
                )
            elif proofreading_level == "通順":
                proofreading_instruction = (
                    "\n智慧校稿規則（通順）：請修飾話中過多的重複用詞與口語習慣，"
                    "並注意句子間的邏輯與連接詞使語意更明確。"
                )
            else:
                proofreading_instruction = (
                    "\n智慧校稿規則（正式）：此為正式演講或書面發表水準。除了具備「通順」的邏輯與語意優化外，"
                    "請將過度口語、輕鬆的表達方式替換為適合正式場合的用詞。"
                )

            emoji_instruction = ""
            if use_emoji and not (task_mode == "逐字稿" and output_mode == "影視字幕"):
                if emoji_level == "少數":
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
                            types.Part.from_bytes(
                                data=audio_bytes, mime_type="audio/wav"
                            ),
                        ],
                    )
                    raw_response_text = response.text

                    if task_mode == "逐字稿" and output_mode == "影視字幕":
                        pattern = re.compile(r'\[\s*([\d:\.]+)\s*,\s*([\d:\.]+)\s*\]\s*(.*)')
                        parsed_subtitles = []
                        
                        def parse_time_to_sec(t_str):
                            t_str = t_str.strip()
                            if ':' in t_str:
                                parts = t_str.split(':')
                                if len(parts) == 2:
                                    return float(parts[0]) * 60 + float(parts[1])
                                elif len(parts) == 3:
                                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                            return float(t_str)

                        for line in raw_response_text.splitlines():
                            match = pattern.search(line)
                            if match:
                                start_sec = parse_time_to_sec(match.group(1))
                                end_sec = parse_time_to_sec(match.group(2))
                                text = match.group(3).strip()
                                parsed_subtitles.append({
                                    "start_sec": start_sec,
                                    "end_sec": end_sec,
                                    "text": text
                                })
                            else:
                                cleaned_line = re.sub(r'\[.*?\]', '', line).strip()
                                if cleaned_line:
                                    parsed_subtitles.append({
                                        "start_sec": 0.0,
                                        "end_sec": 2.0,
                                        "text": cleaned_line
                                    })
                        
                        if parsed_subtitles:
                            def sec_to_srt_time(sec):
                                total_ms = int(sec * 1000)
                                if total_ms < 0: total_ms = 0
                                h = total_ms // 3600000
                                m = (total_ms % 3600000) // 60000
                                s = (total_ms % 60000) // 1000
                                ms = total_ms % 1000
                                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

                            start_offset = srt_start_offset
                            end_offset = srt_end_offset
                            gap_threshold = srt_gap_threshold

                            processed_subs = []
                            for i, sub in enumerate(parsed_subtitles):
                                s_sec = max(0.0, sub["start_sec"] - start_offset)
                                e_sec = sub["end_sec"] + end_offset

                                if i > 0 and gap_threshold > 0:
                                    prev_e = processed_subs[-1]["end_sec"]
                                    gap = s_sec - prev_e
                                    if 0 <= gap < gap_threshold:
                                        processed_subs[-1]["end_sec"] = s_sec

                                processed_subs.append({
                                    "start_sec": s_sec,
                                    "end_sec": e_sec,
                                    "text": sub["text"]
                                })

                            # 防重疊機制
                            for i in range(len(processed_subs) - 1):
                                if processed_subs[i]["end_sec"] > processed_subs[i + 1]["start_sec"]:
                                    processed_subs[i]["end_sec"] = processed_subs[i + 1]["start_sec"]

                            clean_text_lines = []
                            srt_lines = []
                            for idx, sub in enumerate(processed_subs, 1):
                                clean_text_lines.append(sub["text"])
                                s_str = sec_to_srt_time(sub["start_sec"])
                                e_str = sec_to_srt_time(sub["end_sec"])
                                srt_lines.append(f"{idx}\n{s_str} --> {e_str}\n{sub['text']}\n")

                            st.session_state["editable_text"] = "\n".join(clean_text_lines)
                            st.session_state["srt_data"] = "\n".join(srt_lines)
                        else:
                            st.session_state["editable_text"] = raw_response_text
                            st.session_state["srt_data"] = ""
                    else:
                        st.session_state["editable_text"] = raw_response_text
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

        # 提示與警告（精簡版）
        if task_mode == "逐字稿" and output_mode == "影視字幕" and st.session_state["editable_text"]:
            st.warning(
                "⚠️ **影視字幕模式提醒**\n\n"
                "* **編輯限制**：可改錯字，**請勿增減行數或斷行**（會導致時間軸錯亂）。\n"
                "* **精準度限制**：Gemini 屬通用大模型，時間戳記無法像 Whisper 等專用工具那樣精準。\n"
                "* **建議作法**：利用本工具快速產出初版字幕，下載 SRT 檔後再用專業字幕軟體調整。"
            )

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