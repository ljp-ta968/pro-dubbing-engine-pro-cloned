import streamlit as st
import os
import asyncio
import json
from pro_dubbing_engine import ProDubbingEngine
import tempfile

st.set_page_config(page_title="Pro Dubbing Engine Upgrade", page_icon="🎙️", layout="wide")

st.title("🎙️ Pro Dubbing Engine - Advanced Upgrade")
st.markdown("---")

# Try to get API key from secrets first
secret_api_key = st.secrets.get("GEMINI_API_KEY", "")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    if secret_api_key:
        st.success("✅ API Key loaded from Secrets")
        api_key = secret_api_key
    else:
        api_key = st.text_input("Gemini API Key (Optional)", type="password", help="Add GEMINI_API_KEY to Streamlit Secrets for permanent access")
    
    st.info("Upgrade: Now supports Male/Female voice selection and separate audio/SRT downloads!")

# Initialize engine
engine = ProDubbingEngine(api_key=api_key if api_key else None)

tab1, tab2 = st.tabs(["📤 Input & Process", "📊 Analytics"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Provide Input")
        input_type = st.radio("Select Input Type:", ["Text Input", "File Upload (.srt / .txt)"])
        
        script_content = ""
        if input_type == "Text Input":
            script_content = st.text_area("Paste your script with timestamps here:", height=300, 
                                        placeholder="[00:00:00] Hello...\n[00:00:02] Welcome...")
        else:
            uploaded_file = st.file_uploader("Upload .srt or .txt file", type=["srt", "txt"])
            if uploaded_file:
                script_content = uploaded_file.read().decode("utf-8")
                st.success(f"File '{uploaded_file.name}' loaded!")

    with col2:
        st.subheader("2. Process & Dub")
        
        # Parse segments
        segments = []
        if script_content:
            if "[00:" in script_content and "-->" not in script_content:
                srt_temp = engine._simple_text_to_srt(script_content)
                segments = engine.parse_srt(srt_temp)
            else:
                segments = engine.parse_srt(script_content)

        # Slider: Number of chunks
        is_disabled = len(segments) == 0
        max_chunks_limit = 10
        max_val = min(len(segments), max_chunks_limit) if not is_disabled else max_chunks_limit
        
        num_chunks = st.slider(
            "Select Number of Chunks (Parallel Workers):", 
            min_value=1, 
            max_value=max_val if max_val >= 1 else 10, 
            value=min(len(segments), 5) if not is_disabled else 5,
            disabled=is_disabled
        )

        # Language and Gender Selectors
        lang_col, gender_col = st.columns(2)
        
        lang_options = {
            "Myanmar (Burmese)": "my",
            "English": "en",
            "Japanese": "ja",
            "Korean": "ko",
            "Thai": "th",
            "Vietnamese": "vi"
        }
        
        with lang_col:
            selected_lang_name = st.selectbox("Select Output Language:", list(lang_options.keys()), index=0)
            engine.output_language = lang_options[selected_lang_name]
            
        with gender_col:
            selected_gender = st.selectbox("Select Voice Gender:", ["Male", "Female"], index=0)
            engine.voice_gender = selected_gender

        # Output Format Selection
        st.divider()
        st.subheader("3. Select Output Format")
        output_format = st.radio(
            "Choose what to download:",
            ["🎵 Audio File Only", "📄 SRT + Audio (Separate Files)"],
            horizontal=True
        )

        if not is_disabled:
            st.write(f"✅ Found **{len(segments)}** segments.")
            st.write(f"⚡ Mode: **{num_chunks} Workers** | Voice: **{selected_gender}**")
            
            if st.button("🚀 Start Parallel Professional Dubbing", use_container_width=True):
                with st.spinner("Processing..."):
                    final_srt = script_content
                    if "[00:" in script_content and "-->" not in script_content:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        final_srt = loop.run_until_complete(engine.text_to_srt_with_ai(script_content))
                    
                    segments = engine.parse_srt(final_srt)
                    chunks = engine.chunk_segments_by_count(segments, num_chunks)
                    
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        results = loop.run_until_complete(engine.process_workflow_parallel(chunks, tmp_dir))
                        
                        st.session_state.results = results
                        st.session_state.final_srt = final_srt
                        st.session_state.segments = segments
                        st.session_state.tmp_dir = tmp_dir
                        st.success("✅ Dubbing process completed!")
        else:
            st.warning("⚠️ Please provide input to enable dubbing.")

    if "results" in st.session_state:
        st.divider()
        st.subheader("🎧 Processing Results")
        res = st.session_state.results
        st.write(f"Total Segments: {res['total']} | Successful: {res['successful']}")
        with st.expander("View Detailed Segment Status"):
            st.table(res['segments'])

        # Download Section
        st.divider()
        st.subheader("📥 Download Results")
        
        segments_list = st.session_state.segments
        
        if output_format == "🎵 Audio File Only":
            st.info("💾 Merging audio files into a single file...")
            with tempfile.TemporaryDirectory() as download_tmp:
                merged_audio_path = os.path.join(download_tmp, "dubbed_audio.mp3")
                if engine.merge_audio_files(segments_list, merged_audio_path):
                    with open(merged_audio_path, "rb") as f:
                        audio_data = f.read()
                    st.download_button(
                        label="⬇️ Download Audio (MP3)",
                        data=audio_data,
                        file_name="dubbed_audio.mp3",
                        mime="audio/mpeg"
                    )
                else:
                    st.error("❌ Failed to merge audio files.")
        
        elif output_format == "📄 SRT + Audio (Separate Files)":
            col_srt, col_audio = st.columns(2)
            
            with col_srt:
                st.write("**📄 Subtitle File**")
                srt_content = engine.generate_srt_content(segments_list)
                st.download_button(
                    label="⬇️ Download SRT",
                    data=srt_content,
                    file_name="dubbed_subtitles.srt",
                    mime="text/plain"
                )
            
            with col_audio:
                st.write("**🎵 Audio File**")
                with tempfile.TemporaryDirectory() as download_tmp:
                    merged_audio_path = os.path.join(download_tmp, "dubbed_audio.mp3")
                    if engine.merge_audio_files(segments_list, merged_audio_path):
                        with open(merged_audio_path, "rb") as f:
                            audio_data = f.read()
                        st.download_button(
                            label="⬇️ Download Audio (MP3)",
                            data=audio_data,
                            file_name="dubbed_audio.mp3",
                            mime="audio/mpeg"
                        )
                    else:
                        st.error("❌ Failed to generate audio file.")

with tab2:
    st.subheader("📈 Technical Analytics")
    if "results" in st.session_state:
        res = st.session_state.results
        st.write("**Segment Timeline**")
        for s in res['segments']:
            status_color = "🟢" if s['status'] == 'tts_generated' else "🟡" if 'adjusted' in s['status'] else "🔴"
            st.text(f"{status_color} [{s['start']:.2f}s - {s['end']:.2f}s] | {s['text'][:50]}...")
