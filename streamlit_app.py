import streamlit as st
import os
import asyncio
import shutil
import json
import time
from pro_dubbing_engine import ProDubbingEngine
import tempfile

st.set_page_config(page_title="Pro Dubbing Engine Upgrade", page_icon="🎙️", layout="wide")

st.title("🎙️ Pro Dubbing Engine - Advanced Upgrade")
st.markdown("---")

# Try to get API keys from secrets first
secret_api_keys = st.secrets.get("GEMINI_API_KEYS", [])
if not secret_api_keys:
    # Fallback to single key if exists
    single_key = st.secrets.get("GEMINI_API_KEY", "")
    if single_key:
        secret_api_keys = [single_key]

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    if secret_api_keys:
        st.success(f"✅ {len(secret_api_keys)} API Keys loaded from Secrets")
        api_keys_input = ",".join(secret_api_keys)
    else:
        api_keys_input = st.text_area("Gemini API Keys (Comma separated)", help="Paste multiple API keys separated by commas for rotation support.")
    
    api_keys = [k.strip() for k in api_keys_input.split(",") if k.strip()]
    
    st.info("💡 Multi-API Support: Using multiple keys helps avoid rate limits during iterative rewriting.")

    # Sidebar Status Box
    st.divider()
    st.subheader("📊 Worker Status Log")
    status_container = st.empty()
    
    # Initialize worker statuses in session state
    if 'worker_statuses' not in st.session_state:
        st.session_state.worker_statuses = {}

    def update_status(worker_id, message):
        st.session_state.worker_statuses[worker_id] = message
        # Format the statuses for display
        status_text = ""
        for wid in sorted(st.session_state.worker_statuses.keys()):
            status_text += f"**Worker {wid}**: {st.session_state.worker_statuses[wid]}\n\n"
        status_container.markdown(status_text)

# Initialize engine
engine = ProDubbingEngine(api_keys=api_keys if api_keys else [])

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
                # Start Timer
                start_time = time.time()
                timer_placeholder = st.empty()
                
                with st.spinner("Processing..."):
                    # Use a consistent event loop approach for Streamlit
                    async def main_workflow():
                        final_srt = script_content
                        if "[00:" in script_content and "-->" not in script_content:
                            final_srt = await engine.text_to_srt_with_ai(script_content)
                        
                        segments = engine.parse_srt(final_srt)
                        
                        # Reset worker statuses
                        st.session_state.worker_statuses = {i+1: "Idle" for i in range(num_chunks)}
                        update_status(0, "") # Trigger initial display

                        with tempfile.TemporaryDirectory() as tmp_dir:
                            # Custom callback to update UI
                            def ui_callback(worker_id, msg):
                                update_status(worker_id, msg)

                            # Run the process and update timer periodically
                            async def run_process():
                                task = asyncio.create_task(engine.process_workflow_parallel(segments, num_chunks, tmp_dir, status_callback=ui_callback))
                                
                                while not task.done():
                                    elapsed = time.time() - start_time
                                    timer_placeholder.markdown(f"### ⏱️ Running Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
                                    await asyncio.sleep(1)
                                
                                return await task

                            await run_process()
                            
                            # Calculate results for display
                            successful = sum(1 for s in segments if s.status == "tts_generated_adjusted")
                            results = {
                                "total": len(segments),
                                "successful": successful,
                                "segments": [{"id": s.segment_id, "start": s.start, "end": s.end, "text": s.text, "status": s.status} for s in segments]
                            }
                            st.session_state.results = results
                            
                            # Final timer update
                            elapsed = time.time() - start_time
                            timer_placeholder.markdown(f"### ✅ Total Process Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
                            st.session_state.final_srt = final_srt
                            st.session_state.segments = segments

                            # Merge audio and generate SRT content
                            merged_audio_path = os.path.join(tmp_dir, "dubbed_audio.mp3")
                            if engine.merge_audio_files(segments, merged_audio_path):
                                with open(merged_audio_path, "rb") as f:
                                    st.session_state.merged_audio_data = f.read()
                            else:
                                st.session_state.merged_audio_data = None

                            st.session_state.generated_srt_content = engine.generate_srt_content(segments)
                            st.success("✅ Dubbing process completed!")

                    # Execute the async workflow
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    if loop.is_running():
                        # In some Streamlit environments, a loop might already be running
                        import nest_asyncio
                        nest_asyncio.apply()
                        loop.run_until_complete(main_workflow())
                    else:
                        loop.run_until_complete(main_workflow())
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
            if st.session_state.merged_audio_data:
                st.download_button(
                    label="⬇️ Download Audio (MP3)",
                    data=st.session_state.merged_audio_data,
                    file_name="dubbed_audio.mp3",
                    mime="audio/mpeg"
                )
            else:
                st.error("❌ Failed to merge audio files.")
        
        elif output_format == "📄 SRT + Audio (Separate Files)":
            col_srt, col_audio = st.columns(2)
            
            with col_srt:
                st.write("**📄 Subtitle File**")
                if st.session_state.generated_srt_content:
                    st.download_button(
                        label="⬇️ Download SRT",
                        data=st.session_state.generated_srt_content,
                        file_name="dubbed_subtitles.srt",
                        mime="text/plain"
                    )
                else:
                    st.error("❌ Failed to generate SRT content.")
            
            with col_audio:
                st.write("**🎵 Audio File**")
                if st.session_state.merged_audio_data:
                    st.download_button(
                        label="⬇️ Download Audio (MP3)",
                        data=st.session_state.merged_audio_data,
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
            # Fixed status color mapping: tts_generated_adjusted is the successful state in this engine
            status_color = "🟢" if s['status'] in ['tts_generated', 'tts_generated_adjusted'] else "🟡" if 'pending' in s['status'] else "🔴"
            st.text(f"{status_color} [{s['start']:.2f}s - {s['end']:.2f}s] | {s['text'][:50]}...")
