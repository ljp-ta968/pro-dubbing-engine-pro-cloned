"""
Professional Dubbing Engine - Upgraded Version
Handles SRT, TXT to SRT conversion, timestamp-aware chunking, parallel TTS generation, and duration validation.
Supports multiple voices (Male/Female), parallel workers, and audio merging.
"""

import re
import asyncio
import edge_tts
from typing import List, Dict, Tuple, Optional
import os
import json
import time
import datetime
from google import genai
from google.genai import types
from pydub import AudioSegment
import io
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class DubbingSegment:
    def __init__(self, start: float, end: float, lang: str, text: str, segment_id: int):
        self.start = start
        self.end = end
        self.duration = end - start
        self.lang = lang
        self.text = text
        self.segment_id = segment_id
        self.tts_audio_path = None
        self.tts_duration = None
        self.adjusted_text = text
        self.adjusted_speed = 1.0
        self.status = "pending"
        self.original_tts_duration = None
        self.final_audio_path = None
        self.retries = 0
        self.sentence_id = None # To group segments into sentences

class DubbingSentence:
    def __init__(self, segments: List[DubbingSegment], sentence_id: int):
        self.segments = segments
        self.sentence_id = sentence_id
        self.start = segments[0].start
        self.end = segments[-1].end
        self.duration = self.end - self.start
        self.text = " ".join([s.text for s in segments])
        self.adjusted_text = self.text
        self.tts_audio_path = None
        self.tts_duration = None
        self.retries = 0
        self.status = "pending"

class ProDubbingEngine:
    def __init__(self, api_keys: List[str] = None, output_language: str = "my", voice_gender: str = "Male"):
        self.tolerance = 0.3  # ±0.3 seconds
        self.api_keys = api_keys if api_keys else []
        self.output_language = output_language.lower()
        self.voice_gender = voice_gender
        self.current_key_index = 0
        self.api_lock = asyncio.Lock() # Lock for thread-safe/async-safe rotation
        
        # Rate limiting state: {key: [timestamp1, timestamp2, ...]}
        self.key_usage = {key: [] for key in self.api_keys}
        self.max_rpm = 9 # Max requests per minute per key
        
        self._initialize_voice_map()

    async def _get_next_client(self):
        """Rotate through API keys and return a configured GenAI client with rate limit awareness."""
        if not self.api_keys:
            return None, None
        
        while True:
            async with self.api_lock: # Ensure parallel tasks don't pick the same key simultaneously
                attempts = 0
                while attempts < len(self.api_keys):
                    key = self.api_keys[self.current_key_index]
                    self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                    attempts += 1
                    
                    now = time.time()
                    # Clean up old timestamps (older than 60s)
                    self.key_usage[key] = [t for t in self.key_usage[key] if now - t < 60]
                    
                    if len(self.key_usage[key]) < self.max_rpm:
                        # Key is available
                        self.key_usage[key].append(now)
                        client = genai.Client(api_key=key)
                        config = types.GenerateContentConfig(
                            max_output_tokens=65536,
                            temperature=0.7
                        )
                        return client, config
            
            # All keys are at limit, wait a bit and loop
            print("All API keys are at rate limit. Waiting 5 seconds...")
            await asyncio.sleep(5)

    def _initialize_voice_map(self):
        # Voice mapping with Male/Female options
        self.voice_map = {
            "my": {"Male": "my-MM-ThihaNeural", "Female": "my-MM-NilarNeural"},
            "en": {"Male": "en-US-GuyNeural", "Female": "en-US-AvaNeural"},
            "ja": {"Male": "ja-JP-KeitaNeural", "Female": "ja-JP-NanamiNeural"},
            "ko": {"Male": "ko-KR-InJoonNeural", "Female": "ko-KR-SunHiNeural"},
            "th": {"Male": "th-TH-NiwatNeural", "Female": "th-TH-PremwadeeNeural"},
            "vi": {"Male": "vi-VN-NamMinhNeural", "Female": "vi-VN-HoaiMyNeural"}
        }

    def _time_to_seconds(self, time_str: str) -> float:
        """Convert HH:MM:SS,ms or MM:SS to seconds"""
        time_str = time_str.replace(',', '.').strip('[] ')
        parts = time_str.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(time_str)

    def _seconds_to_time(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS,ms"""
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds_int = divmod(remainder, 60)
        milliseconds = int((seconds - total_seconds) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of an audio file using pydub."""
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # duration in seconds
        except Exception as e:
            print(f"Error getting audio duration for {audio_path}: {e}")
            return 0.0

    def _adjust_audio_speed(self, audio_path: str, target_duration: float, output_path: str) -> bool:
        """Adjust audio speed to match target duration."""
        try:
            audio = AudioSegment.from_file(audio_path)
            current_duration = len(audio) / 1000.0
            if current_duration == 0:
                return False
            
            speed_factor = current_duration / target_duration
            
            # Limit speed adjustment to reasonable range (0.5x to 2.0x)
            speed_factor = max(0.5, min(2.0, speed_factor))
            
            # Apply speed adjustment
            # pydub's speedup is a bit crude, but works for small adjustments
            adjusted_audio = audio.speedup(playback_speed=speed_factor)
            
            # Normalize volume
            adjusted_audio = adjusted_audio.normalize()

            adjusted_audio.export(output_path, format="mp3", bitrate="192k")
            return True
        except Exception as e:
            print(f"Error adjusting audio speed for {audio_path}: {e}")
            return False

    def parse_srt(self, srt_content: str) -> List[DubbingSegment]:
        """Parse SRT content into DubbingSegments"""
        segments = []
        pattern = r'(\d+)\s+(\d{2}:\d{2}:\d{2}[,. ]\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}[,. ]\d{3})\s+(.*?)(?=\n\n|\n\d+\n|$)'
        matches = re.finditer(pattern, srt_content, re.DOTALL)
        
        for i, match in enumerate(matches):
            start_s = self._time_to_seconds(match.group(2))
            end_s = self._time_to_seconds(match.group(3))
            text = match.group(4).replace('\n', ' ').strip()
            
            segments.append(DubbingSegment(
                start=start_s,
                end=end_s,
                lang=self.output_language,
                text=text,
                segment_id=i
            ))
        return segments

    def group_segments_into_sentences(self, segments: List[DubbingSegment]) -> List[DubbingSentence]:
        """Group segments based on sentence-ending punctuation."""
        sentences = []
        current_batch = []
        sentence_id = 1
        
        # Sentence ending markers for various languages
        end_markers = r'[.!?။၊၊]' # Includes Myanmar markers
        
        for seg in segments:
            current_batch.append(seg)
            # Check if the text ends with a sentence marker
            if re.search(end_markers + r'\s*$', seg.text) or seg == segments[-1]:
                sentence = DubbingSentence(current_batch, sentence_id)
                for s in current_batch:
                    s.sentence_id = sentence_id
                sentences.append(sentence)
                current_batch = []
                sentence_id += 1
                
        return sentences

    async def text_to_srt_with_ai(self, text: str) -> str:
        """Convert custom formatted text to standard SRT using Gemini AI"""
        client, config = await self._get_next_client()
        if not client:
            return self._simple_text_to_srt(text)

        prompt = f"""
        You are an expert subtitler. Convert the following timestamped text into a professional SRT subtitle format.
        
        INPUT FORMAT DESCRIPTION:
        The input contains timestamps in brackets like [HH:MM:SS] or similar, followed by text. 
        These timestamps usually indicate the start time of the dialogue.
        
        TASK:
        1. Convert these to standard SRT format (Index, Time Range, Text).
        2. Create precise time ranges (Start --> End). The 'End' time of a segment should generally be the 'Start' time of the next segment to ensure continuity, unless there's a natural long pause.
        3. Add milliseconds (e.g., ,070 or ,000) to make it look professional.
        4. Split the text into readable chunks that follow the natural flow of speech.
        5. If the input text spans multiple lines but belongs to one sentence, split it logically across SRT indices.
        
        INPUT TEXT:
        {text}
        
        OUTPUT ONLY THE VALID SRT CONTENT.
        """
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model='gemini-3-flash',
                contents=prompt,
                config=config
            )
            return response.text.strip()
        except Exception as e:
            print(f"AI SRT conversion failed: {e}")
            return self._simple_text_to_srt(text)

    async def _rewrite_text_with_ai(self, original_text: str, target_duration: float, current_tts_duration: float, lang: str) -> str:
        """Use Gemini AI to rewrite text to better fit target duration."""
        client, config = await self._get_next_client()
        if not client:
            return original_text

        duration_diff = current_tts_duration - target_duration
        if duration_diff > 0: # TTS is too long, need to shorten text
            prompt = f"""
            The following {lang} text was spoken in {current_tts_duration:.2f} seconds, but it needs to fit into {target_duration:.2f} seconds. 
            Please rewrite the text to be shorter, while retaining its original meaning as much as possible. 
            Do not add any introductory or concluding phrases. Just provide the rewritten text.
            Original text: {original_text}
            Rewritten text:
            """
        else: # TTS is too short, need to lengthen text
            prompt = f"""
            The following {lang} text was spoken in {current_tts_duration:.2f} seconds, but it needs to be {target_duration:.2f} seconds long. 
            Please rewrite the text to be slightly longer, adding natural pauses or descriptive words, while retaining its original meaning as much as possible. 
            Do not add any introductory or concluding phrases. Just provide the rewritten text.
            Original text: {original_text}
            Rewritten text:
            """
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model='gemini-3-flash',
                contents=prompt,
                config=config
            )
            rewritten_text = response.text.strip()
            # Basic cleanup of potential AI conversational filler
            if rewritten_text.lower().startswith("rewritten text:"):
                rewritten_text = rewritten_text[len("rewritten text:"):].strip()
            return rewritten_text
        except Exception as e:
            print(f"AI rewrite failed: {e}")
            return original_text

    def _simple_text_to_srt(self, text: str) -> str:
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        srt_out = []
        idx = 1
        for i in range(len(lines)):
            match = re.match(r'\[?(\d{2}:\d{2}:\d{2})\]?\s*(.*)', lines[i])
            if match:
                start_time = match.group(1) + ",000"
                content = match.group(2)
                end_time = self._add_seconds_to_time(match.group(1), 2) + ",000"
                srt_out.append(f"{idx}\n{start_time} --> {end_time}\n{content}\n")
                idx += 1
        return "\n".join(srt_out)

    def _add_seconds_to_time(self, time_str: str, seconds_to_add: int) -> str:
        try:
            t = datetime.datetime.strptime(time_str, "%H:%M:%S")
            t_new = t + datetime.timedelta(seconds=seconds_to_add)
            return t_new.strftime("%H:%M:%S")
        except: return time_str

    async def generate_tts_for_sentence(self, sentence: DubbingSentence, output_dir: str, status_callback=None) -> bool:
        """Generate TTS for a full sentence with iterative text rewriting and speed adjustment."""
        target_duration = sentence.duration
        sentence.retries = 0
        max_ai_retries = 50 

        while True:
            try:
                if status_callback:
                    status_callback(sentence.sentence_id, f"Processing Sentence (Attempt {sentence.retries + 1})")
                
                lang_voices = self.voice_map.get(self.output_language, self.voice_map["my"])
                voice = lang_voices.get(self.voice_gender, lang_voices["Male"])
                
                temp_output_path = os.path.join(output_dir, f"temp_sent_{sentence.sentence_id}.mp3")
                try:
                    communicate = edge_tts.Communicate(sentence.adjusted_text, voice)
                    await communicate.save(temp_output_path)
                except Exception as e:
                    print(f"Edge-TTS failed: {e}")
                    # If TTS fails, increment retry and wait
                    sentence.retries += 1
                    if sentence.retries >= max_ai_retries:
                        return False
                    await asyncio.sleep(2)
                    continue
                
                tts_duration = self._get_audio_duration(temp_output_path)
                sentence.tts_duration = tts_duration

                if tts_duration > 0 and (abs(tts_duration - target_duration) <= self.tolerance or sentence.retries >= max_ai_retries):
                    final_output_path = os.path.join(output_dir, f"sent_{sentence.sentence_id}.mp3")
                    if self._adjust_audio_speed(temp_output_path, target_duration, final_output_path):
                        sentence.tts_audio_path = final_output_path
                        sentence.status = "completed"
                        if status_callback: status_callback(sentence.sentence_id, "Completed")
                        
                        # Split the audio back into segments (simple proportional split for now)
                        self._split_sentence_audio_to_segments(sentence, final_output_path, output_dir)
                        
                        if os.path.exists(temp_output_path): os.remove(temp_output_path)
                        return True
                    else:
                        sentence.status = "error"
                        if os.path.exists(temp_output_path): os.remove(temp_output_path)
                        return False
                else:
                    new_text = await self._rewrite_text_with_ai(
                        sentence.adjusted_text, target_duration, tts_duration, self.output_language
                    )
                    # If AI fails to rewrite (returns same text), we still increment retry
                    sentence.adjusted_text = new_text
                    sentence.retries += 1
                    if os.path.exists(temp_output_path): os.remove(temp_output_path)
                    continue
            except Exception as e:
                sentence.status = f"error: {e}"
                if status_callback: status_callback(sentence.sentence_id, f"Error: {e}")
                sentence.retries += 1
                if sentence.retries >= max_ai_retries:
                    return False
                await asyncio.sleep(2) # Wait a bit before retrying on error
                continue

    def _split_sentence_audio_to_segments(self, sentence: DubbingSentence, audio_path: str, output_dir: str):
        """Split a sentence audio file back into its constituent segments based on original durations."""
        try:
            full_audio = AudioSegment.from_file(audio_path)
            total_original_duration = sum(s.duration for s in sentence.segments)
            
            current_pos = 0
            for seg in sentence.segments:
                # Calculate proportional duration in the generated audio
                seg_ratio = seg.duration / total_original_duration
                seg_audio_len = len(full_audio) * seg_ratio
                
                seg_audio = full_audio[current_pos : current_pos + int(seg_audio_len)]
                seg_path = os.path.join(output_dir, f"seg_{seg.segment_id}.mp3")
                seg_audio.export(seg_path, format="mp3")
                
                seg.tts_audio_path = seg_path
                seg.status = "tts_generated_adjusted"
                current_pos += int(seg_audio_len)
        except Exception as e:
            print(f"Error splitting sentence audio: {e}")

    async def process_sentence_chunk(self, sentences: List[DubbingSentence], output_dir: str, chunk_id: int, status_callback=None):
        """Process sentences in a chunk sequentially."""
        for sent in sentences:
            if status_callback:
                status_callback(chunk_id, f"Worker {chunk_id}: Processing Sentence {sent.sentence_id}")
            await self.generate_tts_for_sentence(sent, output_dir, status_callback=lambda sid, msg: status_callback(chunk_id, f"Worker {chunk_id}: Sent {sid} - {msg}") if status_callback else None)

    async def process_workflow_parallel(self, segments: List[DubbingSegment], num_workers: int, output_dir: str, status_callback=None) -> Dict:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        # 1. Group segments into sentences
        sentences = self.group_segments_into_sentences(segments)
        
        # 2. Chunk sentences for parallel workers
        if not sentences: return {}
        num_workers = min(num_workers, len(sentences))
        k, m = divmod(len(sentences), num_workers)
        sentence_chunks = [sentences[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_workers)]
        
        # 3. Process sentence chunks in parallel
        worker_tasks = [self.process_sentence_chunk(chunk, output_dir, i+1, status_callback) for i, chunk in enumerate(sentence_chunks)]
        # Use return_exceptions=True to ensure we catch errors but wait for all
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        
        # After all sentences are processed, segments are already updated via _split_sentence_audio_to_segments
        # Set final_audio_path for successful segments
        successful_count = 0
        for seg in segments:
            if seg.status == "tts_generated_adjusted":
                seg.final_audio_path = seg.tts_audio_path
                successful_count += 1

        return {
            "total": len(segments),
            "successful": successful_count,
            "segments": [{"id": s.segment_id, "start": s.start, "end": s.end, "text": s.text, "status": s.status} for s in segments]
        }

    def merge_audio_files(self, segment_list: List[DubbingSegment], output_path: str) -> bool:
        """Merge all generated audio files into a single audio file with silence padding for timing."""
        try:
            # Sort segments by segment_id to ensure correct order
            sorted_segments = sorted(segment_list, key=lambda x: x.segment_id)
            
            # Filter only segments with valid final audio paths
            valid_segments = [s for s in sorted_segments if s.final_audio_path and os.path.exists(s.final_audio_path)]
            
            if not valid_segments:
                return False

            # Create an empty audio segment
            final_audio = AudioSegment.silent(duration=0)
            
            current_time_ms = 0
            
            for seg in sorted_segments:
                target_start_ms = int(seg.start * 1000)
                
                # Add silence if there's a gap between segments
                if target_start_ms > current_time_ms:
                    silence_duration = target_start_ms - current_time_ms
                    final_audio += AudioSegment.silent(duration=silence_duration)
                    current_time_ms = target_start_ms
                
                if seg.final_audio_path and os.path.exists(seg.final_audio_path):
                    seg_audio = AudioSegment.from_file(seg.final_audio_path)
                    final_audio += seg_audio
                    current_time_ms += len(seg_audio)
                else:
                    # If segment failed, add silence for its duration to maintain timing
                    duration_ms = int(seg.duration * 1000)
                    final_audio += AudioSegment.silent(duration=duration_ms)
                    current_time_ms += duration_ms

            final_audio.export(output_path, format="mp3", bitrate="192k")
            return True
        except Exception as e:
            print(f"Error merging audio files: {e}")
            return False

    def generate_srt_content(self, segments: List[DubbingSegment]) -> str:
        """Generate SRT file content from processed segments."""
        srt_lines = []
        for i, seg in enumerate(segments):
            start_time = self._seconds_to_time(seg.start)
            end_time = self._seconds_to_time(seg.end)
            # Use adjusted_text if available, otherwise fallback to original text
            display_text = getattr(seg, 'adjusted_text', seg.text)
            srt_lines.append(f"{i+1}\n{start_time} --> {end_time}\n{display_text}\n")
        return "\n".join(srt_lines)
