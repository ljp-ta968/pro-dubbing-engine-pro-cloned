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
# from pydub.silence import detect_leading_silence, detect_trailing_silence (Not used and causing ImportError)

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

class ProDubbingEngine:
    def __init__(self, api_keys: List[str] = None, output_language: str = "my", voice_gender: str = "Male"):
        self.tolerance = 0.3  # ±0.3 seconds
        self.api_keys = api_keys if api_keys else []
        self.output_language = output_language.lower()
        self.voice_gender = voice_gender
        self.current_key_index = 0
        
        # Rate limiting state: {key: [timestamp1, timestamp2, ...]}
        self.key_usage = {key: [] for key in self.api_keys}
        self.max_rpm = 9 # Max requests per minute per key
        
        self._initialize_voice_map()

    async def _get_next_client(self):
        """Rotate through API keys and return a configured GenAI client with rate limit awareness."""
        if not self.api_keys:
            return None, None
        
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
        
        # All keys are at limit, wait a bit and try again
        print("All API keys are at rate limit. Waiting 5 seconds...")
        await asyncio.sleep(5)
        return await self._get_next_client()

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
            
            # Apply speed adjustment
            adjusted_audio = audio.speedup(playback_speed=speed_factor)
            
            # Normalize volume (optional, but good for consistency)
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

    async def text_to_srt_with_ai(self, text: str) -> str:
        """Convert custom formatted text to standard SRT using Gemini AI"""
        client, config = await self._get_next_client()
        if not client:
            return self._simple_text_to_srt(text)

        prompt = f"""
        Convert the following timestamped text into a valid SRT subtitle format.
        Input: {text}
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

    def chunk_segments_by_count(self, segments: List[DubbingSegment], num_chunks: int) -> List[List[DubbingSegment]]:
        if not segments: return []
        num_chunks = min(num_chunks, len(segments))
        k, m = divmod(len(segments), num_chunks)
        return [segments[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_chunks)]

    async def generate_tts_for_segment(self, segment: DubbingSegment, output_dir: str) -> bool:
        """Generate TTS with selected voice and gender, with iterative text rewriting and speed adjustment."""
        target_duration = segment.duration
        segment.retries = 0

        while True:
            try:
                # Get specific voice based on language and gender
                lang_voices = self.voice_map.get(self.output_language, self.voice_map["my"])
                voice = lang_voices.get(self.voice_gender, lang_voices["Male"])
                
                temp_output_path = os.path.join(output_dir, f"temp_seg_{segment.segment_id}.mp3")
                communicate = edge_tts.Communicate(segment.adjusted_text, voice)
                await communicate.save(temp_output_path)
                
                tts_duration = self._get_audio_duration(temp_output_path)
                segment.original_tts_duration = tts_duration

                if abs(tts_duration - target_duration) <= self.tolerance:
                    # Within tolerance, apply final speed adjustment
                    final_output_path = os.path.join(output_dir, f"seg_{segment.segment_id}.mp3")
                    if self._adjust_audio_speed(temp_output_path, target_duration, final_output_path):
                        segment.tts_audio_path = final_output_path
                        segment.tts_duration = self._get_audio_duration(final_output_path)
                        segment.status = "tts_generated_adjusted"
                        os.remove(temp_output_path) # Clean up temp file
                        return True
                    else:
                        segment.status = "error: final speed adjustment failed"
                        os.remove(temp_output_path)
                        return False
                else:
                    # Not within tolerance, try rewriting text
                    print(f"Segment {segment.segment_id}: TTS duration {tts_duration:.2f}s vs target {target_duration:.2f}s. Rewriting text (Attempt {segment.retries + 1})...")
                    segment.adjusted_text = await self._rewrite_text_with_ai(
                        segment.adjusted_text, target_duration, tts_duration, segment.lang
                    )
                    segment.retries += 1
                    os.remove(temp_output_path) # Clean up temp file for next attempt
                    continue # Try generating TTS again with rewritten text

            except Exception as e:
                segment.status = f"error: {e}"
                if os.path.exists(temp_output_path): os.remove(temp_output_path)
                return False
        return False

    async def process_chunk(self, chunk: List[DubbingSegment], output_dir: str):
        tasks = [self.generate_tts_for_segment(seg, output_dir) for seg in chunk]
        await asyncio.gather(*tasks)

    async def process_workflow_parallel(self, chunks: List[List[DubbingSegment]], output_dir: str) -> Dict:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        worker_tasks = [self.process_chunk(chunk, output_dir) for chunk in chunks]
        await asyncio.gather(*worker_tasks)
        all_segments = [seg for chunk in chunks for seg in chunk]
        # After all segments are processed, ensure final_audio_path is set for successful segments
        for seg in all_segments:
            if seg.status == "tts_generated_adjusted":
                seg.final_audio_path = seg.tts_audio_path

        return {
            "total": len(all_segments),
            "successful": len([s for s in all_segments if s.status == "tts_generated_adjusted"]),
            "segments": [vars(s) for s in all_segments]
        }

    def merge_audio_files(self, segment_list: List[DubbingSegment], output_path: str) -> bool:
        """Merge all generated audio files into a single audio file"""
        try:
            # Sort segments by segment_id to ensure correct order
            sorted_segments = sorted(segment_list, key=lambda x: x.segment_id)
            
            # Filter only segments with valid final audio paths
            valid_segments = [s for s in sorted_segments if s.final_audio_path and os.path.exists(s.final_audio_path)]
            
            if not valid_segments:
                return False
            
            # Load and concatenate audio files
            combined = AudioSegment.empty()
            for segment in valid_segments:
                audio = AudioSegment.from_mp3(segment.final_audio_path)
                combined += audio
            
            # Export to MP3
            combined.export(output_path, format="mp3", bitrate="192k")
            return True
        except Exception as e:
            print(f"Error merging audio files: {e}")
            return False

    def generate_srt_content(self, segment_list: List[DubbingSegment]) -> str:
        """Generate SRT content from segments"""
        srt_lines = []
        sorted_segments = sorted(segment_list, key=lambda x: x.segment_id)
        
        for idx, segment in enumerate(sorted_segments, 1):
            start_time = self._seconds_to_srt_time(segment.start)
            end_time = self._seconds_to_srt_time(segment.end)
            srt_lines.append(f"{idx}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(segment.text)
            srt_lines.append("")
        
        return "\n".join(srt_lines)

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
