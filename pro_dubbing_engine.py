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
import google.generativeai as genai
from pydub import AudioSegment
import io

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

class ProDubbingEngine:
    def __init__(self, api_key: str = None, output_language: str = "my", voice_gender: str = "Male"):
        self.tolerance = 0.3  # ±0.3 seconds
        self.api_key = api_key
        self.output_language = output_language.lower()
        self.voice_gender = voice_gender
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
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
        if not self.api_key:
            return self._simple_text_to_srt(text)

        prompt = f"""
        Convert the following timestamped text into a valid SRT subtitle format.
        Input: {text}
        """
        try:
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            return response.text.strip()
        except:
            return self._simple_text_to_srt(text)

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
        """Generate TTS with selected voice and gender"""
        try:
            # Get specific voice based on language and gender
            lang_voices = self.voice_map.get(self.output_language, self.voice_map["my"])
            voice = lang_voices.get(self.voice_gender, lang_voices["Male"])
            
            output_path = os.path.join(output_dir, f"seg_{segment.segment_id}.mp3")
            communicate = edge_tts.Communicate(segment.text, voice)
            await communicate.save(output_path)
            
            segment.tts_audio_path = output_path
            segment.status = "tts_generated"
            # Simple duration estimation
            segment.tts_duration = len(segment.text.split()) / 2.5
            return True
        except Exception as e:
            segment.status = f"error: {e}"
            return False

    async def process_chunk(self, chunk: List[DubbingSegment], output_dir: str):
        tasks = [self.generate_tts_for_segment(seg, output_dir) for seg in chunk]
        await asyncio.gather(*tasks)

    async def process_workflow_parallel(self, chunks: List[List[DubbingSegment]], output_dir: str) -> Dict:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        worker_tasks = [self.process_chunk(chunk, output_dir) for chunk in chunks]
        await asyncio.gather(*worker_tasks)
        all_segments = [seg for chunk in chunks for seg in chunk]
        return {
            "total": len(all_segments),
            "successful": len([s for s in all_segments if "error" not in s.status]),
            "segments": [vars(s) for s in all_segments]
        }

    def merge_audio_files(self, segment_list: List[DubbingSegment], output_path: str) -> bool:
        """Merge all generated audio files into a single audio file"""
        try:
            # Sort segments by segment_id to ensure correct order
            sorted_segments = sorted(segment_list, key=lambda x: x.segment_id)
            
            # Filter only segments with valid audio paths
            valid_segments = [s for s in sorted_segments if s.tts_audio_path and os.path.exists(s.tts_audio_path)]
            
            if not valid_segments:
                return False
            
            # Load and concatenate audio files
            combined = AudioSegment.empty()
            for segment in valid_segments:
                audio = AudioSegment.from_mp3(segment.tts_audio_path)
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
