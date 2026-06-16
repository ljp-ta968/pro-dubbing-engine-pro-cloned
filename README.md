# 🎙️ Pro Dubbing Engine - Myanmar TTS

This repository contains the standalone **Pro Dubbing Engine** extracted from the [text-to-tts-myanmar](https://github.com/didyouknowoneminute-bot/text-to-tts-myanmar) project. It is designed for professional-grade text-to-speech dubbing with precise timing control and parallel processing.

## ✨ Key Features
- **Timestamp-Aware Processing**: Handles input scripts with precise timing markers `[00:00 - 00:05]`.
- **Intelligent Chunking**: Splits content into groups for optimized parallel processing.
- **Parallel TTS Generation**: Generates audio for multiple segments concurrently using `edge-tts`.
- **Duration Validation**: Automatic speed adjustment to match original timing (±0.3s tolerance).
- **Multi-language Support**: Supports Myanmar (Burmese), English, Japanese, Korean, Thai, and Vietnamese.
- **AI Integration**: Optional script adjustment and translation using Google Gemini (requires API key).

## 🛠️ Technical Architecture
The core component is the `ProDubbingEngine` class, which orchestrates the entire workflow:
1. **Parse Input**: Extracts timestamps, languages, and text.
2. **Chunking**: Groups segments for parallel execution.
3. **Parallel Generation**: Uses `asyncio` and `ThreadPoolExecutor` for concurrent TTS.
4. **Validation**: Iteratively checks audio duration against target timing.
5. **Speed Normalization**: Adjusts playback speed if segments fall outside the tolerance range.

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- FFmpeg (required for audio processing and duration validation)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Usage Example
```python
from pro_dubbing_engine import ProDubbingEngine
import asyncio

async def main():
    engine = ProDubbingEngine(api_key="YOUR_GEMINI_API_KEY")
    script = "[00:00:00] Hello, welcome to the dubbing engine."
    segments = engine.parse_timestamp_script(script)
    results = await engine.process_segments_parallel(segments, output_dir="output")
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
```

## 📝 License
PhorGet