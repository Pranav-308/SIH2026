"""
Generates synthetic sample WAV audio files for testing Multilingual Voice AI pipeline.
"""

import wave
import math
import struct
from pathlib import Path

SAMPLE_AUDIO_DIR = Path(__file__).parent / "sample_audio"
SAMPLE_AUDIO_DIR.mkdir(exist_ok=True)


def create_sine_wave_audio(filename: str, duration_sec: float = 1.0, freq_hz: float = 440.0):
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    path = SAMPLE_AUDIO_DIR / filename

    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        for i in range(num_samples):
            value = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * freq_hz * i / sample_rate))
            data = struct.pack("<h", value)
            wav_file.writeframesraw(data)

    return path


def generate_sample_audio_set():
    p1 = create_sine_wave_audio("sample_english.wav", duration_sec=1.5, freq_hz=440.0)
    p2 = create_sine_wave_audio("sample_kannada.wav", duration_sec=1.5, freq_hz=523.25)
    p3 = create_sine_wave_audio("sample_hindi.wav", duration_sec=1.5, freq_hz=659.25)
    print(f"Generated 3 sample audio files in {SAMPLE_AUDIO_DIR}:\n - {p1.name}\n - {p2.name}\n - {p3.name}")


if __name__ == "__main__":
    generate_sample_audio_set()
