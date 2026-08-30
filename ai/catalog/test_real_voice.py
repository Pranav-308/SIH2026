"""
CLI Tool for testing Real-World Multilingual Voice AI and Combined Voice+Image Catalog Pipeline.

Usage:
  1. Test Voice STT Only:
     python test_real_voice.py --audio path/to/artisan_voice.wav --language kn-IN

  2. Test Combined Voice + Image Catalog:
     python test_real_voice.py --audio path/to/artisan_voice.wav --image path/to/product.jpg --description "Optional text note"
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure ai/catalog is in import path
sys.path.insert(0, str(Path(__file__).parent))

__test__ = False


from voice import VoiceService
from service import CatalogAIService
from exceptions import CatalogAIException


def test_voice_only(audio_path_str: str, language_hint: str = None):
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        print(f"❌ Error: Audio file not found at '{audio_path}'")
        sys.exit(1)

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    except Exception as e:
        print(f"❌ Error reading audio file: {str(e)}")
        sys.exit(1)

    ext = audio_path.suffix.lower()
    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".m4a": "audio/m4a",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
        ".mpeg": "audio/mpeg",
        ".mp4": "audio/mp4",
    }
    mime_type = mime_map.get(ext, "audio/wav")

    print("\n==================================================")
    print("🎙️ MULTILINGUAL VOICE TRANSCRIPTION TEST")
    print("==================================================")
    print(f"Audio File:    {audio_path.name}")
    print(f"Format/MIME:   {mime_type}")
    print(f"Language Hint: {language_hint if language_hint else 'Automatic Detection'}")
    print("--------------------------------------------------")
    print("⏳ Transcribing audio with Gemini Speech-to-Text Engine...")

    try:
        voice_service = VoiceService()
        result = voice_service.transcribe_audio(
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            language_hint=language_hint
        )

        res_dict = result.model_dump()

        print("\n✅ TRANSCRIPTION SUCCESSFUL:")
        print("--------------------------------------------------")
        print(f"🗣️ Transcribed Text:  \"{result.transcribed_text}\"")
        print(f"🌐 Detected Language: {result.detected_language}")
        print(f"📊 Confidence Score:  {result.confidence if result.confidence is not None else 'N/A'}")
        print("--------------------------------------------------")

        return res_dict

    except CatalogAIException as e:
        print(f"\n❌ Voice AI Exception [{e.status_code}]: {e.message}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {str(e)}")
        sys.exit(1)


def test_combined_voice_and_image(
    audio_path_str: str,
    image_path_str: str,
    description: str = "",
    language_hint: str = None
):
    audio_path = Path(audio_path_str)
    image_path = Path(image_path_str)

    if not audio_path.exists():
        print(f"❌ Error: Audio file not found at '{audio_path}'")
        sys.exit(1)
    if not image_path.exists():
        print(f"❌ Error: Image file not found at '{image_path}'")
        sys.exit(1)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    ext = audio_path.suffix.lower()
    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".m4a": "audio/m4a",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
        ".mpeg": "audio/mpeg",
        ".mp4": "audio/mp4",
    }
    mime_type = mime_map.get(ext, "audio/wav")

    print("\n==================================================")
    print("🎨 COMBINED VOICE + IMAGE CATALOG PIPELINE TEST")
    print("==================================================")
    print(f"Product Image: {image_path.name}")
    print(f"Voice Audio:   {audio_path.name}")
    print(f"Typed Text:    {description if description else '(None)'}")
    print("--------------------------------------------------")
    print("⏳ Processing pipeline (Voice Transcription -> Combined Prompt -> Vision Catalog)...")

    try:
        service = CatalogAIService()
        catalog = service.generate_catalog_combined(
            image_bytes=image_bytes,
            artisan_description=description,
            audio_bytes=audio_bytes,
            audio_mime_type=mime_type,
            language_hint=language_hint
        )

        cat_dict = catalog.model_dump()

        print("\n✅ COMBINED CATALOG GENERATION SUCCESSFUL:")
        print("--------------------------------------------------")
        print(json.dumps(cat_dict, indent=2, ensure_ascii=False))
        print("--------------------------------------------------")

        return cat_dict

    except CatalogAIException as e:
        print(f"\n❌ Catalog AI Exception [{e.status_code}]: {e.message}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test Multilingual Voice STT & Combined Catalog Pipeline.")
    parser.add_argument("--audio", type=str, required=True, help="Path to artisan voice audio file (.wav, .mp3, .m4a).")
    parser.add_argument("--image", type=str, default="", help="Optional path to product image file.")
    parser.add_argument("--description", type=str, default="", help="Optional typed text description.")
    parser.add_argument("--language", type=str, default="", help="Optional language hint (e.g. kn-IN, hi-IN, en-US).")
    args = parser.parse_args()

    if args.image:
        test_combined_voice_and_image(
            audio_path_str=args.audio,
            image_path_str=args.image,
            description=args.description,
            language_hint=args.language
        )
    else:
        test_voice_only(
            audio_path_str=args.audio,
            language_hint=args.language
        )


if __name__ == "__main__":
    main()
