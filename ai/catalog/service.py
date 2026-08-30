import io
import json
import logging
from typing import Optional, List
from PIL import Image, UnidentifiedImageError

from config import Config
from exceptions import (
    MissingImageError,
    InvalidImageError,
    MissingAPIKeyError,
    AIServiceError,
    InvalidAIResponseError,
)
from schema import ProductCatalog, CatalogInput, VoiceTranscriptionResponse
from voice import VoiceService
from image_processing import ImageProcessingModule, process_product_image

logger = logging.getLogger("catalog_ai_service")

PROMPT_TEMPLATE = """You are an expert AI assistant specializing in cataloging handcrafted artisanal products in India.
Analyze the provided product image along with any notes from the artisan.

Artisan Notes / Context: {artisan_notes}

Examine the visual craft details, material texture, color palette, and cultural craft style shown in the image.
Extract and generate a structured JSON object with the following fields:
- product_name: A clear, buyer-friendly e-commerce product title.
- category: Main product category (e.g. Home Decor, Apparel, Jewelry, Pottery, Woodcraft, Leathercraft, Metalcraft, Accessories).
- description: Rich, appealing product description highlighting craftsmanship and unique features.
- materials: Array of materials identified (e.g. Clay, Brass, Cotton, Teak Wood, Silk).
- craft_type: Specific traditional Indian craft technique (e.g. Terracotta, Madhubani Painting, Bandhani, Kantha, Blue Pottery, Brassware, Chikankari, Dhokra).
- colors: List of dominant colors in the product.
- tags: List of relevant search keywords.
- confidence_score: A float between 0.00 and 1.00 indicating extraction confidence.

Return ONLY valid JSON strictly adhering to the schema.
"""


class CatalogAIService:
    """Core service for AI-driven artisan product catalog generation."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self._api_key = api_key
        self._model_name = model_name or Config.get_model_name()
        self.voice_service = VoiceService()
        self.image_processor = ImageProcessingModule()

    def _get_api_key(self) -> str:
        """Retrieves and validates API key."""
        if self._api_key:
            return self._api_key
        return Config.get_api_key(strict=True)

    def validate_and_open_image(self, image_bytes: bytes) -> Image.Image:
        """
        Validates raw image bytes and returns a PIL Image object.
        
        :raises MissingImageError: If image_bytes is empty.
        :raises InvalidImageError: If image_bytes cannot be parsed by PIL.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise MissingImageError("Product image bytes cannot be empty.")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()  # Verify image header/format integrity
            # Re-open after verify because verify() corrupts the stream for subsequent reads
            image = Image.open(io.BytesIO(image_bytes))
            return image
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
            logger.error(f"Image validation failed: {str(e)}")
            raise InvalidImageError(f"Provided data is not a valid image file: {str(e)}")

    def generate_catalog(
        self,
        image_bytes: bytes,
        artisan_description: Optional[str] = None
    ) -> ProductCatalog:
        """
        Processes a product image and optional artisan text description to generate a structured catalog.
        (Preserves 100% backwards compatibility with original API contract).
        """
        # Step 1: Validate input image
        raw_pil_image = self.validate_and_open_image(image_bytes)

        # Step 1b: Image Preparation & Enhancement Module
        pil_image = self.image_processor.process_image(raw_pil_image)

        # Step 2: Validate API key
        api_key = self._get_api_key()

        # Step 3: Format prompt
        artisan_notes = artisan_description.strip() if artisan_description and artisan_description.strip() else "None provided."
        prompt = PROMPT_TEMPLATE.format(artisan_notes=artisan_notes)

        # Step 4: Call Gemini Multimodal API using google-genai SDK
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProductCatalog,
                temperature=0.2,
            )

            response = client.models.generate_content(
                model=self._model_name,
                contents=[pil_image, prompt],
                config=config,
            )

            raw_text = response.text
            if not raw_text or not raw_text.strip():
                raise InvalidAIResponseError("Gemini API returned an empty response.")

        except (MissingAPIKeyError, MissingImageError, InvalidImageError):
            raise
        except Exception as e:
            if "GEMINI_API_KEY" in str(e):
                raise MissingAPIKeyError(str(e))
            logger.error(f"Error during Gemini API call: {str(e)}", exc_info=True)
            raise AIServiceError(f"Gemini API call failed: {str(e)}")

        # Step 5: Parse and validate JSON response with Pydantic
        try:
            cleaned_text = raw_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()

            catalog = ProductCatalog.model_validate_json(cleaned_text)
            return catalog

        except Exception as e:
            logger.error(f"Failed to parse AI output into ProductCatalog schema: {raw_text}", exc_info=True)
            raise InvalidAIResponseError(f"AI response did not match expected schema: {str(e)}")

    def generate_catalog_combined(
        self,
        image_bytes: bytes,
        artisan_description: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
        audio_mime_type: Optional[str] = None,
        language_hint: Optional[str] = None
    ) -> ProductCatalog:
        """
        Extended catalog generation supporting Image + Optional Typed Text + Optional Spoken Voice Audio.
        
        Priority/Combination Rule:
        - If both typed text and voice audio are provided:
          Transcribes audio first, then combines both inputs clearly into artisan notes:
          'Typed Description: <text> | Spoken Transcript (<lang>): <voice_text>'
        - If only voice audio is provided:
          Transcribes audio and uses 'Spoken Transcript (<lang>): <voice_text>' as primary description.
        - If only typed text is provided:
          Uses typed text directly.
        """
        combined_notes_parts = []

        # Process typed text if supplied
        if artisan_description and artisan_description.strip():
            combined_notes_parts.append(f"Typed Description: '{artisan_description.strip()}'")

        # Process voice audio if supplied
        if audio_bytes and len(audio_bytes) > 0:
            mime = audio_mime_type or "audio/wav"
            transcription: VoiceTranscriptionResponse = self.voice_service.transcribe_audio(
                audio_bytes=audio_bytes,
                mime_type=mime,
                language_hint=language_hint
            )
            combined_notes_parts.append(
                f"Spoken Audio Transcript ({transcription.detected_language}): '{transcription.transcribed_text}'"
            )

        final_combined_description = " | ".join(combined_notes_parts) if combined_notes_parts else None

        return self.generate_catalog(
            image_bytes=image_bytes,
            artisan_description=final_combined_description
        )
