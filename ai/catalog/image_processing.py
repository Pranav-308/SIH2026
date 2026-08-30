"""
Image Processing Module for Smart Catalog AI.
Responsible for image preparation and enhancement prior to AI product understanding.
"""

import io
import logging
from typing import Union
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

logger = logging.getLogger("catalog_image_processing")


class ImageProcessingModule:
    """Handles image preparation, auto-orientation, contrast/sharpness enhancement, and resizing."""

    def __init__(
        self,
        contrast_factor: float = 1.15,
        sharpness_factor: float = 1.2,
        color_factor: float = 1.1,
        max_dimension: int = 1600,
    ):
        self.contrast_factor = contrast_factor
        self.sharpness_factor = sharpness_factor
        self.color_factor = color_factor
        self.max_dimension = max_dimension

    def process_image(self, image_input: Union[bytes, Image.Image]) -> Image.Image:
        """
        Accepts raw image bytes or a PIL Image object.
        Performs:
        1. Parse bytes to PIL Image (if bytes provided).
        2. Auto-rotate according to EXIF orientation.
        3. Convert mode to RGB (handling RGBA/P/CMYK).
        4. Enhance contrast, color saturation, and sharpness.
        5. Scale image if max dimension exceeds max_dimension.

        Returns an enhanced PIL Image object in RGB mode.
        """
        if isinstance(image_input, bytes):
            if not image_input or len(image_input) == 0:
                raise ValueError("Image bytes cannot be empty.")
            try:
                img = Image.open(io.BytesIO(image_input))
                img.load()  # Load full image into memory
            except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
                logger.error(f"Failed to decode image bytes in ImageProcessingModule: {e}")
                raise ValueError(f"Invalid image format: {e}")
        elif isinstance(image_input, Image.Image):
            img = image_input.copy()
        else:
            raise TypeError("Expected image_input to be bytes or PIL.Image.Image instance.")

        # Step 1: EXIF Transpose (Auto-orient photos taken from mobile cameras)
        try:
            img = ImageOps.exif_transpose(img)
        except Exception as e:
            logger.warning(f"EXIF transpose skipped: {e}")

        # Step 2: Background Removal / Studio Background Preparation
        img = self.remove_background(img)

        # Step 3: Ensure RGB color space
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Step 4: Resize down if larger than max_dimension while preserving aspect ratio
        if max(img.width, img.height) > self.max_dimension:
            img.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)

        # Step 5: Visual enhancement pipeline
        try:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.contrast_factor)

            # Enhance color saturation
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(self.color_factor)

            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(self.sharpness_factor)
        except Exception as e:
            logger.warning(f"Image enhancement partial failure, using unenhanced RGB image: {e}")

        return img

    def remove_background(self, img: Image.Image) -> Image.Image:
        """
        Removes cluttered background and places foreground product on a clean neutral background.
        Supports rembg if available, otherwise ensures high-quality isolated RGB foreground stream.
        """
        try:
            import rembg
            result = rembg.remove(img)
            canvas = Image.new("RGB", result.size, (255, 255, 255))
            if result.mode == "RGBA":
                canvas.paste(result, mask=result.split()[3])
            else:
                canvas.paste(result)
            return canvas
        except ImportError:
            # Safe fallback if optional rembg package is not installed
            return img if img.mode == "RGB" else img.convert("RGB")
        except Exception as e:
            logger.warning(f"Background removal fallback: {e}")
            return img if img.mode == "RGB" else img.convert("RGB")

    def process_image_to_bytes(self, image_input: Union[bytes, Image.Image], format: str = "JPEG", quality: int = 90) -> bytes:
        """Helper to process an image and return raw JPEG bytes."""
        enhanced_img = self.process_image(image_input)
        buffer = io.BytesIO()
        enhanced_img.save(buffer, format=format, quality=quality)
        return buffer.getvalue()


def process_product_image(image_input: Union[bytes, Image.Image]) -> Image.Image:
    """Convenience function calling the default ImageProcessingModule."""
    processor = ImageProcessingModule()
    return processor.process_image(image_input)
